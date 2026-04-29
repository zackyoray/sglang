from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode

try:
    from nixl_ep import Buffer

    use_nixl = True
except ImportError:
    use_nixl = False

logger = logging.getLogger(__name__)

NixlEPDispatchOutput = DeepEPLLDispatchOutput
NixlEPCombineInput = DeepEPLLCombineInput


class NixlEPBuffer:
    _buffer = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None
    _num_local_experts: Optional[int] = None
    # Exclusive upper bound of EP rank indices already connected via connect_ranks.
    _connected_ep_size: Optional[int] = None
    # Target EP frontier for NIXL mesh: set at first buffer init, then updated only
    # via on_scale() from ElasticEPStateManager._on_scale_nixl after new ranks join
    # (same timing as RFC — not bumped from HTTP scale alone).
    _scale_to: Optional[int] = None

    # Expert ID remapping tables (following vLLM's approach).
    # NIXL routes: rank = physical_id // (nixl_num_experts // num_connected_ranks).
    # We pass nixl_num_experts = num_local_experts * ep_size to dispatch so each
    # rank gets exactly num_local_experts slots. The remapping packs global router
    # IDs [0..num_model_experts-1] into contiguous per-rank blocks of num_local_experts.
    _global_to_physical: Optional[torch.Tensor] = None
    _physical_to_global: Optional[torch.Tensor] = None
    _ep_size: Optional[int] = None

    @classmethod
    def _build_routing_tables(cls, num_experts: int, ep_size: int) -> None:
        """Build global↔physical expert ID mapping for NIXL dispatch.

        Each rank owns a contiguous block of num_local_experts physical IDs.
        nixl_num_experts = num_local_experts * ep_size is passed to dispatch,
        so NIXL routes: rank = physical_id // num_local_experts.

        Global expert i is owned by rank (i % ep_size) and is the
        (i // ep_size)-th expert on that rank:
          physical_id = owner_rank * num_local_experts + local_index

        Pre-scale (ep_size=4):  physical range [0..95],  nixl_num_experts=96
        Post-scale (ep_size=8): physical range [0..179], nixl_num_experts=192
        """
        num_local = cls._num_local_experts
        device = "cuda"
        g = torch.arange(num_experts, dtype=torch.long, device=device)
        owner = g % ep_size
        local_idx = g // ep_size
        physical = owner * num_local + local_idx

        nixl_total = num_local * ep_size
        cls._global_to_physical = physical
        cls._physical_to_global = torch.zeros(
            nixl_total, dtype=torch.long, device=device
        )
        cls._physical_to_global[physical] = g
        cls._ep_size = ep_size
        logger.info(
            "[Elastic EP][nixl] built routing tables: num_experts=%d ep_size=%d "
            "num_local=%d nixl_num_experts=%d "
            "(sample: global[0,1,2,3]→physical%s)",
            num_experts, ep_size, num_local, nixl_total,
            physical[:4].tolist(),
        )

    @classmethod
    def map_global_to_physical(cls, topk_ids: torch.Tensor) -> torch.Tensor:
        if cls._global_to_physical is None:
            return topk_ids
        mask = topk_ids >= 0
        result = torch.where(mask, cls._global_to_physical[topk_ids.clamp(min=0)], topk_ids)
        return result

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        """Called from ElasticEPStateManager._on_scale_nixl after activate_ranks."""
        cls._scale_to = to_ep_size
        if cls._num_experts is not None:
            cls._build_routing_tables(cls._num_experts, to_ep_size)
        logger.info(
            "[Elastic EP][nixl] on_scale(%s -> %s) _scale_to=%s",
            from_ep_size,
            to_ep_size,
            to_ep_size,
        )

    @classmethod
    def _update_connections(cls, scale_to: int) -> None:
        """connect_ranks(range(_connected_ep_size, scale_to)); caller ensures scale_to > _connected_ep_size."""
        new_ranks = list(range(cls._connected_ep_size, scale_to))
        logger.info(
            "[Elastic EP][nixl] update_connections connect_ranks(%s) "
            "(_connected_ep_size -> %s)",
            new_ranks,
            scale_to,
        )
        cls._buffer.connect_ranks(new_ranks)
        cls._connected_ep_size = scale_to
        logger.info(
            "[Elastic EP][nixl] after connect_ranks: buffer.group_size=%s",
            cls._buffer.group_size,
        )

    @classmethod
    def get_nixl_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
        num_local_experts: int = -1,
    ):
        if cls._buffer is not None:
            if (
                cls._scale_to is not None
                and cls._connected_ep_size is not None
                and cls._scale_to > cls._connected_ep_size
            ):
                cls._update_connections(cls._scale_to)
            return cls._buffer

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts
        cls._num_local_experts = num_local_experts

        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        # Joiners with --ep-join-rank-offset N use global EP rank r + N.
        offset = ElasticEPStateManager.get_ep_join_rank_offset()
        global_rank = rank + offset

        # For elastic EP, NIXL needs to allocate buffers for the maximum EP
        # size up front because update_memory_buffers cannot resize at
        # runtime. Following vLLM's approach (NixlEPAll2AllManager._init_buffer
        # in vllm/distributed/device_communicators/all2all.py), we keep
        # num_experts_per_rank fixed and grow num_experts to match the
        # max-sized rank table:
        #     max_num_global_experts = max_ep_size * num_experts_per_rank
        # so the invariant num_ranks * num_experts_per_rank == num_experts
        # holds at the NIXL layer regardless of how many ranks are live.
        # On scale-up, new ranks claim their num_experts_per_rank slice from
        # the already-allocated pool and connect via _update_connections.
        from sglang.srt.server_args import get_global_server_args

        max_ep_size = get_global_server_args().max_ep_size or world_size

        num_rdma_bytes = 0
        if deepep_mode.enable_normal():
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            max_num_global_experts = max_ep_size * num_local_experts
            num_rdma_bytes = Buffer.get_rdma_size_hint(
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                max_ep_size,
                max_num_global_experts,
            )

        # Get the global TCPStore for coordination
        tcp_store = get_global_tcp_store()
        if tcp_store is None:
            raise RuntimeError(
                "Global TCPStore is not initialized. "
                "Make sure init_distributed_environment was called before using NIXL EP."
            )

        logger.info(
            f"Using NIXL EP (world_size={world_size}, max_ep_size={max_ep_size}, "
            f"rank={rank}, global_rank={global_rank}, offset={offset}, "
            f"num_experts={cls._num_experts}, "
            f"num_experts_per_rank={cls._num_local_experts}) "
        )

        cls._buffer = Buffer(
            rank=global_rank,
            tcp_store_group=tcp_store,
        )

        cls._buffer.update_memory_buffers(
            num_ranks=max_ep_size,
            num_experts_per_rank=cls._num_local_experts,
            num_rdma_bytes=num_rdma_bytes,
        )
        # Initial mesh: connect to all known live ranks.
        # Primary (offset=0): connects to [0..world_size-1].
        # Joiner (offset=N): connects to [0..offset+world_size-1] so it can
        # exchange tokens with the primary ranks as well as its own peers.
        # Further growth: _scale_to vs _connected_ep_size on each get_nixl_buffer
        # after on_scale() sets _scale_to.
        live_ranks = list(range(offset + world_size))
        scale_to = offset + world_size
        logger.info(
            "[Elastic EP][nixl] initial connect_ranks(%s) "
            "(world_size=%s, offset=%s, max_ep_size=%s)",
            live_ranks,
            world_size,
            offset,
            max_ep_size,
        )
        cls._buffer.connect_ranks(live_ranks)
        cls._connected_ep_size = scale_to
        cls._scale_to = scale_to

        cls._build_routing_tables(num_experts, world_size)

        return cls._buffer

    @classmethod
    def clean_buffer(cls):
        cls._buffer.clean_buffer(
            cls._num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )


class _NixlEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_nixl:
            raise ImportError(
                "NixlEP is not installed. Please install NixlEP package from "
                "https://github.com/ai-dynamo/nixl."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        # NixlEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        assert self.num_max_dispatch_tokens_per_rank <= 1024
        elastic_state = ElasticEPStateManager.instance()
        self.active_ranks = (
            elastic_state.active_ranks if elastic_state is not None else None
        )
        # NIXL's query_mask_buffer requires the mask tensor numel to equal
        # the buffer's max_num_ranks (which we pre-allocate to max_ep_size).
        # So the mask AND active_ranks must be sized to max_ep_size. NIXL
        # populates only the live-world slots; the rest get a sentinel
        # (not 0), so we track the live portion separately to preserve the
        # "reserved slots stay at 0" invariant required by is_scaling().
        self._active_world_size = dist.get_world_size(group)
        # Joiner with --ep-join-rank-offset N writes its dispatcher mask into
        # active_ranks[N : N + world_size] rather than [:world_size].
        self._active_rank_offset = ElasticEPStateManager.get_ep_join_rank_offset()
        self._mask_buffer = (
            torch.zeros_like(self.active_ranks)
            if self.active_ranks is not None
            else None
        )

        self.handle = None
        self.quant_config = None
        self.overlap_args = None
        self.meta_overlap_args = None

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError


class _NixlEPDispatcherImpl(_NixlEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/ai-dynamo/nixl
        """
        self.return_recv_hook = return_recv_hook
        self.device_module = torch.get_device_module()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        buffer = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        ep_size = NixlEPBuffer._ep_size or buffer.group_size
        expected_m = (
            hidden_states.shape[0] * ep_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        dispatch_topk_ids = NixlEPBuffer.map_global_to_physical(topk_ids)
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            dispatch_topk_ids,
        )
        return (
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        nixl_output = NixlEPDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return nixl_output

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
    ):
        use_fp8 = not envs.SGLANG_NIXL_EP_BF16_DISPATCH.get()

        buffer = self._get_buffer()
        _cep = NixlEPBuffer._connected_ep_size
        if not hasattr(self, "_last_logged_cep") or self._last_logged_cep != _cep:
            logger.info(
                "[Elastic EP][nixl] dispatch group_size=%s "
                "(connected_ep_size=%s, scale_to=%s)",
                buffer.group_size,
                _cep,
                NixlEPBuffer._scale_to,
            )
            self._last_logged_cep = _cep
        # nixl_num_experts = num_local_experts * ep_size so NIXL routes
        # num_local_experts per rank: pre-scale 24*4=96, post-scale 24*8=192.
        nixl_num_experts = NixlEPBuffer._num_local_experts * NixlEPBuffer._ep_size
        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
            buffer.dispatch(
                hidden_states,
                topk_idx,
                self.num_max_dispatch_tokens_per_rank,
                nixl_num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            )
        )
        return packed_recv_hidden, self.packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        combine_topk_ids = NixlEPBuffer.map_global_to_physical(topk_ids)
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            combine_topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        hook() if self.return_recv_hook else event.current_stream_wait()
        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()

        combined_hidden_states, event, hook = buffer.combine(
            x=hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            handle=self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
        if self._mask_buffer is not None:
            buffer.query_mask_buffer(self._mask_buffer)
            # Only update the live-world slots from NIXL's mask. NIXL fills
            # reserved slots (max_ep_size > world_size) with a sentinel that
            # would corrupt is_scaling() / EPLB if we wrote it through.
            n = self._active_world_size
            off = self._active_rank_offset
            self.active_ranks[off : off + n].copy_(
                1 - self._mask_buffer[off : off + n]
            )

        self.packed_recv_count = self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        return NixlEPBuffer.get_nixl_buffer(
            self.group,
            self.hidden_size,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            self.num_local_experts,
        )


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class NixlEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.LOW_LATENCY,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _NixlEPDispatcherImpl(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")

        self._stage = _Stage.INITIAL

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states=hidden_states, topk_output=topk_output)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _NixlEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage
