from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import List, Optional

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

    # ep_size for nixl_num_experts computation. Updated on scale.
    # nixl_num_experts = num_local_experts * _ep_size ensures each rank gets
    # exactly num_local_experts slots in the NIXL dispatch output.
    # No ID remapping needed: SGLang uses contiguous expert assignment
    # (experts 0-23 on rank 0, 24-47 on rank 1, etc.) and the router's
    # topk_ids naturally fall into the correct rank blocks.
    _ep_size: Optional[int] = None

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        """Called from ElasticEPStateManager._on_scale_nixl after activate_ranks."""
        cls._scale_to = to_ep_size
        cls._ep_size = to_ep_size
        logger.info(
            "[Elastic EP][nixl] on_scale(%s -> %s) _scale_to=%s "
            "nixl_num_experts=%s",
            from_ep_size,
            to_ep_size,
            to_ep_size,
            cls._num_local_experts * to_ep_size if cls._num_local_experts else None,
        )

    @classmethod
    def _peek_tcp_store_keys(
        cls, ranks: List[int], *, tag: str, phase: str
    ) -> None:
        """Non-blocking peek of NIXL_EP/{rank} keys in the global TCPStore.

        Idea 2 from session-9 follow-up: confirm whether the metadata each
        peer is supposed to write to TCPStore as part of its connect_ranks
        is actually present at the moment we're about to read it.

        - phase="pre" (just before our connect_ranks call): the peer keys
          for `ranks` should be present (or about to appear). MISSING here
          means the peer hasn't entered its connect_ranks yet (barrier
          ordering bug) or already finished + deleted them (ditto).
        - phase="post" (just after our connect_ranks): all peer keys for
          `ranks` should now be GONE (each peer deletes its key in the
          finally of _fetch_remote_metadata_from_tcp_store), and our own
          NIXL_EP/{self.rank} key should also be GONE. If anything is
          still present, the handshake didn't run cleanly on some side.
        """
        try:
            store = cls._buffer.tcp_store_group
        except Exception as _e:
            logger.warning(
                "[Elastic EP][nixl][peek] %s/%s: buffer has no tcp_store_group: %s",
                tag, phase, _e,
            )
            return
        if store is None:
            logger.warning(
                "[Elastic EP][nixl][peek] %s/%s: tcp_store_group is None", tag, phase,
            )
            return
        # Include own key in post-phase to verify our finally-delete ran.
        keys = [f"NIXL_EP/{r}" for r in ranks if r != cls._buffer.rank]
        if phase == "post":
            keys.append(f"NIXL_EP/{cls._buffer.rank}")
        if not keys:
            return
        present = []
        for k in keys:
            try:
                exists = bool(store.check([k]))
            except Exception as _e:
                exists = f"err({_e})"
            present.append((k, exists))
        logger.info(
            "[Elastic EP][nixl][peek] %s/%s tcpstore keys: %s",
            tag, phase, present,
        )

    @classmethod
    def _sync_connect_ranks(cls, ranks: list, *, tag: str) -> None:
        """Run buffer.connect_ranks(ranks) in lockstep across all active EP ranks.

        NIXL's connect_ranks is a two-sided TCPStore handshake: each side
        writes its own metadata, reads the peers', and registers RDMA
        memory descriptors. For that exchange to populate peer info
        correctly on both sides, BOTH sides must be inside connect_ranks
        at overlapping times. Without an explicit sync, primary hits
        update_connections on its first post-scale dispatch while the
        joiner is still in DeepGEMM warmup / kernel JIT — the two calls
        can be 20+ seconds apart, and the "successful" return on each
        side leaves late-added peer info in an inconsistent state,
        producing silent dispatch-receive timeouts (`masked_m_sum=0`).

        We wrap the call with a barrier on the Mooncake WORLD group,
        which honors `active_ranks` — pre-scale it spans only the
        primary's active ranks (cheap no-op), post-scale it spans all
        primary+joiner ranks (which is exactly the sync we need).

        The post-call barrier guards the reverse failure: a rank that
        returns from its local handshake and jumps into dispatch while
        a peer is still inside connect_ranks would try to RDMA against
        an un-finalized peer memory descriptor.
        """
        from sglang.srt.distributed.parallel_state import get_world_group

        # Optional bypass of the Mooncake WORLD barriers around
        # connect_ranks. NIXL's reference test (`tests/elastic/elastic.py`)
        # has NO such barriers and works cleanly. We added them because
        # without an explicit sync, primary↔joiner enter `connect_ranks`
        # ~22 s apart on our setup (joiner spawns then loads model
        # weights for many seconds before reaching its first dispatch),
        # and TCPStore peer-key races can occur (peer writes key, runs
        # connect, deletes key — all before the other side sees it).
        #
        # The barriers fix that race but introduce another variable
        # (Mooncake collective sync interleaved with NIXL's TCPStore
        # handshake). Toggle via SGLANG_NIXL_NO_BARRIER=1 to test
        # whether the barriers themselves are causing the post-scale
        # primary→joiner delivery failure. Default off (=0) keeps the
        # current barrier behavior.
        import os as _os
        _no_barrier = (
            _os.environ.get("SGLANG_NIXL_NO_BARRIER", "0") == "1"
        )

        world_group = get_world_group().device_group
        logger.info(
            "[Elastic EP][nixl] sync-connect (%s) pre-barrier WORLD "
            "(env SGLANG_NIXL_NO_BARRIER=%s)",
            tag,
            _os.environ.get("SGLANG_NIXL_NO_BARRIER", "0"),
        )
        if not _no_barrier:
            torch.distributed.barrier(group=world_group)

        # Idea 5 (vLLM parity, defensive): re-bind the TCPStore reference
        # before each connect_ranks. In SGLang the global store doesn't
        # change across scale events so this is a no-op; in vLLM the
        # All2AllManager is recreated per scale event and the new
        # tcp_store_group must be re-attached. Matching the call here
        # eliminates one variable in cross-implementation comparison.
        try:
            current_store = get_global_tcp_store()
            if current_store is not None:
                cls._buffer.set_tcp_store_group(current_store)
        except Exception as _e:
            logger.warning(
                "[Elastic EP][nixl] set_tcp_store_group(global) failed: %s", _e,
            )

        # Idea 2: peek before the handshake. Expect peer keys to be
        # present (or about to appear within ms after the entry barrier).
        cls._peek_tcp_store_keys(ranks, tag=tag, phase="pre")

        connect_t0 = time.perf_counter()
        cls._buffer.connect_ranks(ranks)
        connect_dt_ms = (time.perf_counter() - connect_t0) * 1000.0

        logger.info(
            "[Elastic EP][nixl] sync-connect (%s) connect_ranks(%s) done in "
            "%.1f ms; post-barrier WORLD (buffer.group_size=%s, "
            "buffer.num_ranks=%s)",
            tag,
            ranks,
            connect_dt_ms,
            cls._buffer.group_size,
            getattr(cls._buffer, "num_ranks", "N/A"),
        )

        # Idea 2 (post-phase): peer keys should now be GONE, and so
        # should our own. Anything still present indicates a handshake
        # path that returned without running its finally-delete.
        cls._peek_tcp_store_keys(ranks, tag=tag, phase="post")

        if not _no_barrier:
            torch.distributed.barrier(group=world_group)

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
        cls._sync_connect_ranks(new_ranks, tag="update")
        cls._connected_ep_size = scale_to

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
        # Size the buffer for the actual scale-up frontier — NOT inflated to
        # 32 like the older code did.
        #
        # Background: the previous version used `max(max_ep_size, 32)` to
        # match vLLM's `VLLM_NIXL_EP_MAX_NUM_RANKS=32` constant. That was
        # only necessary against a buggy NIXL where `dispatch()`/`combine()`
        # sized the EPLayout from the runtime `(num_ranks, num_experts)`
        # args, so a larger buffer slot count was a workaround for the
        # signaling-region overlap. Since NIXL `8e32438` (PR #1451), the
        # EPLayout is sized from the buffer's `(max_num_ranks,
        # max_experts_per_rank)` registered at `update_memory_buffers` —
        # so over-sizing now reserves a 32×N rdma_buffer with placeholder
        # `nixl_null_agent` slots for ranks 8..31, which the LL recv
        # kernel still strides over. Empirically that produces the
        # 384-timeout (every primary↔joiner pair) symptom we chased for
        # several sessions.
        #
        # NIXL's own `tests/elastic/elastic.py` reference test passes
        # cleanly when `max_num_ranks` matches the actual scale-up target
        # (`plan.get_max_rank() + 1`), with the same `(num_experts_per_rank
        # =24, hidden=2560, num_topk=6)` config as us. So match that
        # convention.
        nixl_max_ranks = max_ep_size

        num_rdma_bytes = 0
        if deepep_mode.enable_normal():
            raise NotImplementedError("Normal mode is not supported for Nixl EP yet.")
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            max_num_global_experts = nixl_max_ranks * num_local_experts
            num_rdma_bytes = Buffer.get_rdma_size_hint(
                num_max_dispatch_tokens_per_rank,
                hidden_size,
                nixl_max_ranks,
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

        # Match NIXL test (`tests/elastic/elastic.py`) Buffer kwargs.
        # NIXL's reference test (which we verified works in this exact
        # container with our exact dispatch params) passes
        # `explicitly_destroy=True` and an explicit `timeout_ms`. We had
        # been using bare-defaults (`explicitly_destroy=False`,
        # `low_latency_mode=True`, `timeout_ms=30000`). Default behavior
        # for `explicitly_destroy=False` is destructor-based cleanup at
        # GC time — the wrapper's docstring warns this can hang Python's
        # exception path and may leave NIXL agent state in a half-torn
        # configuration during long-lived primary processes. Matching
        # the reference test exactly eliminates this as a variable for
        # the post-scale "primary->joiner silently fails" symptom.
        #
        # Behind env var SGLANG_NIXL_EXPLICIT_DESTROY=1 (default 0) so
        # we can A/B isolate whether this is the cause without touching
        # other call paths.
        import os as _os
        _explicit_destroy = (
            _os.environ.get("SGLANG_NIXL_EXPLICIT_DESTROY", "0") == "1"
        )
        _buffer_kwargs = dict(
            rank=global_rank,
            tcp_store_group=tcp_store,
        )
        if _explicit_destroy:
            _buffer_kwargs["explicitly_destroy"] = True
            _buffer_kwargs["timeout_ms"] = 30_000
        logger.info(
            "[Elastic EP][nixl] Buffer kwargs: %s (env "
            "SGLANG_NIXL_EXPLICIT_DESTROY=%s)",
            sorted(_buffer_kwargs.keys()),
            _os.environ.get("SGLANG_NIXL_EXPLICIT_DESTROY", "0"),
        )
        cls._buffer = Buffer(**_buffer_kwargs)

        cls._buffer.update_memory_buffers(
            num_ranks=nixl_max_ranks,
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
        cls._sync_connect_ranks(live_ranks, tag="initial")
        cls._connected_ep_size = scale_to
        cls._scale_to = scale_to

        cls._ep_size = scale_to

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
        # NIXL's query_mask_buffer requires
        # `mask_status.numel() == max_num_ranks` registered via
        # update_memory_buffers — which we now set equal to `max_ep_size`
        # (no 32-cap, see get_nixl_buffer for rationale).
        self._active_world_size = dist.get_world_size(group)
        self._active_rank_offset = ElasticEPStateManager.get_ep_join_rank_offset()
        from sglang.srt.server_args import get_global_server_args
        _max_ep = get_global_server_args().max_ep_size or self._active_world_size
        self._mask_buffer = (
            torch.zeros(_max_ep, dtype=torch.int32, device="cuda")
            if self.active_ranks is not None
            else None
        )
        # Optional: skip NIXL mask query + active_ranks mutation in the
        # combine path. This disables the fault-tolerance signal (NIXL's
        # `query_mask_buffer` returns "this rank looks faulted" 0/1
        # values that we propagate into `active_ranks`). The downstream
        # consumer is the scheduler's "EPLB due to rank faults" trigger
        # at `model_runner.py:3354` — it fires whenever
        # `is_active_equal_last()` becomes False, which a single mask
        # flip in the combine path causes immediately. That trigger then
        # calls `eplb_manager.rebalance()` (collective dump_record + P2P)
        # in the middle of the next forward pass, racing with our
        # late-add NIXL dispatch — which is the dominant cause of the
        # post-scale 380+ NIXL-EP timeouts.
        #
        # Use SGLANG_NIXL_SKIP_FAULT_MASK=1 to disable the mask read
        # for elastic-EP scale-up correctness testing. Cost: no fault
        # tolerance (any rank death is now silent).
        import os as _os
        self._skip_fault_mask = (
            _os.environ.get("SGLANG_NIXL_SKIP_FAULT_MASK", "0") == "1"
        )
        # Track previous mask snapshot so we only log mask transitions
        # (flips), not the every-combine state. This avoids spam.
        self._prev_mask_snapshot: Optional[List[int]] = None
        self._mask_log_count = 0

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

        _ep = NixlEPBuffer._ep_size
        if getattr(self, "_shapes_logged_ep", None) != _ep:
            logger.info(
                "[SHAPES] dispatch_a INPUT: hidden_states=%s topk_ids=%s "
                "ep_size=%d num_experts=%d num_local_experts=%d "
                "nixl_num_experts=%d num_max_dispatch=%d "
                "topk_min=%d topk_max=%d",
                list(hidden_states.shape), list(topk_ids.shape),
                ep_size, self.num_experts, self.num_local_experts,
                NixlEPBuffer._num_local_experts * _ep,
                self.num_max_dispatch_tokens_per_rank,
                topk_ids[topk_ids >= 0].min().item() if topk_ids.numel() > 0 and (topk_ids >= 0).any() else -1,
                topk_ids.max().item() if topk_ids.numel() > 0 else -1,
            )

        # Idea 1: per-target-rank token histogram for the first few
        # dispatches after each ep_size change. This is what tells us
        # whether the primary actually addresses joiner ranks 4..7
        # post-scale. If primary's histogram is `[a,b,c,d,0,0,0,0]`
        # post-scale, NIXL is innocent — the bug is in the per-rank
        # `logical_to_rank_dispatch_physical_map`, not the late-add
        # NIXL handshake. Logged on EVERY rank so we get a symmetric
        # picture for each scale event.
        if getattr(self, "_traffic_diag_ep", None) != _ep:
            self._traffic_diag_ep = _ep
            self._traffic_diag_count = 0
        if (
            getattr(self, "_traffic_diag_count", 0) < 5
            and NixlEPBuffer._num_local_experts
            and _ep
        ):
            nle = NixlEPBuffer._num_local_experts
            valid_mask = topk_ids >= 0
            if valid_mask.any():
                target_ranks = topk_ids[valid_mask] // nle
                clamped = target_ranks.clamp(min=0, max=_ep - 1).to(torch.int64)
                hist = torch.bincount(clamped, minlength=_ep).cpu().tolist()
                oob = int((target_ranks >= _ep).sum().item())
                t_min = int(target_ranks.min().item())
                t_max = int(target_ranks.max().item())
            else:
                hist, oob, t_min, t_max = [], 0, -1, -1
            logger.info(
                "[Elastic EP][nixl][traffic] dispatch #%d ep=%s "
                "tokens_per_target_rank=%s (target_min=%d target_max=%d "
                "oob_count=%d num_local_experts=%d num_tokens=%d)",
                self._traffic_diag_count, _ep, hist, t_min, t_max, oob, nle,
                int(hidden_states.shape[0]) if hidden_states.dim() > 0 else 0,
            )
            self._traffic_diag_count += 1

        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
        )

        if getattr(self, "_shapes_logged_ep", None) != _ep:
            hs_shape = list(hidden_states[0].shape) if isinstance(hidden_states, tuple) else list(hidden_states.shape)
            logger.info(
                "[SHAPES] dispatch_a OUTPUT: hidden_states=%s masked_m=%s "
                "masked_m_sum=%d expected_m=%d",
                hs_shape, list(masked_m.shape),
                masked_m.sum().item(), expected_m,
            )
            self._shapes_logged_ep = _ep

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

        # POST-WAIT masked_m probe (replaces the racy dispatch_a OUTPUT log).
        # `masked_m` is filled atomically by the NIXL recv kernel; reading
        # it before `event.current_stream_wait()` (or `hook()`) returns
        # stale zero data. By logging here -- after the wait -- we get the
        # ground-truth per-local-expert receive count. masked_m_sum=0 here
        # is REAL; non-zero means the dispatch delivered tokens.
        #
        # Logged for the first 5 dispatches per `_ep` change, every rank.
        # Use the BUFFER-level `_ep_size` so the gate fires both pre- and
        # post-scale.
        _ep = NixlEPBuffer._ep_size
        if getattr(self, "_postwait_diag_ep", None) != _ep:
            self._postwait_diag_ep = _ep
            self._postwait_diag_count = 0
        if getattr(self, "_postwait_diag_count", 0) < 5 and _ep:
            try:
                m_list = masked_m.cpu().tolist()
                m_sum = int(sum(m_list))
                m_max = int(max(m_list)) if m_list else 0
                non_zero = sum(1 for v in m_list if v > 0)
                logger.info(
                    "[Elastic EP][nixl][post-wait] dispatch #%d ep=%s "
                    "masked_m_sum=%d masked_m_max_per_expert=%d "
                    "num_local_experts_with_traffic=%d/%d "
                    "expected_m=%d masked_m_per_expert=%s",
                    self._postwait_diag_count, _ep,
                    m_sum, m_max, non_zero, len(m_list),
                    expected_m, m_list,
                )
            except Exception as _e:
                logger.warning(
                    "[Elastic EP][nixl][post-wait] probe failed: %s", _e,
                )
            self._postwait_diag_count += 1

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
            # DEBUG — dump NIXL state + live expert-location metadata
            # ONCE per ep_size change. Helps isolate whether:
            #  - NIXL's rank-fault mask is spuriously set on this rank
            #    (would silently drop tokens from masked src_ranks).
            #  - buffer.num_ranks (the kernel's divisor) matches our
            #    expected num_connected_ranks.
            #  - the global ExpertLocationMetadata on this rank matches
            #    what the primary broadcast + local expansion produced
            #    (p2l.sum is a global identity check across ranks; rtr
            #    rows show where THIS rank routes its own logicals).
            try:
                mask_vals: Optional[List[int]] = None
                if self._mask_buffer is not None:
                    tmp_mask = torch.zeros_like(self._mask_buffer)
                    buffer.query_mask_buffer(tmp_mask)
                    mask_vals = tmp_mask[: max(_cep, 8)].tolist()
                logger.info(
                    "[Elastic EP][nixl][debug] pre-dispatch NIXL state: "
                    "buffer.num_ranks=%s buffer.group_size=%s "
                    "mask[0:%d](0=alive,1=faulted)=%s",
                    getattr(buffer, "num_ranks", "N/A"),
                    buffer.group_size,
                    max(_cep, 8),
                    mask_vals,
                )
            except Exception as _e:
                logger.warning(
                    "[Elastic EP][nixl][debug] pre-dispatch state probe failed: %s",
                    _e,
                )

            try:
                from sglang.srt.eplb.expert_location import (
                    get_global_expert_location_metadata,
                )
                md = get_global_expert_location_metadata()
                if md is not None:
                    p2l = md.physical_to_logical_map
                    rtr = md.logical_to_rank_dispatch_physical_map
                    logger.info(
                        "[Elastic EP][nixl][mapping] live metadata: "
                        "p2l.shape=%s p2l.sum=%d "
                        "p2l[0,0:8]=%s p2l[0,-8:]=%s "
                        "rtr.shape=%s rtr[0,0:8]=%s rtr[0,-8:]=%s",
                        list(p2l.shape),
                        int(p2l.sum().item()),
                        p2l[0, :8].tolist(),
                        p2l[0, -8:].tolist(),
                        list(rtr.shape) if rtr is not None else None,
                        rtr[0, :8].tolist() if rtr is not None else None,
                        rtr[0, -8:].tolist() if rtr is not None else None,
                    )
            except Exception as _e:
                logger.warning(
                    "[Elastic EP][nixl][mapping] metadata probe failed: %s",
                    _e,
                )
            self._last_logged_cep = _cep
        nixl_num_experts = NixlEPBuffer._num_local_experts * NixlEPBuffer._ep_size
        if hidden_states.shape[0] > self.num_max_dispatch_tokens_per_rank:
            logger.error(
                "[Elastic EP][nixl] BATCH TOO LARGE: x.size(0)=%d > "
                "num_max_dispatch_tokens_per_rank=%d, nixl_num_experts=%d, "
                "ep_size=%d, group_size=%d, topk_idx.shape=%s, "
                "topk_idx.min=%d, topk_idx.max=%d",
                hidden_states.shape[0],
                self.num_max_dispatch_tokens_per_rank,
                nixl_num_experts,
                NixlEPBuffer._ep_size,
                buffer.group_size,
                list(topk_idx.shape),
                topk_idx[topk_idx >= 0].min().item() if topk_idx.numel() > 0 and (topk_idx >= 0).any() else -1,
                topk_idx.max().item(),
            )
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
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_ids,
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
        if self._mask_buffer is not None and not self._skip_fault_mask:
            # KNOWN-ARCH-ISSUE: this fault-mask flow is not globally
            # consistent across primary + joiner processes.
            #
            # Per-process state. `ElasticEPStateManager.instance().active_ranks`
            # is a per-process tensor, NOT all_reduced. Each process only
            # writes the slice it locally owns
            # (`[off : off + n]`, see below). Primary owns `[0:4]`, joiner
            # owns `[4:8]`. The SAME LOGICAL TENSOR has different values
            # in each process. Specifically:
            #
            #   primary's active_ranks  = [P0, P1, P2, P3, 1, 1, 1, 1]
            #   joiner's  active_ranks  = [1,  1,  1,  1,  J0, J1, J2, J3]
            #
            # NIXL's `query_mask_buffer` returns a kernel-side view of
            # which src_ranks the local recv kernel marked faulted (1)
            # vs alive (0). Below we DROP the foreign slice
            # (`_mask_buffer[0:off]` and `_mask_buffer[off+n:]`) when
            # writing back, on the rationale that "primary doesn't have
            # authority over joiner's slots" and vice versa.
            #
            # CONSEQUENCE OF DROPPING THE FOREIGN SLICE: when joiner's
            # NIXL kernel marks primary as faulted (which is what the
            # `[mask-flip] cur=[1,1,1,1,0,0,0,0]` log line on the joiner
            # side means), that "primary is faulted" signal never reaches
            # the primary process. Primary's local `active_ranks` keeps
            # showing primary slots as alive (because each process
            # initialized them to 1 at scale time and only updates its
            # own slice). This is fine for scale-up correctness in
            # principle (no real fault occurred) but it means the
            # fault-tolerance path is one-sided in scale-up mode.
            #
            # The "globally sync'd" view that DOES exist is at the
            # Mooncake/NCCL PG level: `self.tp_group.active_ranks`
            # (read by `scheduler.py:2902` and forwarded via
            # `ActiveRanksOutput` to the DataParallelController).
            # That tensor IS maintained collectively by the PG backend.
            # It is a DIFFERENT tensor from the
            # `ElasticEPStateManager.instance().active_ranks` one we
            # update here. Mixing the two is the source of confusion.
            #
            # TODO(elastic-EP): unify the two `active_ranks` tensors so
            # that NIXL's per-rank fault info propagates globally (e.g.
            # an all_reduce(MIN) on the kernel-mask-derived view, or
            # routing all fault detection through the Mooncake PG).
            # Until then, the kernel-mask probe and the global view will
            # disagree under partial-network failures.
            buffer.query_mask_buffer(self._mask_buffer)

            # Probe: log mask transitions (only when the mask actually
            # flips), not every combine. Logs the rank that just got
            # newly marked faulted/recovered, plus the dispatch counter
            # within this dispatcher instance. Capped at 50 lines per
            # dispatcher to avoid spam if NIXL is constantly toggling.
            try:
                cur = self._mask_buffer.cpu().tolist()
                if self._prev_mask_snapshot is not None and self._mask_log_count < 50:
                    flips = [
                        (r, self._prev_mask_snapshot[r], cur[r])
                        for r in range(min(len(cur), len(self._prev_mask_snapshot)))
                        if cur[r] != self._prev_mask_snapshot[r]
                    ]
                    if flips:
                        logger.warning(
                            "[Elastic EP][nixl][mask-flip] dispatcher_id=%d "
                            "ep_size=%s flips=%s prev=%s cur=%s",
                            id(self) % 100000,
                            NixlEPBuffer._ep_size,
                            flips, self._prev_mask_snapshot, cur,
                        )
                        self._mask_log_count += 1
                self._prev_mask_snapshot = cur
            except Exception as _e:
                logger.warning(
                    "[Elastic EP][nixl][mask-flip] probe failed: %s", _e,
                )

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
