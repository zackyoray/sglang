"""NIXL EP v2 transport with the shared ElasticBuffer expert-compute layout."""

from __future__ import annotations

import logging
from functools import cache

import torch
import torch.distributed as dist

from sglang.srt.distributed.utils import get_global_tcp_store
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
    DeepEPv2Buffer,
    DeepEPv2Dispatcher,
    _DeepEPv2Impl,
)
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)


@cache
def _load_nixl_ep_v2():
    try:
        from nixl.runtime import ep
    except ImportError as exc:
        raise ImportError(
            "NIXL EP v2 is unavailable. Install NIXL with build_nixl_ep=true "
            "and the nixl.runtime.ep.ElasticBuffer API."
        ) from exc
    if not hasattr(ep, "ElasticBuffer") or not isinstance(
        getattr(ep, "topk_idx_t", None), torch.dtype
    ):
        raise ImportError("NIXL EP v2 requires ElasticBuffer and topk_idx_t.")
    return ep.ElasticBuffer, ep.topk_idx_t


class NixlEPV2Buffer(DeepEPv2Buffer):
    """Own a NIXL buffer separately from both other EP implementations."""

    _STATE_KEY = "nixl_ep_v2_state"

    @classmethod
    def on_scale(cls, from_ep_size: int, to_ep_size: int) -> None:
        state = cls._state()
        state.target_ep_size = to_ep_size
        if state.buffer is not None:
            cls._activate_world(state, to_ep_size)

    @classmethod
    def _activate_world(cls, state, target_ep_size: int) -> None:
        buffer = state.buffer
        peers = set(range(target_ep_size)) - {buffer.rank_id}
        missing = sorted(peers - set(buffer.get_connected_ranks()))
        buffer.connect_ranks(
            missing, store=get_global_tcp_store(), key_prefix=state.peer_prefix
        )
        buffer.set_active_ranks(list(range(target_ep_size)))
        state.dispatch_ep_size = target_ep_size
        logger.info(
            "NIXL EP v2 active world: rank=%s ep_size=%s mode=%s",
            buffer.rank_id,
            target_ep_size,
            buffer.get_topology()[0],
        )

    @classmethod
    def get_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_local_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        use_fp8_dispatch: bool,
    ):
        buffer_cls, _ = _load_nixl_ep_v2()
        key = (
            group,
            hidden_size,
            router_topk,
            num_local_experts,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            dist.get_world_size(group),
        )
        state = cls._state()
        if state.buffer is not None and state.key == key:
            return state.buffer
        cls.destroy()
        # NIXL derives direct/hybrid mode from the live NVLink topology. Expert
        # slots are fixed at construction, not supplied to each dispatch call.
        # Use native destructor ownership so runtime-context teardown releases it.
        elastic = get_exec().moe.elastic_ep_backend is not None
        offset = ElasticEPStateManager.get_ep_join_rank_offset() if elastic else 0
        cohort_size = dist.get_world_size(group)
        kwargs = {}
        if elastic:
            domain_size = envs.SGLANG_NIXL_EP_V2_MAX_SCALEUP_RANKS.get()
            max_ep_size = get_parallel().max_ep_size
            if (
                not domain_size
                or domain_size < 1
                or not max_ep_size
                or max_ep_size % domain_size
            ):
                raise ValueError(
                    "NIXL EP v2 elastic mode requires a positive "
                    "SGLANG_NIXL_EP_V2_MAX_SCALEUP_RANKS dividing --max-ep-size."
                )
            rank = offset + dist.get_rank(group)
            store = get_global_tcp_store()
            if store is None:
                raise RuntimeError(
                    "NIXL EP v2 elastic mode requires the global TCPStore."
                )
            kwargs = dict(
                max_num_scaleout_ranks=max_ep_size // domain_size,
                max_num_scaleup_ranks=domain_size,
                scaleout_rank_idx=rank // domain_size,
                scaleup_rank_idx=rank % domain_size,
                store=store,
                bootstrap_id=f"sglang/nixl_ep_v2/cohort/{offset}/{cohort_size}",
            )
        buffer = buffer_cls(
            group=group,
            hidden=hidden_size,
            num_topk=router_topk,
            num_experts_per_rank=num_local_experts,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch=use_fp8_dispatch,
            prefer_overlap_with_compute=False,
            **kwargs,
        )
        state.buffer = buffer
        state.key = key
        state.dispatch_ep_size = cohort_size
        if elastic:
            state.peer_prefix = "sglang/nixl_ep_v2/peers"
            buffer.publish(store=store, key_prefix=state.peer_prefix)
            target = max(offset + cohort_size, getattr(state, "target_ep_size", 0))
            cls._activate_world(state, target)
        return buffer


class _NixlEPV2Impl(_DeepEPv2Impl):
    def _ensure_available(self) -> None:
        _load_nixl_ep_v2()

    def _topk_indices_dtype(self) -> torch.dtype:
        return _load_nixl_ep_v2()[1]

    def _get_buffer(self):
        if (
            get_exec().moe.elastic_ep_backend is None
            and self.num_experts
            != self.num_local_experts * dist.get_world_size(self.group)
        ):
            raise ValueError("NIXL EP v2 requires equal expert slots on every EP rank.")
        return NixlEPV2Buffer.get_buffer(
            self.group,
            self.hidden_size,
            self.router_topk,
            self.num_local_experts,
            self.num_max_dispatch_tokens_per_rank,
            self.use_fp8_dispatch,
        )

    def _get_dispatch_ep_size(self):
        return NixlEPV2Buffer._state().dispatch_ep_size

    def _dispatch_buffer(self, buffer, dispatch_x, **kwargs):
        return buffer.dispatch(dispatch_x, **kwargs)


class NixlEPV2Dispatcher(DeepEPv2Dispatcher):
    """NIXL transport; reuse DeepEP v2 layout, FP8 conversion, and combine logic.

    SGLANG_DEEPEP_V2_* settings tune the shared compute/dispatch path. NIXL's
    own NIXL_EP_* settings control its transport and JIT kernels.
    """

    _IMPL_CLASS = _NixlEPV2Impl
