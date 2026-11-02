"""NIXL EP v2 transport with the shared ElasticBuffer expert-compute layout."""

from __future__ import annotations

from functools import cache

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
    DeepEPv2Buffer,
    DeepEPv2Dispatcher,
    _DeepEPv2Impl,
)


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
        buffer = buffer_cls(
            group=group,
            hidden=hidden_size,
            num_topk=router_topk,
            num_experts_per_rank=num_local_experts,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch=use_fp8_dispatch,
            prefer_overlap_with_compute=False,
        )
        state.buffer = buffer
        state.key = key
        return buffer


class _NixlEPV2Impl(_DeepEPv2Impl):
    def _ensure_available(self) -> None:
        _load_nixl_ep_v2()

    def _topk_indices_dtype(self) -> torch.dtype:
        return _load_nixl_ep_v2()[1]

    def _get_buffer(self):
        if self.num_experts != self.num_local_experts * dist.get_world_size(self.group):
            raise ValueError("NIXL EP v2 requires equal expert slots on every EP rank.")
        return NixlEPV2Buffer.get_buffer(
            self.group,
            self.hidden_size,
            self.router_topk,
            self.num_local_experts,
            self.num_max_dispatch_tokens_per_rank,
            self.use_fp8_dispatch,
        )

    def _dispatch_buffer(self, buffer, dispatch_x, **kwargs):
        return buffer.dispatch(dispatch_x, **kwargs)


class NixlEPV2Dispatcher(DeepEPv2Dispatcher):
    """NIXL transport; reuse DeepEP v2 layout, FP8 conversion, and combine logic.

    SGLANG_DEEPEP_V2_* settings tune the shared compute/dispatch path. NIXL's
    own NIXL_EP_* settings control its transport and JIT kernels.
    """

    _IMPL_CLASS = _NixlEPV2Impl
