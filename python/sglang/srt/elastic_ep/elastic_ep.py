from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

logger = logging.getLogger(__name__)


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]

    max_ep_size: int = 0
    scaling_in_progress: bool = False
    target_ep_size: Optional[int] = None

    def is_active_equal_last(self) -> bool:
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self):
        if self.active_ranks is not None:
            self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self.active_ranks is not None:
            self.last_active_ranks = self.active_ranks.clone()


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None
    _on_scale: Optional[Callable[[int, int], None]] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if server_args.elastic_ep_backend is not None:
            ep_size = getattr(server_args, "ep_size", None)
            max_ep_size = getattr(server_args, "max_ep_size", None)
            if max_ep_size is None or max_ep_size <= 0:
                max_ep_size = ep_size
            cls._instance = cls._build_state(
                ep_size=ep_size, max_ep_size=max_ep_size, device=None
            )

            backend = server_args.elastic_ep_backend
            if backend == "nixl":
                cls._on_scale = cls._on_scale_nixl
            elif backend == "mooncake":
                cls._on_scale = cls._on_scale_mooncake
            else:
                cls._on_scale = None

        return cls._instance

    @staticmethod
    def _select_device() -> torch.device:
        if is_cuda():
            return torch.device("cuda")
        elif is_cpu():
            return torch.device("cpu")
        else:
            raise NotImplementedError("Only CUDA and CPU support elastic ep now.")

    @classmethod
    def _build_state(
        cls,
        *,
        ep_size: Optional[int] = None,
        max_ep_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(
            ep_size=ep_size, max_ep_size=max_ep_size, device=device
        )
        resolved_max = max_ep_size if max_ep_size is not None else active.shape[0]
        return ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
            max_ep_size=resolved_max,
        )

    @classmethod
    def healthy_rank_state(
        cls,
        *,
        ep_size: Optional[int] = None,
        max_ep_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return an active_ranks tensor pre-allocated to *max_ep_size*.

        The first *ep_size* slots are set to 1 (active), the rest to 0
        (inactive, available for future scale-up).  When *max_ep_size* is
        ``None`` or equal to *ep_size*, behaviour is identical to the
        original implementation (all-ones tensor).
        """
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        total = max_ep_size if max_ep_size is not None else size
        total = max(total, size)
        dev = device if device is not None else cls._select_device()

        state = torch.zeros(total, dtype=torch.int32, device=dev)
        state[:size] = 1
        return state

    # ------------------------------------------------------------------
    # Public scaling API
    # ------------------------------------------------------------------

    @classmethod
    def scale(cls, new_ep_size: int) -> None:
        """Scale EP to *new_ep_size*.  Called on **all** ranks.

        1. Flips ``active_ranks`` bits at the frontier.
        2. Calls the backend ``_on_scale(from, to)`` set during init.

        EPLB triggers automatically on the next forward pass when
        ``model_runner`` detects the ``active_ranks`` change.

        **Preconditions**:

        * All *new_ep_size* rank processes must be running.
        * New ranks load non-expert weights from disk at startup.
        * ``(new_ep_size - frontier) % attn_tp_size == 0``.

        .. note::

           TBD -- Unify with rank recovery (re-activate faulted slots).
        """
        inst = cls._instance
        if inst is None:
            raise RuntimeError(
                "ElasticEPStateManager is not initialised. "
                "Call init(server_args) first."
            )

        frontier = cls.get_ep_frontier()
        if new_ep_size == frontier:
            return

        if new_ep_size > inst.max_ep_size:
            raise ValueError(
                f"Requested ep_size {new_ep_size} exceeds max_ep_size "
                f"{inst.max_ep_size}.  Launch the server with a larger "
                f"--max-ep-size value."
            )
        if new_ep_size <= 0:
            raise ValueError("new_ep_size must be a positive integer.")

        logger.info("[Elastic EP] Scaling from %d to %d", frontier, new_ep_size)

        inst.scaling_in_progress = True
        inst.target_ep_size = new_ep_size
        try:
            if new_ep_size > frontier:
                inst.active_ranks[frontier:new_ep_size] = 1
            else:
                inst.active_ranks[new_ep_size:frontier] = 0

            if cls._on_scale is not None:
                cls._on_scale(frontier, new_ep_size)
        finally:
            inst.scaling_in_progress = False
            inst.target_ep_size = None

    @classmethod
    def is_scaling(cls) -> bool:
        """``True`` while a :meth:`scale` operation is in progress."""
        inst = cls._instance
        return inst is not None and inst.scaling_in_progress

    @classmethod
    def get_num_active_ranks(cls) -> int:
        """Count of currently active EP ranks (may be non-contiguous after faults)."""
        inst = cls._instance
        if inst is None or inst.active_ranks is None:
            return 0
        return int(inst.active_ranks.sum().item())

    @classmethod
    def get_ep_frontier(cls) -> int:
        """Index after the last active rank -- where ``scale()`` appends.

        With no faults this equals ``get_num_active_ranks()``.  After a
        fault (gap in the middle) the frontier is higher than the active
        count.  Example::

            active_ranks = [1, 1, 0, 1, 0, 0, 0, 0]
            get_num_active_ranks() → 3
            get_ep_frontier()      → 4   (last active is index 3)
        """
        inst = cls._instance
        if inst is None or inst.active_ranks is None:
            return 0
        nonzero = torch.nonzero(inst.active_ranks, as_tuple=True)[0]
        if nonzero.numel() == 0:
            return 0
        return int(nonzero[-1].item()) + 1

    # ------------------------------------------------------------------
    # Backend-specific on_scale wrappers (set via _on_scale in init)
    # ------------------------------------------------------------------

    @staticmethod
    def _on_scale_nixl(from_ep_size: int, to_ep_size: int) -> None:
        """NIXL EP: eagerly connect/disconnect ranks."""
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        NixlEPBuffer.on_scale(from_ep_size, to_ep_size)

    @staticmethod
    def _on_scale_mooncake(from_ep_size: int, to_ep_size: int) -> None:
        """Mooncake: extend/shrink process group.

        See PR #15771 for the upstream Mooncake mechanism.
        """
        raise NotImplementedError(
            "TODO: add EPBuffer.on_scale() in mooncake.py"
        )

