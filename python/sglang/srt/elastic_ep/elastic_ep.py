from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.managers.schedule_batch import ServerArgs
from sglang.srt.utils import is_cpu, is_cuda

logger = logging.getLogger(__name__)


@dataclass
class ElasticEPState:
    active_ranks: Optional[torch.Tensor]
    last_active_ranks: Optional[torch.Tensor]
    active_ranks_cpu: Optional[torch.Tensor]
    effective_ep_size: int = 0
    original_ep_size: int = 0
    # Global EP rank offset for joiner processes (see --ep-join-rank-offset).
    # Local torch rank r maps to global EP rank r + ep_join_rank_offset for
    # elastic EP bookkeeping. 0 on non-joiners.
    ep_join_rank_offset: int = 0

    def is_active_equal_last(self) -> bool:
        return torch.equal(self.active_ranks, self.last_active_ranks)

    def sync_active_to_cpu(self):
        if self.active_ranks is not None:
            self.active_ranks_cpu = self.active_ranks.detach().cpu().clone()

    def snapshot_active_to_last(self):
        if self.active_ranks is not None:
            self.last_active_ranks = self.active_ranks.clone()

    def reset(self):
        if self.active_ranks is not None:
            # Only mark the in-window slots active; reserved scale-up slots
            # beyond effective_ep_size stay 0 until those ranks join.
            self.active_ranks.zero_()
            self.active_ranks[: self.effective_ep_size] = 1
            self.snapshot_active_to_last()
            self.sync_active_to_cpu()


class ElasticEPStateManager:
    _instance: Optional[ElasticEPState] = None
    _on_scale: Optional[callable] = None

    @classmethod
    def instance(cls) -> ElasticEPState:
        return cls._instance

    @classmethod
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if server_args.elastic_ep_backend is not None:
            world_size = torch.distributed.get_world_size()
            # Pre-allocate to max_ep_size so EPLB and is_scaling() can index
            # reserved slots without ever resizing the tensor (resize would
            # race with concurrent readers in the dispatcher / EPLB path).
            # Slots [world_size:] start at 0 (unjoined). reset() preserves
            # this layout by only filling [:effective_ep_size] with 1.
            tensor_size = server_args.max_ep_size or world_size
            assert tensor_size >= world_size, (
                f"--max-ep-size ({tensor_size}) must be >= world_size ({world_size})."
            )
            cls._instance = cls._build_state(ep_size=tensor_size, device=None)
            cls._instance.effective_ep_size = world_size
            cls._instance.original_ep_size = world_size
            if tensor_size > world_size:
                cls._instance.active_ranks[world_size:].zero_()
                cls._instance.snapshot_active_to_last()
                cls._instance.sync_active_to_cpu()

            # _on_scale triggers NIXL connect_ranks for new ranks. Routed
            # whenever NIXL is the MoE transport, regardless of the PG
            # backend; pure-Mooncake MoE elastic is not yet supported.
            if (
                server_args.moe_a2a_backend == "nixl"
                or server_args.elastic_ep_backend == "nixl"
            ):
                cls._on_scale = cls._on_scale_nixl

            cls._instance.ep_join_rank_offset = (
                getattr(server_args, "ep_join_rank_offset", 0) or 0
            )

            if server_args.ep_join_mode in ("scale", "recover"):
                # The PG rank already includes ep_join_rank_offset (set in
                # model_runner), so it IS the global EP index.
                global_rank = torch.distributed.get_rank()
                cls._instance.active_ranks.zero_()
                cls._instance.active_ranks[global_rank] = 1
                cls._instance.snapshot_active_to_last()
                cls._instance.sync_active_to_cpu()

            logger.info(
                "[Elastic EP][init] rank=%d world_size=%d max_ep_size=%s "
                "effective_ep_size=%d ep_join_mode=%s ep_join_rank_offset=%d "
                "backend=%s",
                torch.distributed.get_rank(),
                world_size,
                server_args.max_ep_size,
                cls._instance.effective_ep_size,
                server_args.ep_join_mode,
                cls._instance.ep_join_rank_offset,
                server_args.elastic_ep_backend,
            )

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
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> ElasticEPState:
        active = cls.healthy_rank_state(ep_size=ep_size, device=device)
        return ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.detach().cpu().clone(),
        )

    @classmethod
    def healthy_rank_state(
        cls, *, ep_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        size = ep_size if ep_size is not None else torch.distributed.get_world_size()
        dev = device if device is not None else cls._select_device()

        return torch.ones(size, dtype=torch.int32, device=dev)


    @classmethod
    def set_effective_ep_size(cls, n: int) -> None:
        inst = cls._instance
        if inst is not None:
            inst.effective_ep_size = n

    @classmethod
    def get_effective_ep_size(cls) -> int:
        inst = cls._instance
        if inst is None:
            return 0
        return inst.effective_ep_size

    @classmethod
    def get_ep_join_rank_offset(cls) -> int:
        inst = cls._instance
        if inst is None:
            return 0
        return inst.ep_join_rank_offset

    @staticmethod
    def _on_scale_nixl(from_ep_size: int, to_ep_size: int) -> None:
        """Mark NIXL buffer for lazy connect_ranks on next dispatch."""
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer
        NixlEPBuffer.on_scale(from_ep_size, to_ep_size)

    @classmethod
    def is_recovery_join(cls, rank_ids: List[int]) -> bool:
        """True if any joining rank was previously part of the original world."""
        inst = cls._instance
        if inst is None:
            return False
        return any(r < inst.original_ep_size for r in rank_ids)

    @classmethod
    def is_scaling(cls) -> bool:
        """True iff there are unjoined ranks within the current poll window.

        Reads ``active_ranks_cpu`` (not the GPU view) to stay consistent
        with ``maybe_join_ep_ranks``, which derives the join target list
        from the same CPU snapshot. Reading the GPU view here would let
        a NIXL-driven fault flip GPU ``active_ranks`` to 0 while
        ``active_ranks_cpu`` still reads 1, producing endless "scaling"
        with no recovery progress (Codex review #3).
        """
        inst = cls._instance
        if inst is None or inst.active_ranks_cpu is None:
            return False
        active_count = int(
            inst.active_ranks_cpu[: inst.effective_ep_size].sum().item()
        )
        return active_count < inst.effective_ep_size


# ---------------------------------------------------------------------------
# Helpers for elastic EP recovery
# ---------------------------------------------------------------------------


def _get_process_group_backend(process_group, device: str):
    return process_group._get_backend(torch.device(device))


def _refresh_ep_members() -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    if EPBuffer._buffer is not None:
        EPBuffer._buffer.update_ep_member()


def try_recover_ranks(global_ranks: List[int]) -> bool:
    """Recover (admit) the given global ranks into the Mooncake WORLD group.

    In dp_attention mode all multi-rank sub-groups are equivalent to WORLD,
    so recovering on WORLD is sufficient. Non-dp_attention sub-groups are
    not supported by this elastic path yet.
    """
    from mooncake import ep as mooncake_ep

    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    if not all(mooncake_ep.get_peer_state(world_backend, global_ranks)):
        return False

    mooncake_ep.recover_ranks(world_backend, global_ranks)
    logger.info("[Elastic EP][recover] WORLD recover_ranks(%s) done", global_ranks)
    _refresh_ep_members()
    return True


def join_process_groups():
    """Joiner-side counterpart to try_recover_ranks: join the WORLD group.

    In dp_attention mode the multi-rank sub-groups are equivalent to WORLD,
    so joining WORLD is sufficient.
    """
    from mooncake import ep as mooncake_ep

    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    mooncake_ep.join_group(world_backend)
    logger.info("[Elastic EP][join_pg] join_group(WORLD) returned")
