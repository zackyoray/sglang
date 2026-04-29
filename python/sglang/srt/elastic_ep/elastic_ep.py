from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch

from sglang.srt.distributed import parallel_state
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

            # _on_scale triggers NIXL connect_ranks for new ranks.
            # Use nixl handler whenever NIXL is the MoE transport,
            # regardless of which PG backend (mooncake/nixl) handles coordination.
            if server_args.moe_a2a_backend == "nixl":
                cls._on_scale = cls._on_scale_nixl
            elif server_args.elastic_ep_backend == "nixl":
                cls._on_scale = cls._on_scale_nixl
            elif server_args.elastic_ep_backend == "mooncake":
                cls._on_scale = cls._on_scale_mooncake

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
                "backend=%s active_ranks=%s",
                torch.distributed.get_rank(),
                world_size,
                server_args.max_ep_size,
                cls._instance.effective_ep_size,
                server_args.ep_join_mode,
                cls._instance.ep_join_rank_offset,
                server_args.elastic_ep_backend,
                cls._instance.active_ranks.tolist(),
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
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        # Sets NixlEPBuffer._scale_to so get_nixl_buffer can extend connect_ranks
        # on dispatch (lazy mesh growth). Must run after joins, not at HTTP scale.
        NixlEPBuffer.on_scale(from_ep_size, to_ep_size)

    @staticmethod
    def _on_scale_mooncake(from_ep_size: int, to_ep_size: int) -> None:
        logger.warning("[Elastic EP] Mooncake on_scale not yet implemented")

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

        active_ranks is pre-allocated to max_ep_size with reserved slots
        starting at 0. After /scale_elastic_ep bumps effective_ep_size,
        the reserved slots fall inside the poll window but stay 0 until
        the new ranks complete join. The same expression also flips True
        on a fault inside the window (an active slot drops to 0).
        """
        inst = cls._instance
        if inst is None or inst.active_ranks is None:
            return False
        active_count = int(
            inst.active_ranks[: inst.effective_ep_size].sum().item()
        )
        return active_count < inst.effective_ep_size


# ---------------------------------------------------------------------------
# Helpers for elastic EP recovery
# ---------------------------------------------------------------------------


_PEER_STATE_POLL_INTERVAL_SEC = 0.01


def _get_process_group_backend(process_group, device: str):
    return process_group._get_backend(torch.device(device))


def _iter_live_parallel_groups() -> Iterator[parallel_state.GroupCoordinator]:
    groups = []
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if group is not None:
            groups.append(group)
    for group in sorted(groups, key=lambda x: x.unique_name):
        yield group


def _map_global_to_group_local_ranks(
    group_ranks: List[int], global_ranks: List[int]
) -> List[int]:
    rank_to_local = {rank: idx for idx, rank in enumerate(group_ranks)}
    return [rank_to_local[rank] for rank in global_ranks if rank in rank_to_local]


def _wait_for_peer_state(
    mooncake_ep, backend, ranks: List[int], timeout_s: float = 60.0
) -> None:
    # Poll until all requested peers are ready (joiner in join_group).
    # get_peer_state is a collective — all active ranks must call it.
    deadline = time.time() + timeout_s
    while not all(mooncake_ep.get_peer_state(backend, ranks)):
        if time.time() > deadline:
            logger.warning(
                "[Elastic EP] _wait_for_peer_state timed out after %.0fs "
                "waiting for ranks %s", timeout_s, ranks,
            )
            return
        time.sleep(_PEER_STATE_POLL_INTERVAL_SEC)


def _maybe_create_message_queue(group) -> None:
    if not group.use_message_queue_broadcaster or group.world_size <= 1:
        return

    from sglang.srt.distributed.device_communicators.shm_broadcast import MessageQueue

    group.mq_broadcaster = MessageQueue.create_from_process_group(
        group.cpu_group, 1 << 22, 6
    )


def _refresh_ep_members() -> None:
    from sglang.srt.layers.moe.token_dispatcher.mooncake import EPBuffer

    if EPBuffer._buffer is not None:
        EPBuffer._buffer.update_ep_member()


def try_recover_ranks(global_ranks: List[int]) -> bool:
    from mooncake import ep as mooncake_ep

    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    logger.info("[Elastic EP][recover] polling WORLD get_peer_state(%s)...", global_ranks)
    if not all(mooncake_ep.get_peer_state(world_backend, global_ranks)):
        return False

    logger.info("[Elastic EP][recover] WORLD get_peer_state returned True! Calling recover_ranks...")
    mooncake_ep.recover_ranks(world_backend, global_ranks)
    logger.info("[Elastic EP][recover] WORLD recover_ranks done. Processing sub-groups...")

    # In dp_attention mode, all multi-rank sub-groups (TP, MOE_EP) are
    # equivalent to WORLD — same rank set.  WORLD recover is sufficient.
    # Skip sub-group recover entirely to avoid backendIndex mismatch
    # between separate primary/joiner processes.
    # TODO: For non-dp_attention modes with true sub-groups (e.g. tp=2
    # within a larger world), implement split-ranks pattern per Mooncake.
    logger.info("[Elastic EP][recover] Skipping sub-group recover (dp_attention: sub-groups == WORLD)")
    _refresh_ep_members()
    return True

    # Dead code below — kept for future non-dp_attention implementation
    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue

        # With max_world_size, sub-groups have capacity beyond their current
        # membership. Joiner ranks 4-7 map to local indices 4-7 in the
        # sub-group (same as their global rank, since all sub-groups share
        # the same rank space with max_world_size).
        group_local_ranks = global_ranks  # direct: global == local in max_world_size groups
        logger.info(
            "[Elastic EP][recover] sub-group %s: group.ranks=%s ws=%d "
            "group_local_ranks=%s — polling get_peer_state...",
            group.unique_name, group.ranks, group.world_size, group_local_ranks,
        )

        device_backend = _get_process_group_backend(group.device_group, "cuda")
        _wait_for_peer_state(mooncake_ep, device_backend, group_local_ranks)
        logger.info("[Elastic EP][recover] %s:device peer ready, calling recover_ranks...", group.unique_name)
        mooncake_ep.recover_ranks(device_backend, group_local_ranks)
        logger.info("[Elastic EP][recover] %s:device recover done", group.unique_name)

        cpu_backend = _get_process_group_backend(group.cpu_group, "cpu")
        _wait_for_peer_state(mooncake_ep, cpu_backend, group_local_ranks)
        logger.info("[Elastic EP][recover] %s:cpu peer ready, calling recover_ranks...", group.unique_name)
        mooncake_ep.recover_ranks(cpu_backend, group_local_ranks)
        logger.info("[Elastic EP][recover] %s:cpu recover done", group.unique_name)
        _maybe_create_message_queue(group)

    logger.info("[Elastic EP][recover] ALL sub-groups recovered!")
    _refresh_ep_members()
    return True


def join_process_groups():
    from mooncake import ep as mooncake_ep

    def join_backend(label: str, backend) -> None:
        logger.info("[Elastic EP][join_pg] calling join_group on %s", label)
        mooncake_ep.join_group(backend)
        logger.info("[Elastic EP][join_pg] join_group returned on %s", label)

    join_backend(
        "default_world",
        _get_process_group_backend(torch.distributed.group.WORLD, "cuda"),
    )

    # In dp_attention mode, all multi-rank sub-groups are equivalent to
    # WORLD. Skip sub-group join — WORLD join is sufficient.
    # TODO: For non-dp_attention with true sub-groups, implement split-ranks.
    logger.info("[Elastic EP][join_pg] Skipping sub-group joins (dp_attention: sub-groups == WORLD)")
    return

    # Dead code below — kept for future non-dp_attention implementation
    for group in _iter_live_parallel_groups():
        if group.world_size <= 1:
            continue

        join_backend(
            f"{group.unique_name}:device",
            _get_process_group_backend(group.device_group, "cuda"),
        )
        join_backend(
            f"{group.unique_name}:cpu",
            _get_process_group_backend(group.cpu_group, "cpu"),
        )
        _maybe_create_message_queue(group)

    _refresh_ep_members()
