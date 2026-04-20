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

            backend = server_args.elastic_ep_backend
            if backend == "nixl":
                cls._on_scale = cls._on_scale_nixl
            elif backend == "mooncake":
                cls._on_scale = cls._on_scale_mooncake

            if server_args.ep_join_mode in ("scale", "recover"):
                # Mask out peer ranks to perform cuda graph capture on its own
                cls._instance.active_ranks.zero_()
                cls._instance.active_ranks[torch.distributed.get_rank()] = 1
                cls._instance.snapshot_active_to_last()
                cls._instance.sync_active_to_cpu()

            logger.info(
                "[Elastic EP][init] rank=%d world_size=%d max_ep_size=%s "
                "effective_ep_size=%d ep_join_mode=%s backend=%s active_ranks=%s",
                torch.distributed.get_rank(),
                world_size,
                server_args.max_ep_size,
                cls._instance.effective_ep_size,
                server_args.ep_join_mode,
                backend,
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

    @staticmethod
    def _on_scale_nixl(from_ep_size: int, to_ep_size: int) -> None:
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        # NixlEPBuffer.on_scale lands in a follow-up PR that adds the
        # _scale_to / _connected_ep_size lazy-update path. Until then,
        # NIXL connections will be set up on first dispatch via the
        # existing connect-on-demand path; warn so users know scale-up
        # buffer recapture is deferred.
        on_scale = getattr(NixlEPBuffer, "on_scale", None)
        if on_scale is None:
            logger.warning(
                "[Elastic EP] NixlEPBuffer.on_scale not available; "
                "buffer recapture for ranks %d..%d is deferred to first dispatch.",
                from_ep_size,
                to_ep_size,
            )
            return
        on_scale(from_ep_size, to_ep_size)

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


def _wait_for_peer_state(mooncake_ep, backend, ranks: List[int]) -> None:
    # Relaunched ranks become recoverable asynchronously, so we poll until the
    # target backend reports all requested peers as ready.
    while not all(mooncake_ep.get_peer_state(backend, ranks)):
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

    EPBuffer._buffer.update_ep_member()


def try_recover_ranks(global_ranks: List[int]) -> bool:
    from mooncake import ep as mooncake_ep

    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    if not all(mooncake_ep.get_peer_state(world_backend, global_ranks)):
        # The relaunched ranks have not finished initializing yet.
        return False

    # Recover the world backend first, then recover each derived process group
    # using ranks mapped into that group's local rank space.
    mooncake_ep.recover_ranks(world_backend, global_ranks)

    for group in _iter_live_parallel_groups():
        group_local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not group_local_ranks:
            continue

        device_backend = _get_process_group_backend(group.device_group, "cuda")
        _wait_for_peer_state(mooncake_ep, device_backend, group_local_ranks)
        mooncake_ep.recover_ranks(device_backend, group_local_ranks)

        cpu_backend = _get_process_group_backend(group.cpu_group, "cpu")
        _wait_for_peer_state(mooncake_ep, cpu_backend, group_local_ranks)
        mooncake_ep.recover_ranks(cpu_backend, group_local_ranks)
        _maybe_create_message_queue(group)

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
