from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # some gathered elements
    tp0_info: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                ForwardMode.IDLE.value,  # local_forward_mode
            ],
            device=device,
            dtype=dtype,
        )

    def all_gather(
        self,
        device,
        group: torch.distributed.ProcessGroup,
        use_all_reduce: bool = False,
    ):
        local_info_tensor = self._get_local_tensor(device=device)
        fallback_tensor = self._get_fallback_tensor(device=device)
        # Prefill the gather buffer with the fallback pattern (IDLE forward_mode,
        # zero tokens, etc.) BEFORE the all_gather. Rationale: when the WORLD
        # group is Mooncake PG with `max_world_size` pre-provisioning, its
        # all_gather may not write every slot (see MOONCAKE_MAX_WORLD_SIZE_
        # INTEGRATION.md: "Works for allreduce, BROKEN for allgather/reduce
        # _scatter"). Using `torch.empty` left those slots as uninitialized
        # memory (often 0), which later crashed compute_output() with
        # `ValueError: 0 is not a valid ForwardMode` because 0 is not in the
        # ForwardMode enum. Prefilling with IDLE.value means any un-written
        # slot is correctly interpreted as an idle rank and is filtered out
        # by the idle/prebuilt exclusion in _compute_global_forward_mode().
        global_info_tensor = (
            fallback_tensor
            .expand(self.dp_size, self.tp_size * self.cp_size, 6)
            .contiguous()
        )

        if use_all_reduce:
            # Mooncake WORLD with max_world_size can report group.size()==4 on
            # primary even after active_ranks grows to 8, while joiners report
            # group.size()==8. all_gather_into_tensor sizes its output from
            # get_world_size(group), so primary and joiner don't actually
            # participate in the same 8-slot exchange. Mooncake allreduce is
            # the supported max_world_size collective, so encode each rank's
            # local 6-int record into its global-rank slot and sum the tensor.
            global_info_tensor.zero_()
            flat_info = global_info_tensor.view(-1, 6)
            rank = torch.distributed.get_rank(group)
            if 0 <= rank < flat_info.shape[0]:
                flat_info[rank] = local_info_tensor

            # Low-volume hang diagnostic. Always log non-idle ranks, and log
            # idle ranks at most once/sec so we can prove whether joiners
            # participate as idle batches without flooding the logs. Pair the
            # two lines: if [enter] fires but [exit] never does on any rank,
            # the WORLD all_reduce is the stuck collective. Toggle via
            # SGLANG_DEBUG_MLP_SYNC_HEARTBEAT=1.
            import os as _os_hb
            import time as _time_hb
            _heartbeat_enabled = (
                _os_hb.environ.get("SGLANG_DEBUG_MLP_SYNC_HEARTBEAT", "0") == "1"
            )
            _heartbeat = False
            if _heartbeat_enabled:
                if self.num_tokens > 0:
                    _heartbeat = True
                else:
                    _now_hb = _time_hb.monotonic()
                    _last_hb = getattr(MLPSyncBatchInfo, "_last_idle_hb_ts", 0.0)
                    if _now_hb - _last_hb >= 1.0:
                        MLPSyncBatchInfo._last_idle_hb_ts = _now_hb
                        _heartbeat = True
            if _heartbeat:
                import logging as _logging_hb
                _logging_hb.getLogger(__name__).info(
                    "[mlp-sync][heartbeat] enter rank=%d dp_size=%d "
                    "num_tokens=%d local_forward_mode=%d group_size=%d idle=%s",
                    rank, self.dp_size, self.num_tokens,
                    self.local_forward_mode,
                    torch.distributed.get_world_size(group),
                    self.num_tokens == 0,
                )
            torch.distributed.all_reduce(
                global_info_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
            if _heartbeat:
                _global_num_tokens_view = (
                    global_info_tensor.detach().cpu().view(-1, 6)[:, 0].tolist()
                )
                _logging_hb.getLogger(__name__).info(
                    "[mlp-sync][heartbeat] exit  rank=%d dp_size=%d "
                    "global_num_tokens=%s",
                    rank, self.dp_size, _global_num_tokens_view,
                )
            # Any slot with no participant contribution remains all-zero; turn
            # it back into the existing IDLE fallback so downstream ForwardMode
            # parsing never sees enum value 0.
            missing = flat_info.abs().sum(dim=1) == 0
            flat_info[missing] = fallback_tensor
        else:
            torch.distributed.all_gather_into_tensor(
                global_info_tensor.flatten(),
                local_info_tensor,
                group=group,
            )

        # Probe to verify whether the underlying allgather actually wrote
        # every peer's slot post-scale (the third revision of the elastic-
        # EP timeout investigation: with Mooncake PR #1968 commit 3c068028d
        # in pg_2_9_1.so, the allgather should be functional for all-active
        # max_world_size=actual_size; this probe gives ground truth).
        # Format per slot (6 ints):
        #   [num_tokens, num_tokens_for_logprob, can_cuda_graph,
        #    is_extend_in_batch, local_can_run_tbo, local_forward_mode]
        # IDLE.value=4. So a slot of [0,0,1,0,1,4] is the prefill-fallback
        # (un-written by the collective) — if a rank that we KNOW had
        # tokens shows that, the allgather skipped its slot.
        # Toggle SGLANG_DEBUG_MLP_SYNC=1.
        import os as _os
        if _os.environ.get("SGLANG_DEBUG_MLP_SYNC", "0") == "1":
            import logging as _logging
            try:
                _local_cpu = local_info_tensor.detach().cpu().tolist()
                _gflat = global_info_tensor.detach().cpu().view(-1, 6).tolist()
                _my_rank = torch.distributed.get_rank(group)
                _ws = torch.distributed.get_world_size(group)
                _logging.getLogger(__name__).info(
                    "[mlp-sync][probe] rank=%d world_size=%d dp_size=%d "
                    "local=%s global_per_slot=%s",
                    _my_rank, _ws, self.dp_size, _local_cpu, _gflat,
                )
            except Exception as _e:
                _logging.getLogger(__name__).warning(
                    "[mlp-sync][probe] failed: %s", _e,
                )

        # Set fallback values for inactive ranks (based on TP group's
        # active_ranks view — when the gather ran over WORLD, the prefill
        # above already covers missing slots).
        tp_info = global_info_tensor.view(self.dp_size * self.tp_size * self.cp_size, 6)
        num_ranks_in_tp_info = tp_info.shape[0]
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks
        if tp_active_ranks.shape[0] < num_ranks_in_tp_info:
            tp_active_ranks = torch.ones(
                num_ranks_in_tp_info, dtype=tp_active_ranks.dtype,
                device=tp_active_ranks.device,
            )
        tp_info[tp_active_ranks[:num_ranks_in_tp_info] == 0] = fallback_tensor

        tp0_info = global_info_tensor[:, 0, :]
        self.tp0_info = tp0_info
        # Perform only one Device-to-Host (D2H) memory copy
        cpu_data = tp0_info[:, :2].cpu()
        self.global_num_tokens = cpu_data[:, 0].tolist()
        self.global_num_tokens_for_logprob = cpu_data[:, 1].tolist()
        self.can_cuda_graph = bool(tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(tp0_info[:, 5].tolist())


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt():
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    # After elastic scale, use the Mooncake PG WORLD group for all_gather
    # so all 8 ranks participate. BUT: joiner during init uses LOCAL TP group
    # (can't join the 8-rank all_gather until adopted by primary's controller).
    from sglang.srt.layers.dp_attention import (
        _USE_WORLD_GROUP_FOR_DP_GATHER,
        _ELASTIC_JOINER_SKIP_ALL_GATHER,
    )
    if _USE_WORLD_GROUP_FOR_DP_GATHER and not _ELASTIC_JOINER_SKIP_ALL_GATHER:
        from sglang.srt.distributed.parallel_state import get_world_group
        world = get_world_group()
        # Use the DEFAULT Mooncake WORLD process group, not SGLang's
        # GroupCoordinator-created `world.device_group`. The latter is a
        # `new_group(ranks=[0..3])` created on primary at startup and never
        # grows; the default process group is the one recover_ranks() updates.
        group = torch.distributed.group.WORLD
        device = world.device
        _branch = "WORLD"
    elif len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
        _branch = "TP-device"
    else:
        group = tp_group.cpu_group
        device = "cpu"
        _branch = "TP-cpu"

    # Branch diagnostic: pair with [mlp-sync][probe] to see WHY primary
    # is selecting the wrong group post-scale (the probe confirms its
    # group is 4-rank but doesn't tell us which branch was taken nor
    # what _USE_WORLD_GROUP_FOR_DP_GATHER actually evaluated to).
    import os as _os_branch
    if _os_branch.environ.get("SGLANG_DEBUG_MLP_SYNC", "0") == "1":
        import logging as _logging_branch
        _logging_branch.getLogger(__name__).info(
            "[mlp-sync][branch] picked=%s _USE_WORLD_GROUP=%s "
            "_ELASTIC_JOINER_SKIP=%s dp_size=%d tp_size=%d cp_size=%d "
            "group_size=%d use_all_reduce=%s disable_overlap=%s offload_tags=%s",
            _branch,
            _USE_WORLD_GROUP_FOR_DP_GATHER,
            _ELASTIC_JOINER_SKIP_ALL_GATHER,
            dp_size, attn_tp_size, attn_cp_size,
            torch.distributed.get_world_size(group),
            _branch == "WORLD",
            disable_overlap_schedule,
            sorted(offload_tags) if offload_tags else [],
        )

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
    )

    if not skip_all_gather:
        mlp_sync_info.all_gather(
            device=device,
            group=group,
            use_all_reduce=_branch == "WORLD",
        )

        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info[:, 4:6],
            )
        )

    need_idle_batch = skip_all_gather or max(mlp_sync_info.global_num_tokens) > 0
    if need_idle_batch:
        batch_to_gather = local_batch
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


class SchedulerDPAttnMixin:
    def prepare_mlp_sync_batch(self: Scheduler, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            attn_cp_size=self.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
        )

    def maybe_prepare_mlp_sync_batch(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.require_mlp_sync for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.require_mlp_sync:
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def get_idle_batch(self: Scheduler) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
