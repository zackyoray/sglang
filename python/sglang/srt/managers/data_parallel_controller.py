# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BlockReqInput,
    ElasticScaleWorkerPortsReq,
    ProfileReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    configure_logger,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.network import (
    NetworkAddress,
    bind_port,
    get_zmq_socket,
    get_zmq_socket_on_host,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    """Per-rank load counters used by total_requests / total_tokens dispatchers.

    Sized to ``max_dp_size`` once at construction; an optional ``active_mask``
    filters dispatch decisions to currently-joined ranks. Inactive slots are
    reported as load=infinity so the ``min(...)`` selection skips them.
    """

    def __init__(self, dp_size: int, active_mask: Optional[List[bool]] = None):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        self.active_mask = (
            list(active_mask) if active_mask is not None else [True] * dp_size
        )

    def refresh_active_mask(self, mask: List[bool]) -> None:
        """Replace the active mask. Caller invokes this on scale events."""
        assert len(mask) == self.dp_size, (
            f"active mask length {len(mask)} != dp_size {self.dp_size}"
        )
        self.active_mask = list(mask)

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """Update the budget."""
        for load in load_update.loads:
            self.total_requests[load.dp_rank] = load.num_reqs
            self.total_tokens[load.dp_rank] = load.num_tokens

    def dispatch(self, method: LoadBalanceMethod):
        # Sentinel "do not pick me" load for inactive slots.
        INACTIVE = float("inf")
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            scored = [
                self.total_requests[i] if self.active_mask[i] else INACTIVE
                for i in range(self.dp_size)
            ]
            target_rank = scored.index(min(scored))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal.
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (
                    (INACTIVE, INACTIVE)
                    if not self.active_mask[i]
                    else (self.total_tokens[i], self.total_requests[i])
                ),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        return target_rank


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # For DP balance
        self.global_balance_id = 0

        # Init inter-process communication
        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Recovery-style Phase B: pre-allocate all elasticity-touched
        # bookkeeping to ``max_dp_size`` (the ceiling set by --max-ep-size)
        # so scale events flip activation bits on fixed-shape structures
        # rather than appending. ``max_dp_size`` falls back to ``dp_size``
        # for non-elastic deployments.
        self.launch_dp_size: int = server_args.dp_size
        self.max_dp_size: int = server_args.max_ep_size or server_args.dp_size
        assert self.max_dp_size >= self.launch_dp_size, (
            f"--max-ep-size ({self.max_dp_size}) must be >= "
            f"--dp ({self.launch_dp_size})."
        )

        # Active mask: launch slots are active immediately, future slots
        # remain False until a scale event fills them.
        self.dp_active: List[bool] = (
            [True] * self.launch_dp_size
            + [False] * (self.max_dp_size - self.launch_dp_size)
        )

        # Load balance budget sized to max_dp_size and aware of which
        # slots are currently joined. Refreshed on scale events.
        self.dp_budget = DPBudget(self.max_dp_size, active_mask=self.dp_active)

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        # Worker sockets are pre-allocated to max_dp_size; entries beyond
        # launch_dp_size start as None and are filled on add_elastic_workers.
        self.workers: List[Optional[zmq.Socket]] = [None] * self.max_dp_size
        # status historically tracked NIXL fault-detection per rank. Pre-
        # allocate to max_dp_size; future slots are False until joined,
        # which is consistent with the dp_active mask above.
        self.status: List[bool] = list(self.dp_active)
        # Cached list of currently-active slot indices for routing hot path.
        # Invalidated on every dp_active change via _refresh_active_workers.
        self._active_workers: List[int] = list(range(self.launch_dp_size))
        self._active_count_cache: int = self.launch_dp_size

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            # When local control broadcast is enabled, send control messages to
            # every DP group leader (attn_tp_rank=0) so each leader broadcasts
            # within its own attn_tp_group instead of the full tp_group.
            # Otherwise fall back to the original behaviour: send to only the
            # first leader, which then broadcasts over the full tp_group.
            local_ctrl = server_args.enable_dp_attention_local_control_broadcast
            self.control_message_step = 1 if local_ctrl else server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        # Iterates the full max-sized list; skips inactive (None) slots.
        # status[i] is False for both NIXL-faulted and not-yet-joined slots.
        for i, worker in enumerate(self.workers):
            if worker is not None and self.status[i]:
                worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group. Walk
        # active workers only (slots beyond active count are None).
        for i in self._active_workers[:: self.control_message_step]:
            worker = self.workers[i]
            if worker is not None:
                worker.send_pyobj(obj)

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        # In elastic mode, ranks.status reflects only the primary TP view
        # and joiner-half faults live in the joiner's own active_ranks
        # tensor. Keep registered workers routable by mirroring dp_active
        # (which tracks "is this slot joined?") rather than overwriting
        # with ranks.status. Non-elastic mode uses the direct assignment.
        if self.server_args.elastic_ep_backend is not None:
            self.status = list(self.dp_active)
            return
        # Non-elastic: dp_size == max_dp_size, ranks.status length matches.
        if len(ranks.status) != self.max_dp_size:
            logger.warning(
                "[DPC] update_active_ranks: status len=%d != max_dp_size=%d; "
                "ignoring update",
                len(ranks.status), self.max_dp_size,
            )
            return
        self.status = list(ranks.status)

    def add_elastic_workers(
        self, new_worker_ports: List[int], slot_offset: int = 0
    ):
        """Activate joiner scheduler slots after elastic scale-up.

        Under Phase E, ``self.workers`` is pre-bound to deterministic
        addresses at launch (see launch_dp_attention_schedulers). Joiner
        schedulers PULL from those addresses directly. This method is now
        a pure bit-flip: mark ``dp_active[slot] = True``, refresh the
        budget, no socket creation.

        ``new_worker_ports`` is retained in the message for backwards
        compatibility but is unused under Phase E (the addresses are
        already known to both ends via DP_PREBIND_PORT_DELTA).
        """
        if slot_offset == 0 and any(self.dp_active[: len(new_worker_ports)]):
            # Backwards-compat: no slot_offset and slot 0 already active
            # means the caller is on the legacy code path. Find the first
            # inactive slot and use that as the offset.
            slot_offset = self.dp_active.index(False)

        end = slot_offset + len(new_worker_ports)
        if end > self.max_dp_size:
            raise ValueError(
                f"[Elastic EP] add_elastic_workers: "
                f"slot_offset={slot_offset} + len(ports)={len(new_worker_ports)} "
                f"exceeds max_dp_size={self.max_dp_size}. Restart with a "
                f"larger --max-ep-size."
            )

        for i in range(len(new_worker_ports)):
            slot = slot_offset + i
            assert not self.dp_active[slot], (
                f"[Elastic EP] add_elastic_workers: slot {slot} already active"
            )
            self.dp_active[slot] = True
            self.status[slot] = True

        self.dp_budget.refresh_active_mask(self.dp_active)
        self._refresh_active_workers()
        logger.info(
            "[Elastic EP] DataParallelController activated slots %s "
            "(active=%d / max=%d)",
            list(range(slot_offset, end)),
            self._active_count_cache,
            self.max_dp_size,
        )

    def _refresh_active_workers(self) -> None:
        """Recompute the cached active-worker index list and count."""
        self._active_workers = [
            i for i, active in enumerate(self.dp_active) if active
        ]
        self._active_count_cache = len(self._active_workers)

    def active_dp_count(self) -> int:
        """Number of currently-joined DP workers (read-mostly)."""
        return self._active_count_cache

    def dispatching_with_trace(self, req: Req):
        req.time_stats = DPControllerReqTimeStats.new_from_obj(req.time_stats)

        req.time_stats.set_dp_dispatch_time()
        self.dispatching(req)
        req.time_stats.set_dp_dispatch_finish_time()

    def dispatch_batch_generate(self, batch_req: BatchTokenizedGenerateReqInput):
        for req in batch_req:
            self.dispatching_with_trace(req)

    def dispatch_batch_embedding(self, batch_req: BatchTokenizedEmbeddingReqInput):
        for req in batch_req:
            self.dispatching_with_trace(req)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BatchTokenizedGenerateReqInput, self.dispatch_batch_generate),
                (BatchTokenizedEmbeddingReqInput, self.dispatch_batch_embedding),
                (BlockReqInput, self.send_to_all_workers),
                (ProfileReq, self.send_to_all_workers),
                (WatchLoadUpdateReq, self.handle_load_update_req),
                (ActiveRanksOutput, self.update_active_ranks),
                (
                    ElasticScaleWorkerPortsReq,
                    lambda msg: self.add_elastic_workers(
                        msg.new_worker_ports, slot_offset=msg.slot_offset
                    ),
                ),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        # Determine the endpoint for inter-node communication.
        # Elastic EP joiners share the primary's dist_init_addr for the PG
        # Store, but derive DP-handshake and ZMQ ports from their own HTTP
        # port to avoid collisions with the primary.
        is_joiner = server_args.is_ep_joiner
        if server_args.dist_init_addr is None or is_joiner:
            na = NetworkAddress(
                server_args.host or "127.0.0.1",
                server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA,
            )
        else:
            na = NetworkAddress.parse(server_args.dist_init_addr)
            na = NetworkAddress(na.host, na.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA)
        endpoint = na.to_tcp()

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                rep_socket.send_pyobj(worker_ports)
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            if self.server_args.elastic_ep_backend is None:
                rep_socket.close()
            else:
                threading.Thread(
                    target=self._reply_ports_as_server,
                    args=(rep_socket, worker_ports),
                    daemon=True,
                ).start()

    def _reply_ports_as_server(self, rep_socket: zmq.Socket, worker_ports: List[int]):
        """
        Runs as a background thread to broadcast worker ports for recovered EP ranks
        """
        while True:
            # Wait for client handshake
            try:
                client_rank = rep_socket.recv().decode()
            except Exception:
                logger.exception(
                    "Failed to recv/decode handshake in reply thread; continue"
                )
                continue
            logger.debug(f"Received handshake from node {client_rank}")

            self._release_elastic_worker_sockets_for_primary()

            # Send worker ports to client
            rep_socket.send_pyobj(worker_ports)
            logger.debug(f"Sent worker ports to node {client_rank}")

    def _release_elastic_worker_sockets_for_primary(self):
        """Let the primary controller take ownership of joiner worker endpoints."""
        if not self.server_args.is_ep_joiner:
            return
        if getattr(self, "_elastic_worker_sockets_released", False):
            return

        for i, worker in enumerate(self.workers):
            if worker is None:
                continue
            try:
                worker.close(linger=0)
                logger.info(
                    "[Elastic EP] Released joiner local worker socket dp_rank=%d "
                    "for primary controller binding",
                    i,
                )
            except (zmq.ZMQError, OSError):
                # ZMQError covers the bound-socket close path; OSError covers
                # bad-fd or stuck-socket teardown. Anything outside this is
                # an actual bug we want to surface, not swallow.
                logger.exception(
                    "[Elastic EP] Failed to release joiner worker socket dp_rank=%d",
                    i,
                )
            self.workers[i] = None

        self._elastic_worker_sockets_released = True

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            req_socket.send(str(node_rank).encode())

            # Receive worker ports
            worker_ports = req_socket.recv_pyobj()
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        if server_args.dist_init_addr is None:
            bind_host = "127.0.0.1"
        else:
            bind_host = NetworkAddress.parse(server_args.dist_init_addr).host

        # Phase E: under elastic mode the primary pre-binds PUSH sockets at
        # deterministic addresses derived from dist_init_port. Joiner
        # schedulers PULL from these addresses directly (their PortArgs
        # computes the same addresses), so the bind-then-release handshake
        # is no longer needed. Phase B's pre-allocated self.workers fits
        # this naturally: slots [0, max_dp_size) are bound; [0, dp_size)
        # are active, [dp_size, max_dp_size) wait for joiners to activate.
        elastic_mode_active = (
            server_args.elastic_ep_backend is not None
            and server_args.max_ep_size is not None
            and server_args.max_ep_size > server_args.tp_size
        )
        worker_ports = []
        if server_args.node_rank == 0:
            if elastic_mode_active and not server_args.is_ep_joiner:
                from sglang.srt.server_args import DP_PREBIND_PORT_DELTA
                prebind_base = (
                    NetworkAddress.parse(server_args.dist_init_addr).port
                    + DP_PREBIND_PORT_DELTA
                )
                for slot in range(self.max_dp_size):
                    addr = NetworkAddress(
                        bind_host, prebind_base + slot
                    ).to_tcp()
                    sock = get_zmq_socket(
                        self.context, zmq.PUSH, addr, True
                    )
                    self.workers[slot] = sock
                    if slot < server_args.dp_size:
                        worker_ports.append(prebind_base + slot)
                    logger.info(
                        "[Elastic EP][Phase E] pre-bound DP slot %d at %s "
                        "(active=%s)",
                        slot, addr, slot < server_args.dp_size,
                    )
            else:
                # Non-elastic or joiner: today's auto-pick path (joiner DPC
                # still binds local sockets pre-Phase-E.3; will be removed).
                for dp_rank in range(server_args.dp_size):
                    worker_port, worker_socket = get_zmq_socket_on_host(
                        self.context, zmq.PUSH, host=bind_host
                    )
                    worker_ports.append(worker_port)
                    self.workers[dp_rank] = worker_socket

        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        attn_cp_rank = 0
        moe_dp_rank = 0
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                        server_args.attn_cp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    if server_args.is_ep_joiner:
                        # After adoption, the primary tokenizer owns the HTTP
                        # request state. Joiner schedulers still receive
                        # requests through their adopted worker endpoints, but
                        # their outputs must go back through the primary
                        # detokenizer/tokenizer path.
                        primary_addr = NetworkAddress.parse(server_args.dist_init_addr)
                        primary_port_base = primary_addr.port + 1
                        rank_port_args.tokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base
                        ).to_tcp()
                        rank_port_args.detokenizer_ipc_name = NetworkAddress(
                            primary_addr.host, primary_port_base + 1
                        ).to_tcp()
                        logger.info(
                            "[Elastic EP] Joiner DP%d outputs routed to primary "
                            "tokenizer=%s detokenizer=%s",
                            dp_rank,
                            rank_port_args.tokenizer_ipc_name,
                            rank_port_args.detokenizer_ipc_name,
                        )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                attn_dp_size = (
                    server_args.dp_size if server_args.enable_dp_attention else 1
                )

                # Parallelism hierarchy (outermost to innermost):
                # - Attention: Global(TP) -> DP -> ATTN_CP -> ATTN_TP (innermost)
                # - MoE: Global(TP) -> MOE_DP -> EP -> MOE_TP (innermost)
                attn_tp_size = (
                    server_args.tp_size // attn_dp_size // server_args.attn_cp_size
                )
                attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
                moe_dp_rank = tp_rank // (
                    server_args.tp_size // server_args.moe_dp_size
                )
                moe_ep_rank = (
                    tp_rank
                    % (server_args.tp_size // server_args.moe_dp_size)
                    // (
                        server_args.tp_size
                        // server_args.moe_dp_size
                        // server_args.ep_size
                    )
                )

                # Cosmetic global-rank overrides for the scheduler log prefix
                # and proctitle. Elastic-EP joiners launch with a non-zero
                # ep_join_rank_offset so a joiner with offset=4, tp=4 logs as
                # DP4..7 instead of DP0..3, matching how the primary refers to
                # its joiner peers (e.g. ranks_to_join=[4, 5]). Runtime ranks
                # passed below are unchanged so ZMQ binding, NIXL connect, and
                # Mooncake PG init still see local ranks.
                offset = server_args.ep_join_rank_offset
                display_tp_rank = tp_rank + offset
                display_moe_ep_rank = moe_ep_rank + offset
                display_dp_rank = (
                    dp_rank + offset if dp_rank is not None else None
                )

                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            attn_cp_rank,
                            moe_dp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                            display_tp_rank,
                            display_dp_rank,
                            display_moe_ep_rank,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                        server_args, gpu_id
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.routed_dp_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.routed_dp_rank}")
            self.workers[req.routed_dp_rank].send_pyobj(req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        # Walk only over active slots. self._active_workers is the cached
        # list of currently-joined slot indices; round_robin_counter is an
        # index into THAT list, not into the full self.workers.
        active = self._active_workers
        if not active:
            raise RuntimeError(
                "round_robin_scheduler: no active DP workers (cluster not joined?)"
            )
        attempts = 0
        while attempts < len(active):
            slot = active[self.round_robin_counter % len(active)]
            self.round_robin_counter = (self.round_robin_counter + 1) % len(active)
            if self.status[slot]:
                logger.debug(f"Choose worker {slot}")
                self.workers[slot].send_pyobj(req)
                return
            attempts += 1
        raise RuntimeError(
            f"round_robin_scheduler: all {len(active)} active DP workers "
            f"have status=False (NIXL-faulted?); cannot route request"
        )

    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        active = self._active_workers
        if not active:
            raise RuntimeError(
                "follow_bootstrap_room_scheduler: no active DP workers"
            )

        # Set default bootstrap_room if in FAKE auto mode and room is None
        if (
            req.bootstrap_room is None
            and self.server_args.disaggregation_transfer_backend == "fake"
        ):
            req.bootstrap_room = self.round_robin_counter
            self.round_robin_counter = (self.round_robin_counter + 1) % len(active)

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        # bootstrap_room hashes into the active workers list, not the full
        # max_dp_size list. After scale, this re-hashes (same as today's
        # append-style behavior — see plan.md Phase 11.x for consistent
        # hashing follow-up).
        target_rank = active[req.bootstrap_room % len(active)]
        self.workers[target_rank].send_pyobj(req)

    def total_requests_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        # DPBudget.dispatch already returns a slot index in [0, max_dp_size)
        # filtered by the active mask via INACTIVE sentinels.
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.workers[target_worker].send_pyobj(req)

    def total_tokens_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        self.workers[target_worker].send_pyobj(req)

    def event_loop(self):
        while True:
            while True:
                self.soft_watchdog.feed()
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        # Phase E: under elastic mode, joiner schedulers PULL from primary's
        # pre-bound deterministic addresses (PortArgs.init_new + DPC pre-bind).
        # The joiner-side DPC's event loop has nothing to do post-construction:
        # all routing goes through the primary DPC. Skip it; just supervise
        # the schedulers.
        if server_args.node_rank == 0 and not server_args.is_ep_joiner:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
