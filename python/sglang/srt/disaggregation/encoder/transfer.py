"""Buffer transport shared by the HTTP encoder remote-write delivery path."""

import base64
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager

logger = logging.getLogger(__name__)

REMOTE_WRITE_ENCODER_BACKENDS = ("mooncake", "nixl")
NIXL_OUTCOME_UNCERTAIN_HEADER = "X-SGLang-NIXL-Transfer-Outcome-Uncertain"


class NixlTransferOutcomeUncertain(RuntimeError):
    """The NIXL request may still access both source and destination memory."""


class NixlEmbeddingTransferEngine:
    """NIXL adapter for GPU embeddings, independent of the PD KV manager.

    Callers own tensor storage and must retain it through transfer completion.
    DRAM support is for transport smoke tests; EPD uses VRAM.
    """

    def __init__(self, gpu_id=0, *, memory_type="VRAM", timeout=300.0, initiator=False):
        try:
            from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
        except ImportError as exc:
            raise ImportError(
                "Install nixl to use --encoder-transfer-backend nixl."
            ) from exc
        if memory_type not in ("VRAM", "DRAM"):
            raise ValueError(f"Unsupported embedding memory type: {memory_type}")
        if timeout <= 0:
            raise ValueError("NIXL transfer timeout must be positive")
        self.gpu_id = gpu_id if memory_type == "VRAM" else 0
        self.memory_type = memory_type
        self.timeout = timeout
        requested_name = f"sglang-encoder-{uuid.uuid4()}"
        # Match SGLang's PD NIXL setup: use the configured plugin and params,
        # create it explicitly, and enable strict synchronization because the
        # encoder transport is entered from HTTP and scheduler worker threads.
        from sglang.srt.environ import envs
        from sglang.srt.utils.common import run_with_deadline

        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        backend_params = json.loads(
            envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get()
        )
        if not isinstance(backend_params, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in backend_params.items()
        ):
            raise ValueError(
                "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
                "with string keys and string values"
            )
        num_threads = 8 if initiator else 0
        self.agent = nixl_agent(
            requested_name,
            nixl_agent_config(
                backends=[],
                num_threads=num_threads,
                sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
            ),
        )
        # NIXL may canonicalize the requested identity. Metadata import returns
        # this value, so it is the only safe name for the control contract.
        self.name = self.agent.name
        if num_threads:
            if backend in ("UCX", "OBJ"):
                backend_params.setdefault("num_threads", str(num_threads))
            elif backend == "GDS_MT":
                backend_params.setdefault("thread_count", str(num_threads))
            elif backend == "UCCL":
                backend_params.setdefault("num_cpus", str(num_threads))

        def create_backend():
            if self.memory_type == "VRAM":
                import torch

                torch.get_device_module("cuda").set_device(self.gpu_id)
            return self.agent.create_backend(backend, backend_params)

        run_with_deadline(
            create_backend,
            timeout_s=envs.SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT.get(),
            what=f"NIXL encoder create_backend({backend!r}, {backend_params})",
        )
        available = self.agent.get_plugin_list()
        if backend not in available:
            raise ValueError(
                f"NIXL backend {backend!r} not found. Available: {available}"
            )
        self.backend = backend
        # Serialize API entry (including Python wrapper state), but not polling
        # sleeps. Distinct peers can have outstanding transfers simultaneously.
        self._lock = threading.RLock()
        self._registrations = {}
        self._peers = {}
        self._quarantined_handles = []
        self._session = None

    def register(self, ptr, length):
        if ptr <= 0 or length <= 0:
            raise ValueError(
                "Embedding registrations require positive address and size"
            )
        with self._lock:
            existing = self._registrations.get(ptr)
            if existing is not None:
                if existing[0] != length:
                    raise ValueError("Cannot resize a live embedding registration")
                existing[2] += 1
                return
            desc = self.agent.register_memory(
                [(ptr, length, self.gpu_id, "")], self.memory_type
            )
            if desc is None:
                raise RuntimeError("NIXL embedding memory registration failed")
            self._registrations[ptr] = [length, desc, 1]
            self._session = None

    def deregister(self, ptr):
        with self._lock:
            region = self._registrations[ptr]
            if region[2] > 1:
                region[2] -= 1
                return
            self.agent.deregister_memory(region[1])
            del self._registrations[ptr]
            self._session = None

    @property
    def session_id(self):
        # Mooncake's session is an address; NIXL needs connection and MR metadata.
        # The existing HTTP /send session_id is opaque to the delivery protocol.
        with self._lock:
            if self._session is None:
                self._session = json.dumps(
                    {
                        "backend": "nixl",
                        "version": 1,
                        "agent": self.name,
                        "device": self.gpu_id,
                        "memory_type": self.memory_type,
                        "metadata": base64.b64encode(
                            self.agent.get_agent_metadata()
                        ).decode("ascii"),
                    },
                    separators=(",", ":"),
                )
            return self._session

    @contextmanager
    def _peer(self, name):
        # Metadata may describe a per-request registration that has since been
        # replaced at the same address. Update it only between transfers. Keep
        # stable peers installed, as the PD backend does for fixed KV pools;
        # the default embedding pool likewise has stable registration metadata.
        with self._lock:
            entry = self._peers.setdefault(name, [threading.Lock(), None, False, False])
        with entry[0]:
            if entry[3]:
                raise RuntimeError(
                    f"NIXL peer {name!r} is quarantined after an uncancellable transfer"
                )
            yield entry

    def _pin_registration(self, source_address, size):
        with self._lock:
            for ptr, region in self._registrations.items():
                if ptr <= source_address and source_address + size <= ptr + region[0]:
                    region[2] += 1
                    return ptr
        raise ValueError("Source embedding range is not registered")

    def _quarantine_handle(self, handle, peer_entry, registration_ptr, source_owner):
        """Drain an uncancellable request without blocking the serving thread.

        The registration pin and ``source_owner`` keep DMA memory alive. The
        affected peer fails fast until cleanup succeeds; other peers continue.
        """
        peer_entry[3] = True

        def drain():
            deadline = time.monotonic() + min(self.timeout, 30.0)
            while time.monotonic() < deadline:
                try:
                    with self._lock:
                        self.agent.release_xfer_handle(handle)
                except Exception:  # noqa: BLE001 - NIXL bindings use backend-specific exceptions
                    try:
                        with self._lock:
                            self.agent.check_xfer_state(handle)
                    except Exception:
                        logger.debug("NIXL transfer still settling", exc_info=True)
                    time.sleep(0.001)
                    continue

                self.deregister(registration_ptr)
                with self._lock:
                    peer_entry[3] = False
                return

            # Intentionally retain the handle, registration pin, and tensor.
            # Reusing or freeing them while DMA may still be active is unsafe.
            with self._lock:
                self._quarantined_handles.append(
                    (handle, registration_ptr, source_owner)
                )
            logger.error(
                "NIXL embedding transfer did not settle within %.1fs; "
                "the peer remains quarantined",
                min(self.timeout, 30.0),
            )

        logger.warning(
            "NIXL embedding transfer could not be cancelled; draining it in "
            "the background and quarantining the peer",
        )
        threading.Thread(
            target=drain,
            name="nixl-encoder-transfer-drain",
            daemon=True,
        ).start()

    def _release_or_quarantine(
        self, handle, peer_entry, registration_ptr, source_owner
    ):
        try:
            with self._lock:
                self.agent.release_xfer_handle(handle)
            return True
        except Exception:  # noqa: BLE001 - NIXL bindings use backend-specific exceptions
            self._quarantine_handle(handle, peer_entry, registration_ptr, source_owner)
            return False

    def transfer_sync(
        self,
        session_id,
        source_address,
        destination_address,
        size,
        *,
        source_owner=None,
    ):
        if size < 0:
            raise ValueError("Embedding transfer size must be nonnegative")
        if size == 0:
            return 0
        remote = json.loads(session_id)
        if remote.get("backend") != "nixl" or remote.get("version") != 1:
            raise ValueError("Expected a version 1 NIXL encoder session")
        if remote["memory_type"] not in ("VRAM", "DRAM"):
            raise ValueError("Unsupported remote embedding memory type")
        metadata = base64.b64decode(remote["metadata"], validate=True)
        with self._peer(remote["agent"]) as peer_entry:
            registration_ptr = self._pin_registration(source_address, size)
            peer = None
            handle = None
            try:
                with self._lock:
                    if peer_entry[1] != metadata:
                        if peer_entry[2]:
                            self.agent.remove_remote_agent(remote["agent"])
                            peer_entry[1] = None
                            peer_entry[2] = False
                        peer = self.agent.add_remote_agent(metadata)
                        if isinstance(peer, bytes):
                            peer = peer.decode("utf-8")
                        if peer != remote["agent"]:
                            try:
                                self.agent.remove_remote_agent(peer)
                            except Exception:
                                logger.warning(
                                    "Failed to remove mismatched NIXL peer %r",
                                    peer,
                                    exc_info=True,
                                )
                            raise ValueError(
                                "NIXL metadata does not match remote agent: "
                                f"metadata={remote['agent']!r}, imported={peer!r}"
                            )
                        peer_entry[1] = metadata
                        peer_entry[2] = True
                    else:
                        peer = remote["agent"]
                    source = self.agent.get_xfer_descs(
                        [(source_address, size, self.gpu_id)], self.memory_type
                    )
                    destination = self.agent.get_xfer_descs(
                        [(destination_address, size, remote["device"])],
                        remote["memory_type"],
                    )
                    handle = self.agent.initialize_xfer(
                        "WRITE", source, destination, peer
                    )
                    state = self.agent.transfer(handle)
                deadline = time.monotonic() + self.timeout
                while state == "PROC":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("NIXL encoder transfer timed out")
                    time.sleep(0.0001)
                    with self._lock:
                        state = self.agent.check_xfer_state(handle)
                if state != "DONE":
                    raise RuntimeError(f"NIXL encoder transfer failed: {state}")
                return 0
            finally:
                if handle is not None and not self._release_or_quarantine(
                    handle, peer_entry, registration_ptr, source_owner
                ):
                    registration_ptr = None
                    raise NixlTransferOutcomeUncertain(
                        "NIXL transfer could not be cancelled; source and "
                        "destination memory have been quarantined"
                    )
                if registration_ptr is not None:
                    self.deregister(registration_ptr)
