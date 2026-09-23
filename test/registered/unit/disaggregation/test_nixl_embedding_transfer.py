"""CPU transport-contract tests; no SGLang, CUDA or NIXL installation required."""

import base64
import importlib.util
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/disaggregation/encoder/transfer.py"
)
spec = importlib.util.spec_from_file_location("encoder_transfer", PATH)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def session(name="prefill", device=3):
    return json.dumps(
        {
            "backend": "nixl",
            "version": 1,
            "agent": name,
            "device": device,
            "memory_type": "VRAM",
            "metadata": base64.b64encode(name.encode()).decode(),
        }
    )


class TestNixlEmbeddingTransfer(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.device_module = Mock()
        self.agent.name = "local"
        self.agent.register_memory.side_effect = lambda desc, kind: (desc, kind)
        self.agent.get_agent_metadata.return_value = b"metadata"
        self.agent.add_remote_agent.side_effect = lambda md: md.decode()
        self.agent.get_xfer_descs.side_effect = lambda desc, kind: (desc, kind)
        self.agent.initialize_xfer.side_effect = lambda *args: object()
        self.agent.transfer.return_value = "DONE"
        self.agent.check_xfer_state.return_value = "DONE"
        self.agent.get_plugin_list.return_value = ["UCX"]
        api = SimpleNamespace(
            nixl_agent=Mock(return_value=self.agent),
            nixl_agent_config=Mock(),
            nixl_thread_sync_t=SimpleNamespace(NIXL_THREAD_SYNC_STRICT="strict"),
        )
        fake_envs = SimpleNamespace(
            SGLANG_DISAGGREGATION_NIXL_BACKEND=SimpleNamespace(get=lambda: "UCX"),
            SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS=SimpleNamespace(get=lambda: "{}"),
            SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT=SimpleNamespace(get=lambda: 1),
        )
        modules = {
            "nixl._api": api,
            "torch": SimpleNamespace(
                get_device_module=Mock(return_value=self.device_module)
            ),
            "sglang": SimpleNamespace(),
            "sglang.srt": SimpleNamespace(),
            "sglang.srt.environ": SimpleNamespace(envs=fake_envs),
            "sglang.srt.utils": SimpleNamespace(),
            "sglang.srt.utils.common": SimpleNamespace(
                run_with_deadline=lambda function, **_: function()
            ),
        }
        with patch.dict("sys.modules", modules):
            self.engine = transport.NixlEmbeddingTransferEngine(2, timeout=0.01)
        self.agent.create_backend.assert_called_once_with("UCX", {})
        self.device_module.set_device.assert_called_once_with(2)
        self.engine.register(1000, 1024)

    def test_registration_reference_counts(self):
        self.engine.register(1000, 1024)
        self.agent.register_memory.assert_called_once_with(
            [(1000, 1024, 2, "")], "VRAM"
        )
        self.engine.deregister(1000)
        self.agent.deregister_memory.assert_not_called()
        self.engine.deregister(1000)
        self.agent.deregister_memory.assert_called_once()
        self.assertFalse(self.engine._registrations)

    def test_registration_errors_and_size_mismatch(self):
        with self.assertRaises(ValueError):
            self.engine.register(1000, 2048)
        self.agent.register_memory.side_effect = RuntimeError("registration failed")
        with self.assertRaises(RuntimeError):
            self.engine.register(4000, 32)
        self.assertNotIn(4000, self.engine._registrations)

    def test_metadata_is_cached_and_refreshed(self):
        before = self.engine.session_id
        self.assertEqual(before, self.engine.session_id)
        self.agent.get_agent_metadata.assert_called_once()
        self.agent.get_agent_metadata.return_value = b"updated"
        self.engine.register(4000, 32)
        self.assertNotEqual(before, self.engine.session_id)
        self.engine.deregister(4000)
        refreshed = self.engine.session_id
        self.assertIsInstance(refreshed, str)
        self.assertEqual(self.agent.get_agent_metadata.call_count, 3)

    def test_write_offsets_and_distinct_gpu_ids(self):
        self.agent.add_remote_agent.side_effect = lambda md: md
        self.assertEqual(self.engine.transfer_sync(session(), 1016, 8016, 512), 0)
        args = self.agent.initialize_xfer.call_args.args
        self.assertEqual(
            args,
            (
                "WRITE",
                ([(1016, 512, 2)], "VRAM"),
                ([(8016, 512, 3)], "VRAM"),
                "prefill",
            ),
        )
        self.agent.release_xfer_handle.assert_called_once()
        self.agent.remove_remote_agent.assert_not_called()

    def test_zero_bytes_and_unregistered_source(self):
        self.assertEqual(self.engine.transfer_sync(None, 0, 0, 0), 0)
        with self.assertRaises(ValueError):
            self.engine.transfer_sync(session(), 1900, 8000, 512)
        self.agent.transfer.assert_not_called()

    def test_reject_foreign_session(self):
        with self.assertRaises(ValueError):
            self.engine.transfer_sync('{"backend":"mooncake"}', 1000, 8000, 1)
        self.agent.add_remote_agent.assert_not_called()

    def test_delayed_completion(self):
        self.agent.transfer.return_value = "PROC"
        self.agent.check_xfer_state.side_effect = ["PROC", "DONE"]
        self.engine.transfer_sync(session(), 1000, 8000, 512)
        self.assertEqual(self.agent.check_xfer_state.call_count, 2)
        self.agent.release_xfer_handle.assert_called_once()

    def test_transfer_failure_releases_handle_and_peer(self):
        for failure in ["ERR", RuntimeError("native failure")]:
            with self.subTest(failure=failure):
                self.agent.transfer.reset_mock(side_effect=True)
                if isinstance(failure, Exception):
                    self.agent.transfer.side_effect = failure
                else:
                    self.agent.transfer.return_value = failure
                with self.assertRaises(RuntimeError):
                    self.engine.transfer_sync(session(), 1000, 8000, 512)
        self.assertEqual(self.agent.release_xfer_handle.call_count, 2)
        self.agent.add_remote_agent.assert_called_once()
        self.agent.remove_remote_agent.assert_not_called()

    def test_prepare_failure_removes_peer_without_handle(self):
        self.agent.initialize_xfer.side_effect = RuntimeError("prepare failed")
        with self.assertRaises(RuntimeError):
            self.engine.transfer_sync(session(), 1000, 8000, 512)
        self.agent.release_xfer_handle.assert_not_called()
        self.agent.remove_remote_agent.assert_not_called()

    def test_timeout_cancels_before_returning(self):
        self.agent.transfer.return_value = "PROC"
        self.agent.check_xfer_state.return_value = "PROC"
        with self.assertRaises(TimeoutError):
            self.engine.transfer_sync(session(), 1000, 8000, 512)
        self.agent.release_xfer_handle.assert_called_once()
        self.agent.remove_remote_agent.assert_not_called()

    def test_uncancellable_transfer_drains_before_cleanup(self):
        self.agent.transfer.return_value = "ERR"
        self.agent.release_xfer_handle.side_effect = [RuntimeError("in progress"), None]
        with (
            self.assertLogs(transport.logger, level="WARNING"),
            self.assertRaises(RuntimeError),
        ):
            self.engine.transfer_sync(session(), 1000, 8000, 512)
        deadline = time.monotonic() + 1
        while self.agent.release_xfer_handle.call_count < 2:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        calls = [c[0] for c in self.agent.mock_calls]
        self.assertNotIn("remove_remote_agent", calls)

    def test_permanently_uncancellable_transfer_quarantines_peer(self):
        owner = object()
        self.agent.transfer.return_value = "PROC"
        self.agent.check_xfer_state.return_value = "PROC"
        self.agent.release_xfer_handle.side_effect = RuntimeError("in progress")
        started = time.monotonic()
        with (
            self.assertLogs(transport.logger, level="WARNING"),
            self.assertRaises(transport.NixlTransferOutcomeUncertain),
        ):
            self.engine.transfer_sync(session(), 1000, 8000, 512, source_owner=owner)
        self.assertLess(time.monotonic() - started, 0.2)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self.engine.transfer_sync(session(), 1000, 9000, 512)
        deadline = time.monotonic() + 1
        while not self.engine._quarantined_handles:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        self.assertIs(self.engine._quarantined_handles[0][2], owner)

    def test_same_peer_metadata_lifetime_is_serialized(self):
        started, finish = threading.Event(), threading.Event()
        first_handle = []

        def post(handle):
            if not first_handle:
                first_handle.append(handle)
                started.set()
                return "PROC"
            return "DONE"

        def check(handle):
            return "DONE" if finish.is_set() else "PROC"

        self.engine.timeout = 2
        self.agent.transfer.side_effect = post
        self.agent.check_xfer_state.side_effect = check
        with ThreadPoolExecutor(2) as executor:
            one = executor.submit(self.engine.transfer_sync, session(), 1000, 8000, 512)
            self.assertTrue(started.wait(1))
            two = executor.submit(self.engine.transfer_sync, session(), 1000, 9000, 512)
            self.agent.remove_remote_agent.assert_not_called()
            finish.set()
            self.assertEqual(one.result(2), 0)
            self.assertEqual(two.result(2), 0)
        self.agent.add_remote_agent.assert_called_once()
        self.agent.remove_remote_agent.assert_not_called()
        self.assertIn("prefill", self.engine._peers)

    def test_changed_registration_metadata_reloads_peer(self):
        self.engine.transfer_sync(session(), 1000, 8000, 512)
        changed = json.loads(session())
        changed["metadata"] = base64.b64encode(b"prefill-new").decode()
        self.agent.add_remote_agent.side_effect = ["prefill"]
        self.engine.transfer_sync(json.dumps(changed), 1000, 9000, 512)
        self.agent.remove_remote_agent.assert_called_once_with("prefill")
        self.assertEqual(self.agent.add_remote_agent.call_count, 2)

    def test_failed_metadata_replacement_does_not_poison_cache(self):
        self.engine.transfer_sync(session(), 1000, 8000, 512)
        changed = json.loads(session())
        changed["metadata"] = base64.b64encode(b"prefill-new").decode()
        self.agent.add_remote_agent.side_effect = RuntimeError("import failed")
        with self.assertRaisesRegex(RuntimeError, "import failed"):
            self.engine.transfer_sync(json.dumps(changed), 1000, 9000, 512)
        peer_entry = self.engine._peers["prefill"]
        self.assertIsNone(peer_entry[1])
        self.assertFalse(peer_entry[2])

        self.agent.add_remote_agent.side_effect = lambda _: "prefill"
        self.assertEqual(
            self.engine.transfer_sync(json.dumps(changed), 1000, 9000, 512), 0
        )

    def test_mismatched_imported_peer_is_removed(self):
        self.agent.add_remote_agent.return_value = "unexpected-peer"
        self.agent.add_remote_agent.side_effect = None
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.engine.transfer_sync(session(), 1000, 8000, 512)
        self.agent.remove_remote_agent.assert_called_once_with("unexpected-peer")
        peer_entry = self.engine._peers["prefill"]
        self.assertIsNone(peer_entry[1])
        self.assertFalse(peer_entry[2])


if __name__ == "__main__":
    unittest.main()
