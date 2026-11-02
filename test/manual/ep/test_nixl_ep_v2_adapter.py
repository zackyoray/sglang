"""Dependency-isolated contract tests; no CUDA, NIXL, or PyTorch required.

Execute the adapter definitions with fake transport/tensor boundaries so these
checks also run on development hosts without the serving stack installed.
"""

import ast
import enum
import types
import unittest
from functools import cache
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)


class TestNixlEPV2(unittest.TestCase):
    def setUp(self):
        resources = types.SimpleNamespace(buffers={})
        dtype = type("dtype", (), {})
        self.int32, self.int64 = dtype(), dtype()
        self.transport = Mock()
        self.buffer = self.transport.return_value
        self.ns = dict(
            cache=cache,
            _SCALE_BLOCK_SIZE=128,
            _EXPERT_ALIGNMENT=128,
            torch=types.SimpleNamespace(dtype=dtype, int64=self.int64),
            dist=types.SimpleNamespace(
                get_world_size=lambda group: 2, get_rank=lambda group: 0
            ),
            logger=Mock(),
            get_exec=lambda: types.SimpleNamespace(
                moe=types.SimpleNamespace(elastic_ep_backend=None)
            ),
            get_parallel=lambda: types.SimpleNamespace(max_ep_size=8),
            get_global_tcp_store=Mock(return_value=object()),
            ElasticEPStateManager=types.SimpleNamespace(
                get_ep_join_rank_offset=lambda: 0
            ),
            envs=types.SimpleNamespace(
                SGLANG_NIXL_EP_V2_MAX_SCALEUP_RANKS=types.SimpleNamespace(get=lambda: 8)
            ),
            get_resources=lambda: resources,
            _ensure_deepep_v2_available=Mock(),
            BaseDispatcher=object,
        )
        load_definitions(
            ROOT / "layers/moe/token_dispatcher/deepep_v2.py",
            {"DeepEPv2Buffer", "_DeepEPv2Impl"},
            self.ns,
        )
        self.ns["DeepEPv2Dispatcher"] = type("Dispatcher", (), {})
        load_definitions(
            ROOT / "layers/moe/token_dispatcher/nixl_v2.py",
            {"NixlEPV2Buffer", "_NixlEPV2Impl", "NixlEPV2Dispatcher"},
            self.ns,
        )
        self.ns["_load_nixl_ep_v2"] = Mock(return_value=(self.transport, self.int32))
        self.facade = self.ns["NixlEPV2Buffer"]
        self.group = object()

    def get_buffer(self, **overrides):
        args = dict(
            group=self.group,
            hidden_size=128,
            router_topk=2,
            num_local_experts=4,
            num_max_dispatch_tokens_per_rank=32,
            use_fp8_dispatch=False,
        )
        args.update(overrides)
        return self.facade.get_buffer(**args)

    def test_constructor_matches_nixl_v2_api(self):
        self.get_buffer()
        self.transport.assert_called_once_with(
            group=self.group,
            hidden=128,
            num_topk=2,
            num_experts_per_rank=4,
            num_max_tokens_per_rank=32,
            use_fp8_dispatch=False,
            prefer_overlap_with_compute=False,
        )

    def test_same_configuration_reuses_buffer(self):
        self.assertIs(self.get_buffer(), self.get_buffer())
        self.transport.assert_called_once()

    def test_local_expert_count_is_part_of_cache_key(self):
        self.get_buffer()
        self.get_buffer(num_local_experts=8)
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual(self.transport.call_args.kwargs["num_experts_per_rank"], 8)

    def test_state_is_separate_from_deepep(self):
        self.get_buffer()
        self.assertIsNone(self.ns["DeepEPv2Buffer"]._state().buffer)

    def test_failed_constructor_is_not_cached(self):
        self.transport.side_effect = RuntimeError("construction failed")
        with self.assertRaisesRegex(RuntimeError, "construction failed"):
            self.get_buffer()
        self.assertIsNone(self.facade._state().buffer)
        self.assertIsNone(self.facade._state().key)

    def test_destroy_releases_facade_reference(self):
        self.get_buffer()
        self.facade.destroy()
        self.assertIsNone(self.facade._state().buffer)
        self.assertIsNone(self.facade._state().key)

    def test_dispatch_uses_library_index_dtype(self):
        impl = object.__new__(self.ns["_NixlEPV2Impl"])
        self.assertIs(impl._topk_indices_dtype(), self.int32)
        impl._ensure_available()
        self.ns["_ensure_deepep_v2_available"].assert_not_called()

    def test_dispatch_does_not_pass_removed_num_experts_argument(self):
        impl = object.__new__(self.ns["_NixlEPV2Impl"])
        impl.num_experts = 8
        x, ids = object(), object()
        impl._dispatch_buffer(self.buffer, x, topk_idx=ids, do_cpu_sync=False)
        self.buffer.dispatch.assert_called_once_with(x, topk_idx=ids, do_cpu_sync=False)

    def test_deepep_dispatch_keeps_num_experts_argument(self):
        impl = object.__new__(self.ns["_DeepEPv2Impl"])
        impl.num_experts = 8
        x = object()
        impl._dispatch_buffer(self.buffer, x, do_cpu_sync=False)
        self.buffer.dispatch.assert_called_once_with(
            x, num_experts=8, do_cpu_sync=False
        )
        self.assertIs(impl._topk_indices_dtype(), self.int64)

    def test_rejects_inconsistent_expert_slots(self):
        impl = object.__new__(self.ns["_NixlEPV2Impl"])
        impl.group = self.group
        impl.num_experts, impl.num_local_experts = 7, 4
        with self.assertRaisesRegex(ValueError, "equal expert slots"):
            impl._get_buffer()
        self.transport.assert_not_called()

    def test_masked_dispatch_combine_round_trip_contract(self):
        impl = object.__new__(self.ns["_NixlEPV2Impl"])
        impl.group = self.group
        impl.num_experts, impl.num_local_experts = 8, 4
        impl.hidden_size, impl.router_topk = 128, 2
        impl.activation_scale_block_size = 128
        impl.num_max_dispatch_tokens_per_rank = 32
        impl.use_fp8_dispatch = False
        impl._handle = None
        impl._prefill_expand_enabled = False
        self.ns["get_is_extend_in_batch"] = lambda: False
        self.ns["envs"] = types.SimpleNamespace(
            SGLANG_DEEPEP_V2_NUM_SMS=types.SimpleNamespace(get=lambda: 0)
        )
        # Named tuple construction is irrelevant to the wire protocol here.
        self.ns["DeepEPv2DispatchOutput"] = lambda *args, **kwargs: args
        x = types.SimpleNamespace(shape=(3, 128))
        indices = Mock()
        indices.to.return_value = types.SimpleNamespace(shape=(3, 2))
        weights = object()
        recv_x = types.SimpleNamespace(shape=(256, 128))
        recv_weights = types.SimpleNamespace(shape=(256,))
        handle = types.SimpleNamespace(psum_num_recv_tokens_per_expert=object())
        event = types.SimpleNamespace(event=None)
        self.buffer.dispatch.return_value = (recv_x, None, recv_weights, handle, event)
        result = impl.dispatch(
            x, types.SimpleNamespace(topk_ids=indices, topk_weights=weights)
        )
        indices.to.assert_called_once_with(self.int32)
        call = self.buffer.dispatch.call_args.kwargs
        self.assertNotIn("num_experts", call)
        self.assertFalse(call["do_cpu_sync"])
        self.assertTrue(call["do_expand"])
        self.assertIs(result[0], recv_x)
        self.assertIs(impl._handle, handle)
        with self.assertRaisesRegex(RuntimeError, "unconsumed"):
            impl.dispatch(x, None)
        combined = object()
        self.buffer.combine.return_value = (combined, None, event)
        self.assertIs(
            impl.combine(
                types.SimpleNamespace(
                    hidden_states=recv_x,
                    topk_weights=recv_weights,
                    routewise_layout=None,
                )
            ),
            combined,
        )
        self.buffer.combine.assert_called_once_with(
            recv_x, handle=handle, topk_weights=recv_weights
        )
        self.assertIsNone(impl._handle)
        with self.assertRaisesRegex(RuntimeError, "without a valid dispatch handle"):
            impl.combine(None)

    def test_scale_connects_only_new_peers_and_commits_world(self):
        state = self.facade._state()
        state.buffer = self.buffer
        state.peer_prefix = "test-peers"
        state.dispatch_ep_size = 4
        self.buffer.rank_id = 0
        self.buffer.get_connected_ranks.return_value = [1, 2, 3]
        self.buffer.get_topology.return_value = ("direct", 8)
        self.facade.on_scale(4, 8)
        self.buffer.connect_ranks.assert_called_once_with(
            [4, 5, 6, 7],
            store=self.ns["get_global_tcp_store"](),
            key_prefix="test-peers",
        )
        self.buffer.set_active_ranks.assert_called_once_with(list(range(8)))
        self.assertEqual(state.dispatch_ep_size, 8)

    def test_elastic_constructor_reserves_maximum_world(self):
        self.ns["get_exec"] = lambda: types.SimpleNamespace(
            moe=types.SimpleNamespace(elastic_ep_backend="mooncake")
        )
        self.buffer.rank_id = 0
        self.buffer.get_connected_ranks.return_value = [1]
        self.buffer.get_topology.return_value = ("direct", 8)
        self.get_buffer()
        kwargs = self.transport.call_args.kwargs
        self.assertEqual(kwargs["max_num_scaleup_ranks"], 8)
        self.assertEqual(kwargs["max_num_scaleout_ranks"], 1)
        self.assertEqual(kwargs["scaleup_rank_idx"], 0)
        self.assertEqual(kwargs["bootstrap_id"], "sglang/nixl_ep_v2/cohort/0/2")
        self.buffer.publish.assert_called_once()
        self.buffer.set_active_ranks.assert_called_once_with([0, 1])

    def test_joiner_uses_global_rank_and_connects_existing_world(self):
        self.ns["get_exec"] = lambda: types.SimpleNamespace(
            moe=types.SimpleNamespace(elastic_ep_backend="mooncake")
        )
        self.ns["ElasticEPStateManager"].get_ep_join_rank_offset = lambda: 4
        self.buffer.rank_id = 4
        self.buffer.get_connected_ranks.return_value = [5]
        self.buffer.get_topology.return_value = ("direct", 8)
        self.get_buffer()
        self.assertEqual(self.transport.call_args.kwargs["scaleup_rank_idx"], 4)
        self.buffer.connect_ranks.assert_called_once_with(
            [0, 1, 2, 3],
            store=self.ns["get_global_tcp_store"](),
            key_prefix="sglang/nixl_ep_v2/peers",
        )
        self.buffer.set_active_ranks.assert_called_once_with(list(range(6)))

    def test_combine_failure_releases_handle(self):
        impl = object.__new__(self.ns["_NixlEPV2Impl"])
        impl._handle = object()
        impl._pad_empty_combine = True
        with self.assertRaisesRegex(ValueError, "route finalization"):
            impl.combine(types.SimpleNamespace(routewise_layout=object()))
        self.assertIsNone(impl._handle)
        self.assertFalse(impl._pad_empty_combine)

    def test_backend_selects_v2_compute_without_selecting_v1(self):
        ns = {"Enum": enum.Enum}
        load_definitions(ROOT / "layers/moe/utils.py", {"MoeA2ABackend"}, ns)
        backend = ns["MoeA2ABackend"]
        v2 = backend("nixl_ep_v2")
        self.assertTrue(v2.is_nixl_ep_v2())
        self.assertTrue(v2.is_elastic_buffer())
        self.assertFalse(v2.is_nixl())
        self.assertFalse(v2.is_deepep_v2())
        self.assertFalse(backend("nixl").is_elastic_buffer())
        self.assertTrue(backend("deepep_v2").is_elastic_buffer())


if __name__ == "__main__":
    unittest.main()
