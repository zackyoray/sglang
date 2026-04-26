"""
Manual tests for elastic EP scale-up.

Two test classes:

  TestElasticScaleServerLaunch
    4-GPU server, validates the HTTP endpoints exist and reject malformed
    / scale-down / over-max requests. Pure control-plane; does NOT exercise
    a real scale (no joining ranks are launched).

  TestElasticScaleColdStartThenScale
    8-GPU gsm8k smoke with --max-ep-size 8 (baseline elastic / NIXL plumbing).

  TestElasticScaleUpEndToEndNodes2 / TestElasticScaleUpEndToEndNodes1
    8-GPU full scale-up. Primary runs on GPUs 0..3; the joiner on GPUs 4..7
    differs only in how it is invoked:
      * Nodes2: --nnodes 2 --tp 8 --node-rank 1  (SGLang cross-node mode)
      * Nodes1: --nnodes 1 --tp 4 --node-rank 0  (SGLang single-node mode)
    Both kept so we can A/B compare Mooncake PG joiner-attach semantics.

Run with:

  # Control plane only (needs 4 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
      test/manual/ep/test_elastic_scale.py::TestElasticScaleServerLaunch \\
      -v -s

  # 8-GPU gsm8k baseline:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \\
      test/manual/ep/test_elastic_scale.py::TestElasticScaleColdStartThenScale \\
      -v -s

  # Scale-up variants (need 8 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \\
      test/manual/ep/test_elastic_scale.py::TestElasticScaleUpEndToEndNodes2 \\
      -v -s
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \\
      test/manual/ep/test_elastic_scale.py::TestElasticScaleUpEndToEndNodes1 \\
      -v -s
"""

import os
import subprocess
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

TEST_MODEL = os.environ.get("NIXL_EP_TEST_MODEL", DEFAULT_MODEL_NAME_FOR_TEST_MLA)
os.environ.setdefault("SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

ib_devices = get_rdma_devices_args()

SERVER_ARGS = [
    "--trust-remote-code",
    "--moe-a2a-backend",
    "nixl",
    "--deepep-mode",
    "low_latency",
    "--tp",
    "4",
    "--dp",
    "4",
    "--enable-dp-attention",
    "--elastic-ep-backend",
    "mooncake",
    "--mooncake-ib-device",
    ib_devices,
    "--enable-eplb",
    "--ep-num-redundant-experts",
    "24",
    "--max-ep-size",
    "8",
    "--mem-fraction-static",
    "0.5",
]


class TestElasticScaleServerLaunch(CustomTestCase):
    """Test that the server launches with --max-ep-size and exposes the scale API."""

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=SERVER_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.process.wait(timeout=15)
        time.sleep(2)

    def test_scale_endpoint_exists(self):
        """Verify the scale API endpoint is reachable and reports idle."""
        url = f"{self.base_url}/is_scaling_elastic_ep"
        response = requests.post(url, timeout=10)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("is_scaling_elastic_ep", data)
        # No scale has been triggered yet, so the system must be idle.
        self.assertFalse(data["is_scaling_elastic_ep"])

    def test_scale_up_request_validation(self):
        """Verify the scale API validates input at the HTTP layer."""
        url = f"{self.base_url}/scale_elastic_ep"

        # Missing new_ep_size
        response = requests.post(url, json={}, timeout=10)
        self.assertEqual(response.status_code, 400)

        # Non-positive new_ep_size
        response = requests.post(url, json={"new_ep_size": -1}, timeout=10)
        self.assertEqual(response.status_code, 400)

        # Wrong type
        response = requests.post(url, json={"new_ep_size": "8"}, timeout=10)
        self.assertEqual(response.status_code, 400)

    def test_scale_down_rejected(self):
        """Scheduler must reject new_ep_size <= current effective_ep_size."""
        url = f"{self.base_url}/scale_elastic_ep"
        # Server launched with tp=4 → current effective_ep_size = 4.
        response = requests.post(url, json={"new_ep_size": 4}, timeout=30)
        self.assertEqual(response.status_code, 500)
        self.assertIn("scale-down", response.json().get("error", ""))

    def test_scale_above_max_rejected(self):
        """Scheduler must reject new_ep_size > --max-ep-size."""
        url = f"{self.base_url}/scale_elastic_ep"
        # Server launched with --max-ep-size 8 → 16 must fail.
        response = requests.post(url, json={"new_ep_size": 16}, timeout=30)
        self.assertEqual(response.status_code, 500)
        self.assertIn("max-ep-size", response.json().get("error", ""))


def _count_visible_gpus() -> int:
    """Return the number of CUDA devices visible to this process."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return len([x for x in env.split(",") if x.strip()])
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Cold-start 8-rank smoke test
# ---------------------------------------------------------------------------
# Launches all 8 ranks together at world_size=8 with --max-ep-size 8 (i.e. no
# headroom, max == world). No scaling involved -- this just proves that our
# --max-ep-size plumbing doesn't break the baseline when everything is
# cold-started like TestNixlMoeMooncakeElasticEP in test_nixl_ep.py. If this
# passes but the "real" scale (separate process joining later) doesn't, it
# confirms the crash is in the Mooncake PG extend-then-join path, not in
# our infrastructure.

COLD_START_8RANK_ARGS = [
    "--trust-remote-code",
    "--moe-a2a-backend",
    "nixl",
    "--deepep-mode",
    "low_latency",
    "--tp",
    "8",
    "--dp",
    "8",
    "--enable-dp-attention",
    "--elastic-ep-backend",
    "mooncake",
    "--mooncake-ib-device",
    ib_devices,
    "--enable-eplb",
    "--ep-num-redundant-experts",
    "24",
    "--max-ep-size",
    "8",
    "--mem-fraction-static",
    "0.5",
]


@unittest.skipUnless(
    _count_visible_gpus() >= 8,
    "Cold-start 8-rank smoke test needs 8 GPUs.",
)
class TestElasticScaleColdStartThenScale(CustomTestCase):
    """8-GPU cold-start gsm8k with --max-ep-size 8 (baseline smoke).

    Same as TestNixlMoeMooncakeElasticEP plus --max-ep-size; does not POST
    /scale_elastic_ep.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=COLD_START_8RANK_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.process.wait(timeout=15)
        time.sleep(2)

    def _run_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        return run_eval(args)

    def test_gsm8k(self):
        """gsm8k on the 8-rank cold-started cluster with --max-ep-size 8."""
        metrics = self._run_gsm8k()
        self.assertGreater(metrics["score"], 0.60)


TP_PER_GROUP = 4
TOTAL_EP_SIZE = TP_PER_GROUP * 2  # 8
# Primary and joiner share the SAME dist_init_addr so their Mooncake PG
# metadata lands on the same Store.  ZMQ / DP-handshake port collisions
# are avoided because joiners (--ep-join-mode scale) derive those ports
# from their own --port (PORT_B) instead of from dist_init_addr.
DIST_INIT_ADDR = os.environ.get(
    "SGLANG_ELASTIC_SCALE_DIST_INIT", "127.0.0.1:24555"
)
PORT_A = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_A", "21000"))
PORT_B = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_B", "21001"))
BASE_URL_A = f"http://127.0.0.1:{PORT_A}"


def _scale_up_common_args(
    dist_init_addr: str,
    tp_size: int,
    nnodes: int,
    node_rank: int,
) -> list[str]:
    """Shared CLI for primary + joiner. Varies: tp_size, nnodes, node_rank,
    dist_init_addr. Joiner shape (nnodes=2 vs nnodes=1) is chosen by the
    concrete TestElasticScaleUpEndToEnd* subclass.
    """
    return [
        "--trust-remote-code",
        "--moe-a2a-backend",
        "nixl",
        "--deepep-mode",
        "low_latency",
        "--tp",
        str(tp_size),
        "--dp",
        str(tp_size),
        "--enable-dp-attention",
        "--elastic-ep-backend",
        "mooncake",
        "--mooncake-ib-device",
        ib_devices,
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "24",
        "--max-ep-size",
        str(TOTAL_EP_SIZE),
        "--mem-fraction-static",
        "0.5",
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        dist_init_addr,
    ]


class _ElasticScaleUpEndToEndBase(CustomTestCase):
    """Shared scale-up E2E plumbing — abstract base, not collected directly.

    Subclasses set `JOIN_TP`, `JOIN_NNODES`, `JOIN_NODE_RANK` and pytest
    collects them as TestElasticScaleUpEndToEndNodes{1,2}.

    The __init_subclass__ hook ensures pytest skips this base class
    when it lacks the required JOIN_* attributes.

    Sequence:
      1. launch primary --tp TP_PER_GROUP --nnodes 1 on GPUs 0..3
      2. primary healthy, /generate sanity-check at ep_size=4
      3. launch joiner on GPUs 4..7 (subclass-specific shape)
      4. POST /scale_elastic_ep {new_ep_size: TOTAL_EP_SIZE}
      5. wait for join
      6. /generate post-scale
    """

    JOIN_TP: int
    JOIN_NNODES: int
    JOIN_NODE_RANK: int
    JOIN_RANK_OFFSET: int = 0

    def setUp(self):
        if not hasattr(type(self), "JOIN_TP") or type(self) is _ElasticScaleUpEndToEndBase:
            self.skipTest("Abstract base — run a concrete subclass instead")

    @classmethod
    def setUpClass(cls):
        if cls is _ElasticScaleUpEndToEndBase:
            raise unittest.SkipTest("Abstract base — run a concrete subclass instead")
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A
        cls._joining_proc = None

        primary_args = _scale_up_common_args(
            DIST_INIT_ADDR, tp_size=TP_PER_GROUP, nnodes=1, node_rank=0
        )
        primary_env = os.environ.copy()
        primary_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(TP_PER_GROUP)
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=primary_env,
        )

    @classmethod
    def _launch_joining_group(cls) -> None:
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *_scale_up_common_args(
                DIST_INIT_ADDR,
                tp_size=cls.JOIN_TP,
                nnodes=cls.JOIN_NNODES,
                node_rank=cls.JOIN_NODE_RANK,
            ),
            "--ep-join-mode",
            "scale",
            "--ep-join-rank-offset",
            str(cls.JOIN_RANK_OFFSET),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT_B),
            "--device",
            "cuda",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(TP_PER_GROUP, TOTAL_EP_SIZE)
        )
        joining_log = os.environ.get(
            "SGLANG_ELASTIC_SCALE_JOINING_LOG",
            f"/tmp/elastic_scale_joining_nnodes{cls.JOIN_NNODES}_{int(time.time())}.log",
        )
        print(
            f"[TEST] Launching joiner (nnodes={cls.JOIN_NNODES}, "
            f"tp={cls.JOIN_TP}, node_rank={cls.JOIN_NODE_RANK}); logs -> {joining_log}"
        )
        cls._joining_log_path = joining_log
        cls._joining_log_fh = open(joining_log, "w")
        cls._joining_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=cls._joining_log_fh,
            stderr=subprocess.STDOUT,
        )

    @classmethod
    def tearDownClass(cls):
        for proc in (
            getattr(cls, "process", None),
            getattr(cls, "_joining_proc", None),
        ):
            if proc is None:
                continue
            try:
                kill_process_tree(proc.pid)
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
            except Exception:
                pass
        fh = getattr(cls, "_joining_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
        joining_log = getattr(cls, "_joining_log_path", None)
        if joining_log:
            print(f"[TEST] Joining group log preserved at {joining_log}")
        time.sleep(2)

    def _post(self, path: str, **kwargs) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)

    def _generate_ok(self, msg_suffix: str) -> None:
        resp = self._post(
            "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
            },
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"/generate {msg_suffix} failed: {resp.text}",
        )

    def test_scale_up_on_demand(self):
        """The real scale use case: serve on N ranks, then attach N more.

        Order of operations:

          1. sanity-check primary serves traffic at ep_size=4
          2. launch joining group -- it reaches init_process_group and
             BLOCKS there. Mooncake PG's recovered_rank=True attach
             requests ranks 4..7 of an 8-rank group; the primary's
             Mooncake group is still size 4, so attach waits for extend.
          3. POST /scale_elastic_ep -- primary's extend_group_size_to(8)
             grows the group so the joining-side attach unblocks
             naturally (Mooncake PG rendezvous completes on both sides
             at the same time).
          4. wait for the join to complete: joining group finishes model
             load + cuda graph capture, primary's poll loop detects
             ranks 4..7 live, EPLB rebalances, NIXL connects.
          5. verify post-scale inference works.

        Launch-BEFORE-scale order is required: if we scale BEFORE the
        joining group is up, the primary's poll loop tries to call
        get_peer_state on phantom ranks, hitting Mooncake's
        multi_transport.cpp:148 'task.slice_count' assertion.
        """
        # Step 1: sanity-check that the 4-rank primary serves traffic.
        self._generate_ok("pre-scale (4 ranks)")

        # Step 2: launch the joining group FIRST. It will block in
        # init_process_group until the primary's Mooncake group is
        # extended (done in step 3 via POST /scale_elastic_ep).
        self._launch_joining_group()

        # Give the joining group time to reach init_process_group.
        # Model weights aren't loaded yet (init_process_group happens
        # before load_weight in model_runner.__init__), so this should
        # only take ~10-20 seconds after the subprocess starts.
        time.sleep(30)

        # Step 3: trigger the scale. This runs extend_group_size_to(8)
        # on the primary's Mooncake PG, which unblocks the joining
        # group's init_process_group attach.
        resp = self._post(
            "/scale_elastic_ep", json={"new_ep_size": TOTAL_EP_SIZE}
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"scale request failed: {resp.text}",
        )
        body = resp.json()
        self.assertEqual(body["old_ep_size"], TP_PER_GROUP)
        self.assertEqual(body["new_ep_size"], TOTAL_EP_SIZE)

        # Step 4: wait for the join to complete. After
        # init_process_group unblocks, the joining group still has to
        # finish model load (~20s), cuda graph capture (~30s), and
        # reach the elastic-EP join path. Meanwhile the primary's poll
        # loop tries try_recover_ranks every forward pass.
        time.sleep(240)

        # Step 5: post-scale inference works.
        self._generate_ok("post-scale (8 ranks)")


@unittest.skipUnless(
    _count_visible_gpus() >= TOTAL_EP_SIZE,
    f"Full scale-up E2E needs {TOTAL_EP_SIZE} GPUs.",
)
class TestElasticScaleUpEndToEndNodes2(_ElasticScaleUpEndToEndBase):
    """Joiner as nnodes=2, tp=8, node_rank=1 (current behavior).

    SGLang computes joiner ranks as 4..7 via _calculate_rank_ranges. torch
    init_process_group(world_size=8, rank=4..7) relies on Mooncake
    recovered_rank=True to skip the TCP-store rendezvous. Triggers
    SGLang's cross-node code paths (ZMQ/DP-attention/tensor transport).
    """

    JOIN_TP = TOTAL_EP_SIZE
    JOIN_NNODES = 2
    JOIN_NODE_RANK = 1


@unittest.skipUnless(
    _count_visible_gpus() >= TOTAL_EP_SIZE,
    f"Full scale-up E2E needs {TOTAL_EP_SIZE} GPUs.",
)
class TestElasticScaleUpEndToEndNodes1(_ElasticScaleUpEndToEndBase):
    """Joiner as nnodes=1, tp=4, node_rank=0 with --ep-join-rank-offset 4.

    Mooncake Shape (A): init_process_group(world_size=8, rank=4..7).
    The --ep-join-rank-offset shifts local ranks 0..3 to global 4..7
    and sets world_size=max_ep_size. Stays on single-node code path.
    """

    JOIN_TP = TP_PER_GROUP
    JOIN_NNODES = 1
    JOIN_NODE_RANK = 0
    JOIN_RANK_OFFSET = TP_PER_GROUP


if __name__ == "__main__":
    unittest.main()
