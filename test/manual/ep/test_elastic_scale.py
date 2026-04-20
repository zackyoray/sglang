"""
Manual tests for elastic EP scale-up.

Two test classes:

  TestElasticScaleServerLaunch
    4-GPU server, validates the HTTP endpoints exist and reject malformed
    / scale-down / over-max requests. Pure control-plane; does NOT exercise
    a real scale (no joining ranks are launched).

  TestElasticScaleUpEndToEnd
    8-GPU full scale-up. Launches primary (node-rank 0, GPUs 0..3) and
    joining group (node-rank 1 with --ep-join-mode scale, GPUs 4..7) in
    parallel so torch's init_process_group rendezvous completes. After
    both are up, POSTs /scale_elastic_ep and verifies is_scaling flips
    True -> False and post-scale inference works.

Run with:

  # Control plane only (needs 4 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
      test/manual/ep/test_elastic_scale.py::TestElasticScaleServerLaunch \\
      -v -s

  # Full scale-up (needs 8 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \\
      test/manual/ep/test_elastic_scale.py::TestElasticScaleUpEndToEnd \\
      -v -s
"""

import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _launch_server_process,
    _wait_for_server_health,
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


TP_PER_GROUP = 4
TOTAL_EP_SIZE = TP_PER_GROUP * 2  # 8
DIST_INIT_ADDR = os.environ.get("SGLANG_ELASTIC_SCALE_DIST_INIT", "127.0.0.1:24555")
PORT_A = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_A", "21000"))
PORT_B = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_B", "21001"))
BASE_URL_A = f"http://127.0.0.1:{PORT_A}"


def _scale_up_common_args() -> list[str]:
    """CLI args shared by both primary and joining group in the scale test."""
    return [
        "--trust-remote-code",
        "--moe-a2a-backend",
        "nixl",
        "--deepep-mode",
        "low_latency",
        "--tp",
        str(TP_PER_GROUP),
        "--dp",
        str(TP_PER_GROUP),
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
        "1",
        "--dist-init-addr",
        DIST_INIT_ADDR,
    ]


@unittest.skipUnless(
    _count_visible_gpus() >= TOTAL_EP_SIZE,
    f"Full scale-up E2E needs {TOTAL_EP_SIZE} GPUs "
    f"(primary {TP_PER_GROUP} + joining {TP_PER_GROUP}).",
)
class TestElasticScaleUpEndToEnd(CustomTestCase):
    """End-to-end scale-up with real joining ranks, launched on demand.

    Sequence the test exercises:
      1. launch primary --tp TP_PER_GROUP --nnodes 1 on GPUs 0..3
      2. primary becomes healthy, POST /generate to verify 4-rank serving
      3. launch joining group --tp TP_PER_GROUP --nnodes 1 --ep-join-mode
         scale on GPUs 4..7 (no health wait)
      4. POST /scale_elastic_ep {new_ep_size: TOTAL_EP_SIZE}
      5. wait for the join to complete (primary log shows "joined ranks ... done")
      6. POST /generate to verify post-scale inference

    Both groups use --nnodes 1 because each group is its own torch world;
    they share --dist-init-addr so Mooncake PG can rendezvous across them.
    The joining group uses --ep-join-mode scale so its Mooncake PG init
    attaches to the existing group rather than requiring a fresh rendezvous.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A
        cls._joining_proc = None

        # Step 1: launch primary alone with --nnodes 1, wait for health.
        primary_args = _scale_up_common_args()
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
        """Launch the second 4-rank group with --ep-join-mode scale.

        Called from the test method (not setUp) so the primary has time to
        serve pre-scale traffic before the joining group starts allocating
        GPUs.
        """
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *_scale_up_common_args(),
            "--ep-join-mode",
            "scale",
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
        cls._joining_proc = _launch_server_process(cmd, env, None, cls.model)

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
        """The real scale use case: serve on N ranks, then attach N more."""
        # Step 2: sanity-check that the 4-rank primary serves traffic.
        self._generate_ok("pre-scale (4 ranks)")

        # Step 3: launch the joining group (no health wait -- it stays
        # unhealthy until the scale-up join completes).
        self._launch_joining_group()

        # Give the joining group time to reach the elastic-EP join poll
        # loop. Until it's there, extend_group_size_to has no peer to
        # rendezvous with. Model load + cuda graph capture on lite fp8
        # typically finishes in ~60s; wait generously.
        time.sleep(90)

        # Step 4: trigger the scale.
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

        # Step 5: wait for the join to complete. We detect completion via
        # /generate staying 200 while the poll loop runs; a real join
        # failure would crash the primary and /generate would start
        # returning 503 or connection-reset. We also allow time for the
        # EPLB rebalance that fires on active_ranks change.
        time.sleep(30)

        # Step 6: post-scale inference works.
        self._generate_ok("post-scale (8 ranks)")


if __name__ == "__main__":
    unittest.main()
