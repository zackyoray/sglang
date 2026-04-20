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


TP_PER_NODE = 4
TOTAL_EP_SIZE = TP_PER_NODE * 2  # 8
DIST_INIT_ADDR = os.environ.get("SGLANG_ELASTIC_SCALE_DIST_INIT", "127.0.0.1:24555")
PORT_A = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_A", "21000"))
PORT_B = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_B", "21001"))
BASE_URL_A = f"http://127.0.0.1:{PORT_A}"
BASE_URL_B = f"http://127.0.0.1:{PORT_B}"


def _scale_up_common_args() -> list[str]:
    """CLI args shared by both node-rank 0 and node-rank 1 in the scale test."""
    return [
        "--trust-remote-code",
        "--moe-a2a-backend",
        "nixl",
        "--deepep-mode",
        "low_latency",
        "--tp",
        str(TP_PER_NODE),
        "--dp",
        str(TP_PER_NODE),
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
        "2",
        "--dist-init-addr",
        DIST_INIT_ADDR,
    ]


@unittest.skipUnless(
    _count_visible_gpus() >= TOTAL_EP_SIZE,
    f"Full scale-up E2E needs {TOTAL_EP_SIZE} GPUs "
    f"(primary {TP_PER_NODE} + joining {TP_PER_NODE}).",
)
class TestElasticScaleUpEndToEnd(CustomTestCase):
    """End-to-end scale-up with real joining ranks.

    Launches primary (node-rank 0, GPUs 0..TP_PER_NODE-1) and joining group
    (node-rank 1, GPUs TP_PER_NODE..TOTAL_EP_SIZE-1 with --ep-join-mode scale)
    in PARALLEL. Parallel launch is required because torch's
    init_process_group rendezvous expects all --nnodes to participate before
    either process returns; waiting for primary's health before launching the
    joining group deadlocks.

    The joining group stays HTTP-unhealthy until /scale_elastic_ep is POSTed
    to the primary and the poll loop completes its join. We therefore wait
    only for the primary's /health_generate; the joining group is launched
    as a bare subprocess without a health check.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A

        primary_args = _scale_up_common_args() + [
            "--node-rank",
            "0",
        ]
        joining_args = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *_scale_up_common_args(),
            "--node-rank",
            "1",
            "--ep-join-mode",
            "scale",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT_B),
            "--device",
            "cuda",
        ]

        # Start the joining group first in a background thread so its own
        # blocking init (model load, cuda graph capture, etc.) runs in
        # parallel with the primary's launch. It intentionally does NOT
        # wait for HTTP health -- that only happens after the scale.
        cls._joining_env = os.environ.copy()
        cls._joining_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(TP_PER_NODE, TOTAL_EP_SIZE)
        )
        cls._joining_proc = _launch_server_process(
            joining_args, cls._joining_env, None, cls.model
        )

        # Launch the primary with popen_launch_server, which handles the
        # /health_generate wait. By the time the primary finishes torch
        # init_process_group the joining group will have reached the same
        # rendezvous, so both unblock together.
        primary_env = os.environ.copy()
        primary_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(TP_PER_NODE)
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=primary_env,
        )

    @classmethod
    def tearDownClass(cls):
        for proc in (getattr(cls, "process", None), getattr(cls, "_joining_proc", None)):
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

    def _is_scaling(self) -> bool:
        return self._post("/is_scaling_elastic_ep").json()["is_scaling_elastic_ep"]

    def test_scale_up_end_to_end(self):
        """Full flow: baseline -> POST scale -> wait for join -> verify inference."""
        # 1. Baseline: primary alone, joining group waiting in poll loop.
        self.assertFalse(
            self._is_scaling(), "is_scaling should be False before any scale request"
        )

        # 2. Trigger the scale.
        resp = self._post(
            "/scale_elastic_ep", json={"new_ep_size": TOTAL_EP_SIZE}
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"scale request failed: {resp.text}",
        )
        body = resp.json()
        self.assertEqual(body["old_ep_size"], TP_PER_NODE)
        self.assertEqual(body["new_ep_size"], TOTAL_EP_SIZE)

        # 3. Wait for the join to complete (poll loop flips is_scaling back).
        join_deadline = time.perf_counter() + 120
        while time.perf_counter() < join_deadline:
            if not self._is_scaling():
                break
            time.sleep(1)
        self.assertFalse(
            self._is_scaling(),
            "is_scaling never flipped back to False within 120s; "
            "the joining ranks may have failed to complete join_group",
        )

        # 4. Post-scale inference sanity check.
        gen = self._post(
            "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
            },
        )
        self.assertEqual(
            gen.status_code,
            200,
            f"post-scale inference failed: {gen.text}",
        )


if __name__ == "__main__":
    unittest.main()
