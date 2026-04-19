"""
Manual smoke test for elastic EP scale-up.

Usage (4 GPUs initial, scale headroom to 8):

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
      test/manual/ep/test_elastic_scale.py -v -s

This test launches a 4-GPU server with --max-ep-size 8 and verifies:
  * the /is_scaling_elastic_ep and /scale_elastic_ep endpoints are
    mounted and reachable;
  * HTTP-layer input validation rejects malformed bodies;
  * scheduler-layer validation rejects scale-down and over-max requests;
  * a successful scale request flips is_scaling to True;
  * inference still works on the original ranks while pending join;
  * a second concurrent scale request is rejected.

What we do NOT cover here (would require launching 4 new processes with
--ep-join-mode scale on additional GPUs, which is fragile inside a single
pytest process):
  * the new ranks actually joining (active_ranks[4..7] flipping to 1);
  * EPLB rebalance to the new ranks;
  * NIXL connect_ranks to the new ranks;
  * post-scale inference correctness on the larger group.

For the full end-to-end procedure see the docstring on
TestElasticScaleInProgress.test_scale_up_pending_join below.
"""

import os
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
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


class TestElasticScaleInProgress(CustomTestCase):
    """Verify a successful scale request transitions the server to "scaling".

    NOTE: this test does NOT launch the additional 4 ranks needed to actually
    complete the scale. After test_scale_up_pending_join runs, the server is
    permanently stuck in is_scaling=True (poll loop keeps calling
    get_peer_state for ranks 4..7 which never come up). All assertions are
    written so they pass with the new ranks still missing.

    Each test method creates its OWN server because once a scale is committed
    via extend_group_size_to we cannot reset the process group; the state
    only resolves when the new ranks join (out of scope for in-process tests)
    or the server is killed.

    To run the full end-to-end procedure manually:
        # Terminal A (initial 4 ranks):
        CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve <model> \\
            --tp 4 --dp 4 --enable-dp-attention --max-ep-size 8 \\
            --elastic-ep-backend mooncake --moe-a2a-backend nixl ...
        # Terminal B (kick off the scale):
        curl -X POST http://127.0.0.1:30000/scale_elastic_ep \\
             -d '{"new_ep_size": 8}'
        # Terminal C (4 new ranks join the live group):
        CUDA_VISIBLE_DEVICES=4,5,6,7 sglang serve <model> \\
            --tp 4 --dp 4 --enable-dp-attention --max-ep-size 8 \\
            --elastic-ep-backend mooncake --moe-a2a-backend nixl \\
            --ep-join-mode scale --node-rank 1 ...
        # Verify on Terminal A: is_scaling flips back to false, EPLB
        # rebalance log line appears, gsm8k still passes.
    """

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

    def _post(self, path: str, **kwargs) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)

    def _is_scaling(self) -> bool:
        return self._post("/is_scaling_elastic_ep").json()["is_scaling_elastic_ep"]

    def test_scale_up_pending_join(self):
        """Scale request 4 -> 8 succeeds and the server reports scaling=True.

        With no new ranks launched, the system stays in this state. We assert:
          1. baseline /is_scaling_elastic_ep is False;
          2. /scale_elastic_ep {new_ep_size: 8} returns 200 with old=4 new=8;
          3. /is_scaling_elastic_ep flips to True;
          4. inference still works on the original 4 ranks while pending;
          5. a second scale request is rejected by the is_scaling guard.
        """
        # 1. Baseline: not scaling.
        self.assertFalse(self._is_scaling(), "server should be idle at startup")

        # 2. Trigger the scale.
        scale_resp = self._post("/scale_elastic_ep", json={"new_ep_size": 8})
        self.assertEqual(
            scale_resp.status_code,
            200,
            f"scale request failed: {scale_resp.text}",
        )
        body = scale_resp.json()
        self.assertEqual(body["old_ep_size"], 4)
        self.assertEqual(body["new_ep_size"], 8)

        # 3. State must now report scaling-in-progress.
        self.assertTrue(
            self._is_scaling(),
            "is_scaling should flip to True after extend_group_size_to",
        )

        # 4. Inference on the original ranks must still work. We use the
        # /generate endpoint with a tiny request so the gsm8k harness isn't
        # required here (this test focuses on control-plane correctness).
        gen_resp = self._post(
            "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
            },
        )
        self.assertEqual(
            gen_resp.status_code,
            200,
            f"inference should still work while scale is pending: {gen_resp.text}",
        )

        # 5. A second scale must be rejected because the previous one has not
        # finished (no new ranks ever joined).
        second = self._post("/scale_elastic_ep", json={"new_ep_size": 8})
        self.assertEqual(
            second.status_code,
            500,
            "second scale must be rejected while a scale is pending",
        )
        self.assertIn("not completed", second.json().get("error", ""))


if __name__ == "__main__":
    unittest.main()
