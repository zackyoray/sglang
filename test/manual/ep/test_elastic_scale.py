"""
Manual smoke test for elastic EP scale-up control plane.

Usage (4 GPUs initial, scale headroom to 8):

  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
      test/manual/ep/test_elastic_scale.py -v -s

This test launches a 4-GPU server with --max-ep-size 8 and verifies:
  * the /is_scaling_elastic_ep and /scale_elastic_ep endpoints are
    mounted and reachable;
  * HTTP-layer input validation rejects malformed bodies;
  * scheduler-layer validation rejects scale-down and over-max requests.

What we do NOT cover here (would require launching 4 new processes with
--ep-join-mode scale on additional GPUs, which Mooncake PG doesn't tolerate
if you call extend_group_size_to without the new ranks being up -- the
transfer engine aborts with a slice_count assertion on phantom transfers):
  * the actual /scale_elastic_ep POST succeeding end-to-end;
  * new ranks joining (active_ranks[4..7] flipping to 1);
  * EPLB rebalance to the new ranks;
  * NIXL connect_ranks to the new ranks;
  * post-scale inference correctness on the larger group.

See TestElasticScaleInProgress below for the manual end-to-end procedure.
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


@unittest.skip(
    "Committing extend_group_size_to without new ranks actually being up "
    "puts Mooncake PG into an unstable state where getTransferStatus hits "
    "an internal assertion on 'empty' transfer tasks to the phantom ranks. "
    "The full scale-up flow requires the new ranks to be launched with "
    "--ep-join-mode scale -- exercised by run_elastic_scale_up.sh, not "
    "from a single-process pytest (mirrors PR #15771's Accuracy Tests style)."
)
class TestElasticScaleInProgress(CustomTestCase):
    """Full scale-up flow. Requires new ranks to be launched externally.

    Run the standalone script instead of pytest:

        test/manual/ep/run_elastic_scale_up.sh

    The script:
      1. launches primary 4-rank cluster on GPUs 0..3 (--node-rank 0),
      2. launches joining 4-rank group on GPUs 4..7 (--node-rank 1
         --ep-join-mode scale) that waits in the poll loop,
      3. POSTs /scale_elastic_ep {"new_ep_size": 8} to the primary,
      4. verifies /is_scaling_elastic_ep flips True -> False,
      5. confirms "joined ranks [...] done" appears in the primary log,
      6. runs a post-scale /generate sanity check.

    This mirrors PR #15771's recovery Accuracy Tests procedure: the
    multi-process coordination isn't a fit for a single-process pytest,
    so we document it as a shell-driven procedure.
    """

    pass


if __name__ == "__main__":
    unittest.main()
