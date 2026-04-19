"""
Manual test for elastic EP scale-up.

Usage (4 GPUs initial, scale to 8):

  # Terminal 1: start server with room for 8 ranks
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest test/manual/ep/test_elastic_scale.py -v -s

This test launches a 4-GPU server with --max-ep-size 8, then would
require launching 4 new ranks with --ep-join-mode scale and calling
POST /scale_elastic_ep to trigger the scale-up.

NOTE: Full end-to-end scale-up requires multi-process coordination
that is not easily expressed in a single pytest file. This test
validates the server launch with max_ep_size and the HTTP API
endpoint availability. The actual rank joining is tested manually.
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
        """Verify the scale API endpoint is reachable."""
        url = f"{self.base_url}/is_scaling_elastic_ep"
        response = requests.post(url, timeout=10)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("is_scaling_elastic_ep", data)
        self.assertFalse(data["is_scaling_elastic_ep"])

    def test_scale_up_request_validation(self):
        """Verify the scale API validates input."""
        url = f"{self.base_url}/scale_elastic_ep"

        # Missing new_tp_size
        response = requests.post(url, json={}, timeout=10)
        self.assertEqual(response.status_code, 400)

        # Invalid new_tp_size
        response = requests.post(url, json={"new_tp_size": -1}, timeout=10)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
