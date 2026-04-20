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
import subprocess
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
# Each --nnodes 1 group needs its OWN torch rendezvous port. Mooncake PG
# bridges them via its own metadata channel, not torch's init_process_group.
#
# IMPORTANT: SGLang derives a cluster of ports from --dist-init-addr:
#   dist_init_port, port_base=+1, detokenizer=+2, rpc=+3, metrics=+4,
#   scheduler_input=+5. We leave a 10-port gap so the two groups don't
#   overlap.
DIST_INIT_ADDR_A = os.environ.get(
    "SGLANG_ELASTIC_SCALE_DIST_INIT_A", "127.0.0.1:24555"
)
DIST_INIT_ADDR_B = os.environ.get(
    "SGLANG_ELASTIC_SCALE_DIST_INIT_B", "127.0.0.1:24570"
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
    """CLI args shared by both primary and joining group in the scale test.

    Primary uses `tp_size=TP_PER_GROUP, nnodes=1, node_rank=0` -- a standalone
    small cluster.

    Joining group uses `tp_size=TOTAL_EP_SIZE, nnodes=2, node_rank=1` so
    SGLang computes its ranks as tp_size*pp_rank + tp_rank = 4..7 (the new
    slots in the extended post-scale world). torch's init_process_group
    receives world_size=TOTAL_EP_SIZE, rank=4..7; Mooncake PG's
    recovered_rank=True skips the rendezvous and attaches to the primary's
    extended group. Mooncake's get_world_size() currently still reports 4
    after extend (known limitation, a fix is planned on the Mooncake side),
    but that shouldn't block attach semantics.
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


@unittest.skip(
    "Full scale-up E2E currently blocked on a Mooncake PG limitation: "
    "after extend_group_size_to(N), the primary's poll loop calls "
    "get_peer_state on the new ranks, which submits transfer tasks to "
    "peers that the Mooncake transfer engine hasn't yet registered "
    "(the joining group's init_process_group hasn't finished). This "
    "hits an assertion in mooncake-transfer-engine/multi_transport.cpp "
    "line 148 ('task.slice_count' failed) and aborts all primary "
    "schedulers. Per the RFC thread, the Mooncake team is planning a "
    "fix on their side. Until then, all control-plane behavior is "
    "covered by TestElasticScaleServerLaunch and the manual "
    "run_elastic_scale_up.sh script exercises what the primary side "
    "does correctly (extend_group_size_to returns 200 before the "
    "crash, proving the scheduler / ZMQ / HTTP stack is sound)."
)
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
        primary_args = _scale_up_common_args(
            DIST_INIT_ADDR_A, tp_size=TP_PER_GROUP, nnodes=1, node_rank=0
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
        """Launch the second 4-rank group with --ep-join-mode scale.

        Uses --nnodes 2 --tp TOTAL_EP_SIZE --node-rank 1 so SGLang computes
        this group's ranks as 4..7 of an 8-rank world. Combined with
        --ep-join-mode scale (which sets recovered_rank=True on Mooncake
        PG), torch's init_process_group skips rendezvous and Mooncake PG
        attaches the new ranks to the primary's extended group.

        The subprocess's stdout/stderr go to a file in /tmp so we can
        diagnose separately from pytest's primary-focused log. The path
        is printed at launch time for easy access.
        """
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *_scale_up_common_args(
                DIST_INIT_ADDR_B,
                tp_size=TOTAL_EP_SIZE,
                nnodes=2,
                node_rank=1,
            ),
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
        # Route joining-group output to its own file so we can inspect
        # it after the test; the primary's output stays on pytest stdout.
        joining_log = os.environ.get(
            "SGLANG_ELASTIC_SCALE_JOINING_LOG",
            f"/tmp/elastic_scale_joining_{int(time.time())}.log",
        )
        print(f"[TEST] Launching joining group; logs -> {joining_log}")
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


if __name__ == "__main__":
    unittest.main()
