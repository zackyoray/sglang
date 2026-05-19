"""Manual tests for elastic EP scale-up.

Test classes:
  TestElasticScaleServerLaunch          control-plane HTTP validation (4 GPUs)
  TestElasticScaleColdStartThenScale    cold-start gsm8k baseline (8 GPUs)
  TestElasticScaleUpEndToEndNodes2 / Nodes1
                                        full primary + joiner scale-up (8 GPUs)

Run (8-GPU full scale-up):

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \\
        test/manual/ep/test_elastic_scale.py::TestElasticScaleUpEndToEndNodes1 \\
        -v -s
"""

import os
import shutil
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


def _extra_server_args() -> list[str]:
    """Extra `--flag [value]` tokens appended to every spawned server.

    Set via ``SGLANG_ELASTIC_EXTRA_SERVER_ARGS`` as a single space-separated
    string, e.g. ``--disable-overlap-schedule``.
    """
    raw = os.environ.get("SGLANG_ELASTIC_EXTRA_SERVER_ARGS", "").strip()
    return raw.split() if raw else []


SERVER_ARGS = [
    "--trust-remote-code",
    "--moe-a2a-backend", "nixl",
    "--deepep-mode", "low_latency",
    "--tp", "4",
    "--dp", "4",
    "--enable-dp-attention",
    "--elastic-ep-backend", "mooncake",
    "--mooncake-ib-device", ib_devices,
    "--enable-eplb",
    "--ep-num-redundant-experts", "24",
    "--max-ep-size", "8",
    "--mem-fraction-static", "0.5",
] + _extra_server_args()


class TestElasticScaleServerLaunch(CustomTestCase):
    """4-GPU server: validate the scale API endpoints exist and reject bad input."""

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
        url = f"{self.base_url}/is_scaling_elastic_ep"
        response = requests.post(url, timeout=10)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["is_scaling_elastic_ep"])

    def test_scale_up_request_validation(self):
        url = f"{self.base_url}/scale_elastic_ep"
        for body in ({}, {"new_ep_size": -1}, {"new_ep_size": "8"}):
            response = requests.post(url, json=body, timeout=10)
            self.assertEqual(response.status_code, 400, body)

    def test_scale_down_rejected(self):
        # Server launched with tp=4; new_ep_size <= current must fail.
        response = requests.post(
            f"{self.base_url}/scale_elastic_ep",
            json={"new_ep_size": 4},
            timeout=30,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("scale-down", response.json().get("error", ""))

    def test_scale_above_max_rejected(self):
        # Server launched with --max-ep-size 8.
        response = requests.post(
            f"{self.base_url}/scale_elastic_ep",
            json={"new_ep_size": 16},
            timeout=30,
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("max-ep-size", response.json().get("error", ""))


def _count_visible_gpus() -> int:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return len([x for x in env.split(",") if x.strip()])
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


COLD_START_8RANK_ARGS = [
    "--trust-remote-code",
    "--moe-a2a-backend", "nixl",
    "--deepep-mode", "low_latency",
    "--tp", "8",
    "--dp", "8",
    "--enable-dp-attention",
    "--elastic-ep-backend", "mooncake",
    "--mooncake-ib-device", ib_devices,
    "--enable-eplb",
    "--ep-num-redundant-experts", "24",
    "--max-ep-size", "8",
    "--mem-fraction-static", "0.5",
    "--chunked-prefill-size", "1024",
    "--disable-cuda-graph",
] + _extra_server_args()


@unittest.skipUnless(
    _count_visible_gpus() >= 8,
    "Cold-start 8-rank smoke test needs 8 GPUs.",
)
class TestElasticScaleColdStartThenScale(CustomTestCase):
    """8-GPU cold-start gsm8k with --max-ep-size 8 (no scale event)."""

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

    def test_gsm8k(self):
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=200,
                num_threads=128,
            )
        )
        self.assertGreater(metrics["score"], 0.60)


TP_PER_GROUP = 4
TOTAL_EP_SIZE = TP_PER_GROUP * 2  # 8

DIST_INIT_ADDR = os.environ.get(
    "SGLANG_ELASTIC_SCALE_DIST_INIT", "127.0.0.1:24555"
)
PORT_A = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_A", "21000"))
PORT_B = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_B", "21001"))
BASE_URL_A = f"http://127.0.0.1:{PORT_A}"


def _preserve_gsm8k_report(model: str) -> None:
    report_dir = os.environ.get(
        "SGLANG_ELASTIC_GSM8K_REPORT_DIR",
        "/lustre/fsw/portfolios/coreai/users/yorayz/logs",
    )
    os.makedirs(report_dir, exist_ok=True)
    src_stem = f"/tmp/gsm8k_{model.replace('/', '_')}"
    dst_stem = os.path.join(report_dir, f"elastic_scale_gsm8k_{int(time.time())}")
    for ext in ("html", "json"):
        src, dst = f"{src_stem}.{ext}", f"{dst_stem}.{ext}"
        if os.path.exists(src):
            shutil.copy2(src, dst)


def _scale_up_common_args(
    dist_init_addr: str,
    tp_size: int,
    nnodes: int,
    node_rank: int,
) -> list[str]:
    return [
        "--trust-remote-code",
        "--moe-a2a-backend", "nixl",
        "--deepep-mode", "low_latency",
        "--tp", str(tp_size),
        "--dp", str(tp_size),
        "--enable-dp-attention",
        "--elastic-ep-backend", "mooncake",
        "--mooncake-ib-device", ib_devices,
        "--enable-eplb",
        "--ep-num-redundant-experts", "24",
        "--max-ep-size", str(TOTAL_EP_SIZE),
        "--mem-fraction-static", "0.5",
        "--disable-cuda-graph",
        "--chunked-prefill-size", "1024",
        "--nnodes", str(nnodes),
        "--node-rank", str(node_rank),
        "--dist-init-addr", dist_init_addr,
    ] + _extra_server_args()


class _ElasticScaleUpEndToEndBase(CustomTestCase):
    """Shared scale-up E2E plumbing. Subclasses set JOIN_TP/JOIN_NNODES/JOIN_NODE_RANK."""

    JOIN_TP: int
    JOIN_NNODES: int
    JOIN_NODE_RANK: int
    JOIN_RANK_OFFSET: int = 0

    def setUp(self):
        if (
            not hasattr(type(self), "JOIN_TP")
            or type(self) is _ElasticScaleUpEndToEndBase
        ):
            self.skipTest("Abstract base — run a concrete subclass instead")

    @classmethod
    def setUpClass(cls):
        if cls is _ElasticScaleUpEndToEndBase:
            raise unittest.SkipTest("Abstract base")
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
            "sglang", "serve",
            "--model-path", cls.model,
            *_scale_up_common_args(
                DIST_INIT_ADDR,
                tp_size=cls.JOIN_TP,
                nnodes=cls.JOIN_NNODES,
                node_rank=cls.JOIN_NODE_RANK,
            ),
            "--ep-join-mode", "scale",
            "--ep-join-rank-offset", str(cls.JOIN_RANK_OFFSET),
            "--host", "127.0.0.1",
            "--port", str(PORT_B),
            "--device", "cuda",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(TP_PER_GROUP, TOTAL_EP_SIZE)
        )
        joining_log = os.environ.get(
            "SGLANG_ELASTIC_SCALE_JOINING_LOG",
            f"/tmp/elastic_scale_joining_nnodes{cls.JOIN_NNODES}_{int(time.time())}.log",
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
        """Scale primary at ep_size=TP_PER_GROUP up to TOTAL_EP_SIZE."""
        # Step 1: pre-scale sanity.
        self._generate_ok("pre-scale")

        # Step 2: trigger the scale.
        resp = self._post(
            "/scale_elastic_ep", json={"new_ep_size": TOTAL_EP_SIZE}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["old_ep_size"], TP_PER_GROUP)
        self.assertEqual(body["new_ep_size"], TOTAL_EP_SIZE)

        # Step 3: launch the joining group; it blocks in init_process_group
        # until the primary's recover_ranks observes its join.
        self._launch_joining_group()

        # Step 4: poll is_scaling and drive forward passes so the primary's
        # poll loop runs (maybe_join_ep_ranks fires at end-of-forward).
        deadline = time.time() + 300
        while time.time() < deadline:
            resp = self._post("/is_scaling_elastic_ep")
            if resp.ok and not resp.json().get("is_scaling_elastic_ep", True):
                break
            try:
                self._post(
                    "/generate",
                    json={
                        "text": "ping",
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                    },
                )
            except Exception:
                pass
            time.sleep(2)
        else:
            self.fail("Timed out waiting for scaling to complete (300s)")

        # Step 5: post-scale sanity.
        self._generate_ok("post-scale")

        # Step 6: post-scale gsm8k.
        gsm8k_num_examples = int(os.environ.get("SGLANG_ELASTIC_GSM8K_EXAMPLES", "50"))
        gsm8k_num_threads = int(os.environ.get("SGLANG_ELASTIC_GSM8K_THREADS", "32"))
        gsm8k_max_tokens = int(os.environ.get("SGLANG_ELASTIC_GSM8K_MAX_TOKENS", "512"))
        gsm8k_min_score = float(os.environ.get("SGLANG_ELASTIC_GSM8K_MIN_SCORE", "0.50"))
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                max_tokens=gsm8k_max_tokens,
                num_examples=gsm8k_num_examples,
                num_threads=gsm8k_num_threads,
            )
        )
        _preserve_gsm8k_report(self.model)
        print(f"[TEST] Post-scale GSM8K accuracy: {metrics['score']:.2%}")
        self.assertGreater(
            metrics["score"], gsm8k_min_score,
            f"Post-scale GSM8K accuracy too low: {metrics['score']:.2%}"
        )


@unittest.skipUnless(
    _count_visible_gpus() >= TOTAL_EP_SIZE,
    f"Full scale-up E2E needs {TOTAL_EP_SIZE} GPUs.",
)
class TestElasticScaleUpEndToEndNodes2(_ElasticScaleUpEndToEndBase):
    """Joiner as --nnodes 2 --tp TOTAL_EP_SIZE --node-rank 1."""

    JOIN_TP = TOTAL_EP_SIZE
    JOIN_NNODES = 2
    JOIN_NODE_RANK = 1
    JOIN_RANK_OFFSET = 0


@unittest.skipUnless(
    _count_visible_gpus() >= TOTAL_EP_SIZE,
    f"Full scale-up E2E needs {TOTAL_EP_SIZE} GPUs.",
)
class TestElasticScaleUpEndToEndNodes1(_ElasticScaleUpEndToEndBase):
    """Joiner as --nnodes 1 --tp TP_PER_GROUP --node-rank 0 with --ep-join-rank-offset."""

    JOIN_TP = TP_PER_GROUP
    JOIN_NNODES = 1
    JOIN_NODE_RANK = 0
    JOIN_RANK_OFFSET = TP_PER_GROUP


if __name__ == "__main__":
    unittest.main()
