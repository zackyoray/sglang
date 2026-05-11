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

import json
import os
import shutil
import subprocess
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "--chunked-prefill-size",
    "1024",
    "--disable-cuda-graph",
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


def _preserve_gsm8k_report(model: str) -> None:
    report_dir = os.environ.get(
        "SGLANG_ELASTIC_GSM8K_REPORT_DIR",
        "/lustre/fsw/portfolios/coreai/users/yorayz/logs",
    )
    os.makedirs(report_dir, exist_ok=True)

    src_stem = f"/tmp/gsm8k_{model.replace('/', '_')}"
    dst_stem = os.path.join(report_dir, f"elastic_scale_gsm8k_{int(time.time())}")
    for ext in ("html", "json"):
        src = f"{src_stem}.{ext}"
        dst = f"{dst_stem}.{ext}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[TEST][step 6] preserved GSM8K {ext} report: {dst}", flush=True)
        else:
            print(f"[TEST][step 6] GSM8K {ext} report missing: {src}", flush=True)


def _worker_probe_prompt(index: int) -> str:
    return f"""Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2/2=<<2/2=1>>1 bolt of white fiber.
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric.
#### 3

Question: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?
Answer: He sprints 3*3=<<3*3=9>>9 times.
So he runs 9*60=<<9*60=540>>540 meters.
#### 540

Question: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week? Probe index {index}.
Answer:"""


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
        "--disable-cuda-graph",
        "--chunked-prefill-size",
        "1024",
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
        print(f"[TEST] /generate {msg_suffix} start", flush=True)
        t0 = time.perf_counter()
        try:
            resp = self._post(
                "/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
                },
            )
        except Exception as exc:
            print(
                f"[TEST] /generate {msg_suffix} raised after "
                f"{time.perf_counter() - t0:.2f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        print(
            f"[TEST] /generate {msg_suffix} done status={resp.status_code} "
            f"latency={time.perf_counter() - t0:.2f}s body={resp.text[:200]}",
            flush=True,
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"/generate {msg_suffix} failed: {resp.text}",
        )

    def _debug_completion_probe(self) -> None:
        """Issue a small explicit /v1/completions batch before GSM8K.

        This is a client-side diagnostic for the current post-scale stall:
        if these requests all start and finish, the SGLang server path is
        responsive and any later stall is likely inside the GSM8K eval harness.
        If this hangs, the server-side logs around the same request ids tell us
        which scheduler/NIXL phase stopped.
        """
        if os.environ.get("SGLANG_ELASTIC_DEBUG_COMPLETION_PROBE", "0") != "1":
            return

        num_requests = int(os.environ.get("SGLANG_ELASTIC_DEBUG_COMPLETION_REQUESTS", "8"))
        max_workers = int(os.environ.get("SGLANG_ELASTIC_DEBUG_COMPLETION_THREADS", "4"))
        url = f"{self.base_url}/v1/completions"
        print(
            f"[TEST][completion-probe] start url={url} "
            f"num_requests={num_requests} max_workers={max_workers}",
            flush=True,
        )

        def _one(i: int):
            prompt = (
                f"Question: What is {i} + {i}?\n"
                "Answer:"
            )
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 32,
                "temperature": 0.0,
            }
            t0 = time.perf_counter()
            print(f"[TEST][completion-probe] request {i} start", flush=True)
            resp = requests.post(url, json=payload, timeout=120)
            dt = time.perf_counter() - t0
            text = resp.text[:160].replace("\n", "\\n")
            print(
                f"[TEST][completion-probe] request {i} done "
                f"status={resp.status_code} latency={dt:.2f}s body={text}",
                flush=True,
            )
            resp.raise_for_status()
            return i, dt

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one, i) for i in range(num_requests)]
            for fut in as_completed(futures, timeout=300):
                i, dt = fut.result()
                print(
                    f"[TEST][completion-probe] request {i} observed_done "
                    f"latency={dt:.2f}s",
                    flush=True,
                )
        print(
            f"[TEST][completion-probe] all_done total_latency={time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    def _worker_correlation_probe(self) -> None:
        """Send deterministic sequential completions and preserve raw outputs.

        The DP controller logs rid→target worker when
        SGLANG_ELASTIC_WORKER_TRACE=1. This JSON preserves probe index→rid→text,
        so the logs can be joined with output quality after the run.
        """
        if os.environ.get("SGLANG_ELASTIC_WORKER_CORRELATION_PROBE", "0") != "1":
            return

        num_requests = int(os.environ.get("SGLANG_ELASTIC_WORKER_PROBE_REQUESTS", "8"))
        max_tokens = int(os.environ.get("SGLANG_ELASTIC_WORKER_PROBE_MAX_TOKENS", "128"))
        report_dir = os.environ.get(
            "SGLANG_ELASTIC_GSM8K_REPORT_DIR",
            "/lustre/fsw/portfolios/coreai/users/yorayz/logs",
        )
        os.makedirs(report_dir, exist_ok=True)
        out_path = os.path.join(
            report_dir, f"elastic_scale_worker_probe_{int(time.time())}.json"
        )
        url = f"{self.base_url}/v1/completions"
        print(
            f"[TEST][worker-probe] start url={url} "
            f"num_requests={num_requests} max_tokens={max_tokens}",
            flush=True,
        )

        results = []
        for i in range(num_requests):
            payload = {
                "model": self.model,
                "prompt": _worker_probe_prompt(i),
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stop": ["Question", "Assistant:", "<|separator|>"],
            }
            t0 = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=180)
            latency = time.perf_counter() - t0
            body = resp.json() if resp.ok else {"error": resp.text[:1000]}
            text = ""
            if resp.ok:
                text = body.get("choices", [{}])[0].get("text") or ""
            rid = body.get("id")
            result = {
                "probe_index": i,
                "rid": rid,
                "status_code": resp.status_code,
                "latency": latency,
                "text": text,
                "looks_correct": "#### 460" in text or text.rstrip().endswith("460"),
            }
            results.append(result)
            print(
                f"[TEST][worker-probe] request {i} done rid={rid} "
                f"status={resp.status_code} latency={latency:.2f}s "
                f"looks_correct={result['looks_correct']} "
                f"text={text[:120].replace(chr(10), ' ')}",
                flush=True,
            )
            resp.raise_for_status()

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[TEST][worker-probe] wrote {out_path}", flush=True)

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

        Scale-BEFORE-launch order follows the Mooncake confirmed protocol:
          extend_group_size_to → joiner init → joiner join_group →
          primary get_peer_state → primary recover_ranks.
        Mooncake PR #1968 makes get_peer_state return False for slots
        that haven't joined yet, so polling before the joiner is up
        is safe (no crash, just returns False).
        """
        # Step 1: sanity-check that the 4-rank primary serves traffic.
        print("[TEST][step 1] pre-scale /generate sanity start", flush=True)
        self._generate_ok("pre-scale (4 ranks)")
        print("[TEST][step 1] pre-scale /generate sanity done", flush=True)

        # Step 2: trigger the scale FIRST.  extend_group_size_to(8) on
        # the primary grows the PG so new slots exist.  get_peer_state
        # on empty slots returns False (Mooncake PR #1968).
        print("[TEST][step 2] POST /scale_elastic_ep start", flush=True)
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
        print(
            f"[TEST][step 2] POST /scale_elastic_ep done body={body}",
            flush=True,
        )

        # Step 3: launch the joining group.  It will init_process_group
        # (extension mode), load model, skip CUDA graphs, then call
        # join_group() which blocks until recover_ranks().
        print("[TEST][step 3] launch joining group start", flush=True)
        self._launch_joining_group()
        print("[TEST][step 3] launch joining group done", flush=True)

        # Step 4: wait for the join to complete.  The primary's poll
        # loop (maybe_join_ep_ranks) runs at the end of every forward
        # pass, so we must keep sending requests to drive forward
        # passes -- without traffic the poll never fires.
        print("[TEST][step 4] wait for scaling complete start", flush=True)
        deadline = time.time() + 300
        poll_count = 0
        while time.time() < deadline:
            poll_count += 1
            resp = self._post("/is_scaling_elastic_ep")
            print(
                f"[TEST][step 4] poll {poll_count} "
                f"status={resp.status_code} body={resp.text[:200]}",
                flush=True,
            )
            if resp.ok and not resp.json().get("is_scaling_elastic_ep", True):
                print(
                    f"[TEST][step 4] scaling complete after {poll_count} polls",
                    flush=True,
                )
                print("[TEST] Scaling complete!", flush=True)
                break
            # Drive a forward pass so the poll loop runs on all ranks.
            try:
                print(
                    f"[TEST][step 4] poll {poll_count} drive /generate start",
                    flush=True,
                )
                self._post(
                    "/generate",
                    json={
                        "text": "ping",
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                    },
                )
                print(
                    f"[TEST][step 4] poll {poll_count} drive /generate done",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[TEST][step 4] poll {poll_count} drive /generate "
                    f"raised {type(exc).__name__}: {exc}",
                    flush=True,
                )
            time.sleep(2)
        else:
            self.fail("Timed out waiting for scaling to complete (300s)")

        # Step 5: post-scale inference works.
        print("[TEST][step 5] post-scale /generate sanity start", flush=True)
        self._generate_ok("post-scale (8 ranks)")
        print("[TEST][step 5] post-scale /generate sanity done", flush=True)

        # Optional client-side diagnostic before entering the GSM8K harness.
        print("[TEST][step 5.5] optional completion probe start", flush=True)
        self._debug_completion_probe()
        print("[TEST][step 5.5] optional completion probe done", flush=True)
        if os.environ.get("SGLANG_ELASTIC_STOP_AFTER_COMPLETION_PROBE", "0") == "1":
            print(
                "[TEST][step 5.5] stop-after-completion-probe requested; "
                "marking post-scale single-completion milestone complete",
                flush=True,
            )
            return

        # Step 5.6: deterministic worker/output correlation probe.
        print("[TEST][step 5.6] optional worker correlation probe start", flush=True)
        self._worker_correlation_probe()
        print("[TEST][step 5.6] optional worker correlation probe done", flush=True)

        # Step 6: accuracy check on primary post-scale.
        print("[TEST][step 6] post-scale GSM8K run_eval start", flush=True)
        gsm8k_num_examples = int(os.environ.get("SGLANG_ELASTIC_GSM8K_EXAMPLES", "50"))
        gsm8k_num_threads = int(os.environ.get("SGLANG_ELASTIC_GSM8K_THREADS", "32"))
        gsm8k_max_tokens = int(os.environ.get("SGLANG_ELASTIC_GSM8K_MAX_TOKENS", "512"))
        gsm8k_min_score = float(os.environ.get("SGLANG_ELASTIC_GSM8K_MIN_SCORE", "0.50"))
        print(
            "[TEST][step 6] GSM8K config "
            f"num_examples={gsm8k_num_examples} "
            f"num_threads={gsm8k_num_threads} "
            f"max_tokens={gsm8k_max_tokens} "
            f"min_score={gsm8k_min_score:.2f}",
            flush=True,
        )
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=gsm8k_max_tokens,
            num_examples=gsm8k_num_examples,
            num_threads=gsm8k_num_threads,
        )
        metrics = run_eval(args)
        _preserve_gsm8k_report(self.model)
        print("[TEST][step 6] post-scale GSM8K run_eval done", flush=True)
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
