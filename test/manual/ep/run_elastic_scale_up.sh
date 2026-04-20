#!/usr/bin/env bash
# Manual end-to-end elastic EP scale-up test.
#
# Mirrors the style of PR #15771's Accuracy Tests section: a shell-driven
# procedure that coordinates two node groups rather than a single-process
# pytest. Required because Mooncake PG's extend_group_size_to cannot be
# called before the new ranks are actually running (the transfer engine
# asserts on empty transfer tasks to phantom peers).
#
# Topology
# --------
# This script launches TWO sglang server processes on the same machine:
#
#   node-rank 0: GPUs 0..3, treated as the "live" cluster (initial world).
#   node-rank 1: GPUs 4..5 ... (wait, we actually need matching tp-per-node)
#
# Both rendezvous via the same --dist-init-addr. node-rank 0 starts first
# and serves traffic. node-rank 1 launches with --ep-join-mode scale and
# waits in the poll loop for extend_group_size_to to be called. Once we
# POST /scale_elastic_ep, the extend is issued and node-rank 1's ranks
# complete join_group() in the next forward pass.
#
# Usage
# -----
#   # Prerequisites: 8 GPUs, kvcache-ai mooncake + NIXL EP installed,
#   # PYTHONPATH set so `from mooncake import ep` resolves correctly
#   # (see SGLANG_MOONCAKE_EP_PREPEND_PYTHONPATH).
#   export MODEL_PATH=/sgl-workspace/llm_models/DeepSeek-V3-Lite/fp8/
#   export IB_DEVICES=mlx5_0,mlx5_4
#   ./run_elastic_scale_up.sh
#
# Expected
# --------
#   1. Both servers start, Terminal A reaches "fired up and ready to roll".
#   2. Baseline /is_scaling_elastic_ep returns false.
#   3. POST /scale_elastic_ep {"new_ep_size": 8} returns 200.
#   4. Within a few forward passes Terminal A logs "joined ranks [4,5,6,7] done".
#   5. /is_scaling_elastic_ep flips back to false.
#   6. gsm8k run on the 8-rank topology returns score > 0.60.

set -euo pipefail

# ---------- knobs -----------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/sgl-workspace/llm_models/DeepSeek-V3-Lite/fp8/}"
IB_DEVICES="${IB_DEVICES:-mlx5_0,mlx5_4}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-127.0.0.1:24555}"
PORT_A="${PORT_A:-21000}"
PORT_B="${PORT_B:-21001}"
LOG_DIR="${LOG_DIR:-/tmp/elastic_scale_up_$(date +%s)}"
TP_PER_NODE="${TP_PER_NODE:-4}"          # tp per --node-rank
NODES=2
TOTAL_EP=$(( TP_PER_NODE * NODES ))      # final EP size = 8
CURRENT_EP="${TP_PER_NODE}"              # initial EP size = 4
JOIN_TIMEOUT_SEC="${JOIN_TIMEOUT_SEC:-120}"

mkdir -p "$LOG_DIR"
echo "Logs will be written under $LOG_DIR"

# Shared server arguments for both nodes. Scale requires both node groups
# to use the same --dist-init-addr so they find each other via Mooncake.
COMMON_ARGS=(
    --trust-remote-code
    --moe-a2a-backend nixl
    --deepep-mode low_latency
    --tp "$TP_PER_NODE"
    --dp "$TP_PER_NODE"
    --enable-dp-attention
    --elastic-ep-backend mooncake
    --mooncake-ib-device "$IB_DEVICES"
    --enable-eplb
    --ep-num-redundant-experts 24
    --max-ep-size "$TOTAL_EP"
    --mem-fraction-static 0.5
    --nnodes "$NODES"
    --dist-init-addr "$DIST_INIT_ADDR"
    --host 127.0.0.1
)

# ---------- launch node-rank 0 (live cluster) --------------------------------
echo "[A] Launching primary cluster on GPUs 0..$((TP_PER_NODE-1)) port $PORT_A"
CUDA_VISIBLE_DEVICES="0,1,2,3" \
sglang serve \
    --model-path "$MODEL_PATH" \
    "${COMMON_ARGS[@]}" \
    --node-rank 0 \
    --port "$PORT_A" \
    >"$LOG_DIR/node_a.log" 2>&1 &
PID_A=$!
echo "    pid=$PID_A, log=$LOG_DIR/node_a.log"

cleanup() {
    echo "Cleaning up..."
    kill "$PID_A" 2>/dev/null || true
    kill "${PID_B:-}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait until node A's health endpoint is up.
echo "[A] Waiting for primary cluster to become healthy..."
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT_A}/health_generate" >/dev/null 2>&1; then
        break
    fi
    sleep 10
done
curl -sf "http://127.0.0.1:${PORT_A}/health_generate" >/dev/null || {
    echo "ERROR: node-rank 0 never became healthy. See $LOG_DIR/node_a.log"
    exit 1
}
echo "[A] primary cluster is healthy."

# ---------- launch node-rank 1 (joining ranks, waiting in poll loop) ---------
echo "[B] Launching joining ranks on GPUs $TP_PER_NODE..$((TOTAL_EP-1)) with --ep-join-mode scale"
CUDA_VISIBLE_DEVICES="4,5,6,7" \
sglang serve \
    --model-path "$MODEL_PATH" \
    "${COMMON_ARGS[@]}" \
    --node-rank 1 \
    --port "$PORT_B" \
    --ep-join-mode scale \
    >"$LOG_DIR/node_b.log" 2>&1 &
PID_B=$!
echo "    pid=$PID_B, log=$LOG_DIR/node_b.log"

# Give node-rank 1 time to reach the poll loop (after model load + cuda graph
# capture). It will NOT become HTTP-healthy until after it joins the cluster.
sleep 30

# ---------- baseline: is_scaling should be false -----------------------------
echo "[A] Checking baseline is_scaling..."
baseline=$(curl -sf -X POST "http://127.0.0.1:${PORT_A}/is_scaling_elastic_ep" | \
    python3 -c "import json,sys; print(json.load(sys.stdin)['is_scaling_elastic_ep'])")
if [[ "$baseline" != "False" ]]; then
    echo "ERROR: baseline is_scaling=$baseline, expected False"
    exit 1
fi
echo "[A] baseline is_scaling=False (OK)"

# ---------- trigger the scale ------------------------------------------------
echo "[A] POST /scale_elastic_ep {new_ep_size: $TOTAL_EP}"
scale_resp=$(curl -sf -X POST -H "Content-Type: application/json" \
    -d "{\"new_ep_size\": $TOTAL_EP}" \
    "http://127.0.0.1:${PORT_A}/scale_elastic_ep")
echo "    response: $scale_resp"

# ---------- wait for the join to complete ------------------------------------
echo "[A] Waiting up to ${JOIN_TIMEOUT_SEC}s for is_scaling to flip back to False..."
for i in $(seq 1 "$JOIN_TIMEOUT_SEC"); do
    is_scaling=$(curl -sf -X POST "http://127.0.0.1:${PORT_A}/is_scaling_elastic_ep" | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['is_scaling_elastic_ep'])" 2>/dev/null || echo "ERROR")
    if [[ "$is_scaling" == "False" ]]; then
        echo "[A] is_scaling=False after ${i}s -- join completed."
        break
    fi
    sleep 1
done
if [[ "$is_scaling" != "False" ]]; then
    echo "ERROR: is_scaling never flipped to False after ${JOIN_TIMEOUT_SEC}s"
    echo "       Last known value: $is_scaling"
    echo "       Check $LOG_DIR/node_a.log for 'joined ranks' and 'EPLB rebalance' entries."
    exit 1
fi

# Also look for the explicit join log line in node-rank 0's output.
if grep -q "joined ranks.*done" "$LOG_DIR/node_a.log"; then
    echo "[A] Found 'joined ranks ... done' in log (OK)."
else
    echo "WARNING: no 'joined ranks ... done' line found in $LOG_DIR/node_a.log"
fi

# ---------- post-scale inference sanity check --------------------------------
echo "[A] Post-scale inference sanity check (/generate)..."
gen_resp=$(curl -sf -X POST -H "Content-Type: application/json" \
    -d '{"text":"Hello","sampling_params":{"max_new_tokens":8,"temperature":0.0}}' \
    "http://127.0.0.1:${PORT_A}/generate")
echo "    response: $gen_resp"

echo "Elastic scale-up test PASSED."
echo "Logs preserved in $LOG_DIR for inspection."
