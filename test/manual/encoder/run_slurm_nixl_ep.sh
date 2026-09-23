#!/bin/bash
set -euo pipefail

REPO=/lustre/fs1/portfolios/coreai/projects/coreai_tritoninference_triton3/users/yorayz/sglang-nixl-ep
RESULTS="$REPO/slurm-results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS"
cd "$REPO"

export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_tritoninference_triton3/users/yorayz/.hf-cache
export XDG_CACHE_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_tritoninference_triton3/users/yorayz/.cache
export SGLANG_CACHE_DIR="$XDG_CACHE_HOME/sglang"
export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
export TRITON_CACHE_DIR="$SGLANG_CACHE_DIR/triton"
mkdir -p "$SGLANG_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR"

# Match the patched NIXL/UCX overlay from the last known-good SGLang run.
export NIXL_ASYNC_PREFIX=/lustre/fs1/portfolios/coreai/projects/coreai_tritoninference_triton3/users/yorayz/nixl-async-install-b0cbb237
export NIXL_ASYNC_UCX_PREFIX=/lustre/fs1/portfolios/coreai/projects/coreai_tritoninference_triton3/users/yorayz/ucx-v1.22-install
export PYTHONPATH="$REPO/python:$NIXL_ASYNC_PREFIX/lib/python3/dist-packages:/usr/local/lib/python3.12/dist-packages:/sgl-workspace/nixl/install/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export NIXL_PLUGIN_DIR="$NIXL_ASYNC_PREFIX/lib/x86_64-linux-gnu/plugins"
export LD_LIBRARY_PATH="$NIXL_PLUGIN_DIR:$NIXL_ASYNC_PREFIX/lib/x86_64-linux-gnu:$NIXL_ASYNC_UCX_PREFIX/lib:$NIXL_ASYNC_UCX_PREFIX/lib/ucx:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$NIXL_ASYNC_UCX_PREFIX/lib/libucp.so.0:$NIXL_ASYNC_UCX_PREFIX/lib/libuct.so.0:$NIXL_ASYNC_UCX_PREFIX/lib/libucs.so.0:$NIXL_ASYNC_UCX_PREFIX/lib/libucm.so.0:$NIXL_ASYNC_PREFIX/lib/x86_64-linux-gnu/libnixl.so${LD_PRELOAD:+:$LD_PRELOAD}"
export NIXL_UCX_DEEPBIND=true
export UCX_LOG_LEVEL=error
unset UCX_NET_DEVICES

# mlx5_3 is the management NIC on this cluster and must not be selected.
export SGLANG_CI_RDMA_ALL_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8

nvidia-smi --query-gpu=index,name,memory.total --format=csv
python - <<'PY'
import nixl, nixl_cu13, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("nixl", getattr(nixl, "__version__", "unknown"))
print("nixl module", nixl.__file__)
print("nixl variant", nixl_cu13.__file__)
print("cuda available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
PY

if [[ "${RUN_UNIT_TESTS:-1}" == "1" ]]; then
  case "${UNIT_TEST_SUITE:-all}" in
    adapter)
      unit_tests=(test/registered/unit/disaggregation/test_nixl_embedding_transfer.py)
      ;;
    encoder)
      unit_tests=(
        test/registered/unit/disaggregation/test_encode_receiver.py
        test/registered/unit/disaggregation/test_encode_server.py
      )
      ;;
    all)
      unit_tests=(
        test/registered/unit/disaggregation/test_nixl_embedding_transfer.py
        test/registered/unit/disaggregation/test_encode_receiver.py
        test/registered/unit/disaggregation/test_encode_server.py
      )
      ;;
    *)
      echo "Unknown UNIT_TEST_SUITE=${UNIT_TEST_SUITE}" >&2
      exit 2
      ;;
  esac
  pytest_args=(-q)
  if [[ "${UNIT_TEST_VERBOSE:-0}" == "1" ]]; then
    pytest_args=(-vv -s)
  fi
  python -m pytest "${pytest_args[@]}" "${unit_tests[@]}"
fi

if [[ "${RUN_MICROBENCH:-1}" == "1" ]]; then
  microbench_backend=${MICROBENCH_BACKEND:-nixl}
  receiver_extra=()
  sender_extra=()
  if [[ "$microbench_backend" == "mooncake" ]]; then
    receiver_extra=(--local-host 127.0.0.1 --ib-device mlx5_1)
    sender_extra=(--local-host 127.0.0.1 --ib-device mlx5_0)
  fi

  python test/manual/encoder/bench_embedding_transfer.py \
    --role receiver --backend "$microbench_backend" --host 127.0.0.1 --port 29590 \
    --gpu-id 1 --sizes 4096 1048576 16777216 67108864 \
    --iterations 50 --warmup 10 --timeout 120 \
    "${receiver_extra[@]}" \
    >"$RESULTS/receiver.log" 2>&1 &
  receiver_pid=$!
  trap 'kill "$receiver_pid" 2>/dev/null || true' EXIT

  for _ in $(seq 1 60); do
    if grep -q "Listening" "$RESULTS/receiver.log" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$receiver_pid" 2>/dev/null; then
      cat "$RESULTS/receiver.log"
      exit 1
    fi
    sleep 1
  done

  python test/manual/encoder/bench_embedding_transfer.py \
    --role sender --backend "$microbench_backend" --host 127.0.0.1 --port 29590 \
    --gpu-id 0 --sizes 4096 1048576 16777216 67108864 \
    --iterations 50 --warmup 10 --timeout 120 \
    "${sender_extra[@]}" \
    --output "$RESULTS/$microbench_backend-gpu.json"
  wait "$receiver_pid"
  trap - EXIT
  cat "$RESULTS/receiver.log"
fi

if [[ "${RUN_FULL_EPD:-1}" == "1" ]]; then
  # The shared cache may contain the first weight shard from an interrupted
  # download. SGLang treats any local snapshot as offline-complete, so fetch
  # the second indexed shard explicitly before server startup.
  python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    filename="model-00002-of-00002.safetensors",
)
PY
  EPD_ENCODER_TRANSFER_BACKEND=nixl \
  SGLANG_TEST_PD_DISAGG_BACKEND=nixl \
  python test/manual/encoder/test_nixl_epd_smoke.py
fi
