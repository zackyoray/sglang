# NIXL EP v2 backend

Select the backend with `--moe-a2a-backend nixl_ep_v2`. The original `nixl`
backend continues to use `nixl_ep.Buffer`.

This adapter targets the `nixl.runtime.ep.ElasticBuffer` API in NIXL revision
`16e0da3c14956fc6045b936a7a5d6e779614ed11` from the elastic-EP repository.
Build that NIXL revision with `-Dbuild_nixl_ep=true`, following its
`runtime/ep/README.md` for UCX, CUDA, and runtime paths. The older `nixl_ep`
package alone is insufficient.

For a model supported by SGLang's DeepEP v2 compute path:

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 \
  --moe-a2a-backend nixl_ep_v2 \
  --moe-runner-backend deep_gemm
```

NIXL determines direct/hybrid communication from the NVLink topology;
`--deepep-v2-mode` does not select NIXL's transport mode. NIXL's `NIXL_EP_*`
settings control its transport and JIT compilation.

The adapter shares the DeepEP v2 tensor layout and DeepGEMM compute path,
including BF16/FP8 dispatch selection, masked decode, and expanded prefill.
Shared tuning uses `SGLANG_DEEPEP_V2_*`, notably
`SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`. Increase that capacity
when the per-rank prefill/decode budget requires it.

Static EP requires equal expert slots per rank. TBO/SBO, fused shared experts,
and use as a speculative draft backend are rejected. Prefill CUDA graphs
are disabled, matching the shared compute path.

Elastic scale-up uses a maximum topology reserved when the buffer is created.
Set `SGLANG_NIXL_EP_V2_MAX_SCALEUP_RANKS` to the number of ranks per NVLink
domain in that maximum topology; it must divide `--max-ep-size`. For a
single-node 4-to-8 GPU test, set both to 8. Joining ranks publish transport
metadata through SGLang's global TCPStore before the join-ready barrier.
Scale-up connects the new peers and commits the expanded active world.

Fault recovery is unsupported by this integration. Combining `nixl_ep_v2`
with `--elastic-ep-join-mode recover` is rejected during argument validation,
before transport initialization. Bootstrap requires the constructor cohort to
participate together; restarting one rank cannot rejoin surviving buffers.
Successful scale-up does not establish fault tolerance.

Run the basic serving test and then the existing elastic scale test with v2:

```bash
export NIXL_EP_TEST_MODEL=/path/to/model
export SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
python -m pytest -v -s test/manual/ep/test_nixl_ep_v2.py::TestNixlEPV2DPAttn

export SGLANG_ELASTIC_MOE_A2A_BACKEND=nixl_ep_v2
export SGLANG_NIXL_EP_V2_MAX_SCALEUP_RANKS=8
export SGLANG_ELASTIC_EXTRA_SERVER_ARGS="--moe-runner-backend deep_gemm --disable-overlap-schedule --moe-dense-tp-size 1"
python -m pytest -v -s test/manual/ep/test_elastic_scale.py::TestElasticScaleUp4To8
```

The DP-attention test uses dense TP=1, which also accommodates FP8 checkpoints
whose dense weight dimensions cannot be partitioned into eight quantization
blocks. The separate `TestNixlEPV2TP` class requires a compatible checkpoint.
The container needs SGLang's testing dependencies, including `sgl-eval`.

Run the dependency-isolated adapter contract tests with:

```bash
python test/manual/ep/test_nixl_ep_v2_adapter.py -v
```

These tests mock the transport and tensor boundaries. They do not establish
GPU numerical correctness, multi-node correctness, or performance; those
require an installed NIXL v2 / SGLang GPU environment.

## Cluster validation

Validated on one node with eight H100 80GB GPUs using the NIXL revision above
and DeepSeek-V2-Lite-Chat (BF16), DP attention, and dense TP=1:

| Test | Result | GSM8K |
| --- | --- | --- |
| `TestNixlEPV2DPAttn` | Passed, decode CUDA graphs enabled | 63.5% / 200 examples |
| `TestElasticScaleUp4To8` with v2 | Passed | 60.16% / 256 examples |
| `TestElasticScaleUp4To8CudaGraph` with v2 | Passed, graphs recaptured at size 8 | 60.94% / 256 examples |

The existing accuracy thresholds (60% basic, 50% scale) were unchanged.
The scale tests used Mooncake coordination and
`SGLANG_CI_RDMA_ALL_DEVICES=mlx5_0` to select a valid NIC on that node.
Use the appropriate NIC for your cluster. The environment also set
`SGLANG_DEEPEP_BF16_DISPATCH=1` for this BF16 checkpoint.

Separate eight-rank transport checks passed for eager dispatch/combine,
an empty-token rank, and CUDA graph replay. Multi-node serving, scale-down,
and asynchronous scaling remain unvalidated. Fault recovery is unsupported.
