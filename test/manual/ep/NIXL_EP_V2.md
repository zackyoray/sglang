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

The initial integration requires a fixed EP world with equal expert slots
per rank. SGLang elastic membership changes, TBO/SBO, fused shared experts,
and use as a speculative draft backend are rejected. Prefill CUDA graphs
are disabled, matching the shared compute path.

Run the dependency-isolated adapter contract tests with:

```bash
python test/manual/ep/test_nixl_ep_v2_adapter.py -v
```

These tests mock the transport and tensor boundaries. They do not establish
GPU numerical correctness, multi-node correctness, or performance; those
require an installed NIXL v2 / SGLang GPU environment.
