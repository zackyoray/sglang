"""GPU serving tests for the nixl.runtime.ep.ElasticBuffer backend."""

import os
import unittest

from test_nixl_ep import DP_ATTN, NIXL_COMMON, _EPTestBase

os.environ.setdefault("SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

V2_COMMON = ["nixl_ep_v2" if arg == "nixl" else arg for arg in NIXL_COMMON]
V2_COMMON += [
    "--moe-runner-backend",
    "deep_gemm",
    "--chunked-prefill-size",
    "1024",
    "--cuda-graph-max-bs-decode",
    "128",
]


class TestNixlEPV2TP(_EPTestBase):
    server_args = V2_COMMON


class TestNixlEPV2DPAttn(_EPTestBase):
    server_args = [
        *V2_COMMON,
        *DP_ATTN,
        "--moe-dense-tp-size",
        "1",
        "--enable-dp-lm-head",
    ]


if __name__ == "__main__":
    unittest.main()
