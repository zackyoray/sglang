#!/bin/bash
set -euo pipefail

IMAGE=/lustre/fs1/portfolios/coreai/projects/coreai_tritoninference_triton3/users/yorayz/sglang/sglang_nightly_dev_cu13_20260915_8874c51a.sqsh
SCRIPT=/lustre/fs1/portfolios/coreai/projects/coreai_tritoninference_triton3/users/yorayz/sglang-nixl-ep/test/manual/encoder/run_slurm_nixl_ep.sh

srun --container-image="$IMAGE" --container-mounts=/lustre:/lustre "$SCRIPT"
