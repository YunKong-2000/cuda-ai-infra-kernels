#!/usr/bin/env bash
set -euo pipefail

IMPL="${1:-naive}"
M="${2:-4096}"
N="${3:-4096}"
K="${4:-4096}"
OUTPUT="${5:-profiling.ncu-rep}"

ncu \
  --target-processes all \
  --set full \
  --import-source yes \
  --source-folders csrc \
  -f \
  -o "${OUTPUT}" \
  python benchmarks/bench_gemm.py \
    --impl "${IMPL}" \
    --m "${M}" \
    --n "${N}" \
    --k "${K}" \
    --warmup 1 \
    --repeat 1 \
    --profile
  

