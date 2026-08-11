#!/usr/bin/env bash
set -euo pipefail

IMPL="${1:-tiled}"
M="${2:-4096}"
N="${3:-4096}"
K="${4:-4096}"

ncu \
  --target-processes all \
  --metrics-file profiling/ncu_metrics.txt \
  python benchmarks/bench_gemm.py \
    --impl "${IMPL}" \
    --m "${M}" \
    --n "${N}" \
    --k "${K}" \
    --warmup 10 \
    --repeat 20 \
    --profile

