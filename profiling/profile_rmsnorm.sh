#!/usr/bin/env bash
set -euo pipefail

IMPL="${1:-cuda}"
ROWS="${2:-4096}"
HIDDEN="${3:-4096}"

ncu \
  --target-processes all \
  --metrics-file profiling/ncu_metrics.txt \
  python benchmarks/bench_rmsnorm.py \
    --impl "${IMPL}" \
    --rows "${ROWS}" \
    --hidden "${HIDDEN}" \
    --warmup 10 \
    --repeat 20 \
    --profile

