#!/usr/bin/env bash
set -euo pipefail

IMPL="${1:-cuda}"
ROWS="${2:-4096}"
COLS="${3:-2048}"

ncu \
  --target-processes all \
  --metrics-file profiling/ncu_metrics.txt \
  python benchmarks/bench_softmax.py \
    --impl "${IMPL}" \
    --rows "${ROWS}" \
    --cols "${COLS}" \
    --warmup 10 \
    --repeat 20 \
    --profile

