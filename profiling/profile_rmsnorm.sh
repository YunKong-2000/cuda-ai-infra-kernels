#!/usr/bin/env bash
set -euo pipefail

IMPL="${1:-naive}"
ROWS="${2:-4096}"
HIDDEN="${3:-4096}"
OUTPUT="${4:-profiling.ncu-rep}"
ncu \
  --target-processes all \
  --set full \
  --import-source yes \
  --source-folders csrc \
  -f \
  -o "${OUTPUT}" \
  python benchmarks/bench_rmsnorm.py \
    --impl "${IMPL}" \
    --rows "${ROWS}" \
    --hidden "${HIDDEN}" \
    --warmup 10 \
    --repeat 20 \
    --profile

