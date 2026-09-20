#!/usr/bin/env bash
set -euo pipefail

KERNEL="${1:-fast}"
M="${2:-4096}"
N="${3:-4096}"
K="${4:-4096}"
OUTPUT="${5:-adaptive_gemm_${KERNEL}_${M}x${N}x${K}.ncu-rep}"

case "${KERNEL}" in
  fast|fallback|fast_mwarp|fallback_mwarp|auto)
    ;;
  *)
    echo "kernel must be one of: fast, fallback, fast_mwarp, fallback_mwarp, auto" >&2
    exit 2
    ;;
esac

if [[ ! "${M}" =~ ^[1-9][0-9]*$ ||
      ! "${N}" =~ ^[1-9][0-9]*$ ||
      ! "${K}" =~ ^[1-9][0-9]*$ ]]; then
  echo "M, N, and K must be positive integers" >&2
  exit 2
fi

if [[ "${KERNEL}" == "fast" || "${KERNEL}" == "fast_mwarp" ]] && (( N % 4 != 0 || K % 4 != 0 )); then
  echo "${KERNEL} requires N and K to be divisible by 4" >&2
  exit 2
fi

if [[ "${OUTPUT}" != *.ncu-rep ]]; then
  OUTPUT="${OUTPUT}.ncu-rep"
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu was not found in PATH" >&2
  exit 127
fi

ncu \
  --target-processes all \
  --set full \
  --replay-mode kernel \
  --nvtx \
  --nvtx-include "adaptive_gemm_profile/" \
  --import-source yes \
  --source-folders csrc \
  --force-overwrite \
  --export "${OUTPUT}" \
  python benchmarks/bench_adaptive_gemm.py \
    --kernel "${KERNEL}" \
    --m "${M}" \
    --n "${N}" \
    --k "${K}" \
    --warmup 5 \
    --repeat 1 \
    --profile

echo "Nsight Compute report: ${OUTPUT}"
