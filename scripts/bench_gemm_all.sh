#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON:-python3}"
# Existing async-copy kernels require Ampere or newer. Supply the target GPU
# architecture explicitly when building on a machine without GPU visibility.
"$PYTHON_BIN" setup.py build_ext --inplace
export PYTHONPATH="$ROOT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" benchmarks/bench_gemm_all.py "$@"
