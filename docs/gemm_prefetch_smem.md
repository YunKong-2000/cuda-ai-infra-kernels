# Shared-memory and register prefetch GEMM

`gemm(a, b, impl="prefetch_smem")` and `gemm(a, b, impl="prefetch_reg")` compute
`C[M,N] = A[M,K] @ B[K,N]` in FP32. Both use two shared-memory stages and the
same input restrictions and asynchronous global-to-shared copies:

- `prefetch_smem` uses SM80+ 16-byte `cp.async.cg.shared.global` copies. It
  commits the next A/B tile's copies together, computes the current tile, then
  executes `cp.async.wait_group 0` and a block barrier before switching stages.
  The initial tile is committed and waited on before the first computation.
  The wait completes each thread's copies; the barrier permits other threads
  to consume them and protects reuse of the old stage. LDS/FFMA still use one
  register fragment per operand.
- `prefetch_reg` uses the same asynchronous copies for its outer
  double buffering and adds two register buffers for shared-memory operands:
  preload row 0, load row k+1 while computing row k, and compute the final row
  separately. Its block/warp/thread mapping matches `prefetch_smem`.

A must have strides `(1, M)` (column-major); B and C are row-major. The interface
preserves this layout and performs no implicit input copies. If your existing
storage is contiguous `[K,M]`, pass `a_storage.t()`, a zero-copy view.

The current 256-thread, two-stage kernel supports positive M/N divisible by 128
and K divisible by 8. Inputs must be CUDA FP32 tensors on one device, with
16-byte-aligned data pointers. Unsupported shapes/layouts are rejected rather
than silently truncated. Arbitrary strides and edge tiles are not supported.

All runtime size, layout and pointer-alignment checks run on the host before
launch. The kernel has no element bounds checks, alignment checks, masked
loads/stores or scalar fallback. Its `alignas(16)` / `__align__(16)` declarations
only specify compile-time storage alignment. The main K loop always prefetches
the next tile; the final tile is computed separately so no per-iteration
"next tile exists" branch is needed inside the loop body.

```python
import torch
from cuda_ai_kernels import gemm

m, n, k = 256, 384, 40
# Logical A[M,K], physically column-major; no transpose copy.
a = torch.randn(k, m, device="cuda", dtype=torch.float32).t()
b = torch.randn(k, n, device="cuda", dtype=torch.float32)
c = gemm(a, b, impl="prefetch_smem")
c_reg = gemm(a, b, impl="prefetch_reg")
reference = gemm(a, b, impl="cublas")  # Exactly the same tensor objects/storage.
torch.testing.assert_close(c, reference, atol=1e-3, rtol=1e-3)
torch.testing.assert_close(c_reg, reference, atol=1e-3, rtol=1e-3)
```

The cuBLAS interface supports both the existing row-major A and the new
column-major A. For column-major A it computes `C^T = B^T @ A^T` with
`op(B_storage)=N`, `op(A_storage)=T`, `lda=N`, `ldb=M`, and `ldc=N`.
The reference uses `CUBLAS_COMPUTE_32F_PEDANTIC` (strict FP32, no TF32), including
for row-major inputs. Results/performance therefore need not match old cuBLAS
measurements that allowed reduced-precision algorithms. Both implementations
use the input device and PyTorch's current CUDA stream.

## Build and correctness

Run from the repository root on a CUDA development machine (SM80+; PyTorch detects the GPU or uses
`TORCH_CUDA_ARCH_LIST`):

```bash
python3 -m pip install -e '.[dev]' --no-build-isolation
python3 -m pytest -q tests/test_gemm_prefetch_smem.py tests/test_bench_gemm_prefetch_smem.py tests/test_gemm.py
```

The same tests run against both prefetch implementations. They cover
one/two/multiple K tiles, unequal M/N/K, multiple output blocks,
invalid sizes/layouts/alignment, aligned storage offsets, non-default streams,
and device guards (the last requires two GPUs). Exact-valued tests with a
distinct contribution from each K row detect repeated or dropped register
prefetch rows, including across shared-memory stage transitions. An independent FP64 matmul
reference is cast to FP32 for correctness checks. CPU tests verify that dispatch
preserves layout and that both benchmark implementations receive the same inputs.

Optional memory and synchronization checks:

```bash
compute-sanitizer --tool memcheck python3 -m pytest -q tests/test_gemm_prefetch_smem.py -k 'matches_fp64 or each_k'
compute-sanitizer --tool racecheck python3 -m pytest -q tests/test_gemm_prefetch_smem.py -k 'matches_fp64 or each_k'
```

## Compare against cuBLAS

```bash
python3 benchmarks/bench_gemm_prefetch_smem.py --m 1024 --n 1024 --k 1024 --warmup 50 --repeat 200
python3 benchmarks/bench_gemm_prefetch_smem.py --m 256 --n 384 --k 40 --save results/raw/prefetch_rectangular.json
python3 benchmarks/bench_gemm_prefetch_smem.py --impl prefetch_reg --m 1024 --n 1024 --k 1024 --warmup 50 --repeat 200
python3 benchmarks/bench_gemm_prefetch_smem.py --impl prefetch_reg --m 256 --n 384 --k 40 --save results/raw/prefetch_reg_rectangular.json
```

The script retains its original filename and defaults to `--impl prefetch_smem`;
use `--impl prefetch_reg` for the register version. Both commands construct the
same values for the same shape and seed. Each run compares the selected prefetch
implementation with cuBLAS using exactly the same tensor objects.

The script first checks both implementations against FP64, then warms up and
measures each with CUDA events using the same A/B objects. Input creation,
layout preparation and correctness checks occur outside the timed region.
Each timed call includes the normal allocating GEMM API; these measurements are
not isolated preallocated-output kernel timings, and host launch/allocation gaps
can affect small shapes. JSON records GPU/environment, strides, maximum absolute
errors, mean/median/min/max latency, TFLOPS and `prefetch_speedup_over_cublas`.
A speedup above 1 means the selected prefetch kernel is faster. There is no assumed performance
threshold. Results are printed and saved under `results/raw` unless `--save`
specifies a different path. Avoid other GPU workloads during measurement.

The existing single-implementation benchmark also accepts `--impl prefetch_smem` or `--impl prefetch_reg`
and prepares column-major A outside timing. Use the dedicated paired benchmark
above for comparisons on identical physical inputs.

Register prefetching is not guaranteed to improve performance. Inspect ptxas
register/spill reports and generated SASS alongside measured latency: the extra
operand registers may reduce occupancy, and instruction scheduling determines
whether LDS latency is actually hidden.
