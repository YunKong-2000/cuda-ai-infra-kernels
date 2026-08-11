# Profiling Notes

## Environment

- GPU:
- Driver:
- CUDA:
- PyTorch:

## Method

- Warmup:
- Repeat:
- Timing:
- Nsight Compute command:

## Metrics To Watch

- SM throughput
- DRAM throughput
- L2 throughput
- Achieved occupancy
- Register usage
- Shared memory usage
- Warp stall reasons
- Global memory load/store efficiency
- Shared memory bank conflicts

## Findings

| Kernel | Impl | Shape | Bottleneck | Evidence | Next Step |
| --- | --- | --- | --- | --- | --- |
| GEMM | naive | TBD | TBD | TBD | TBD |
| RMSNorm | cuda | TBD | TBD | TBD | TBD |
| Softmax | cuda | TBD | TBD | TBD | TBD |

