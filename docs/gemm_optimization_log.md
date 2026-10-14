# FP32 GEMM：A800 性能记录

## 数据与环境

来源：[gemm_1791702354072.csv](results/gemm_1791702354072.csv)。完整保留 7 组形状、140 行结果，包含基础实现、tile、预取、Split-K 及其布局变体和 cuBLAS。

- GPU：NVIDIA A800 80GB PCIe，compute capability 8.0。
- 数据精度：FP32（原始实验说明，且与当前手写 kernel 的 float 类型一致）。
- CSV 记录：timing=graph，warmup=10，repeat=50，batch=10，seed=0。
- driver_version=13020，runtime_version=12080，cublas_version=120804；保留原始 API 数值。
- A 为 column_major；当前实现对应 B/C row_major。
- CSV 的 `verified` 均为 true；这是历史记录的标记，不代表本次文档整理重新执行了 GPU 回归。

## 各形状的最短自定义中位耗时

单位：ms。按该形状下非 cuBLAS 记录的 median_ms 选择最快变体；完整比较与退化结果保留在 CSV 中。

| M×N×K | 实现 | 自定义 median | cuBLAS median | 自定义 TFLOP/s | 相对 cuBLAS 吞吐 |
| --- | --- | ---: | ---: | ---: | ---: |
| 256×256×256 | `naive` | 0.014746 | 0.009114 | 2.2756 | 61.81% |
| 1024×1024×1024 | `prefetch_reg` | 0.233882 | 0.132352 | 9.1819 | 56.59% |
| 2048×2048×2048 | `split_k` | 1.119437 | 0.966298 | 15.3469 | 86.32% |
| 4096×4096×2048 | `prefetch_reg` | 4.082995 | 3.687066 | 16.8307 | 90.30% |
| 4096×1024×4096 | `split_k` | 2.156442 | 1.943194 | 15.9335 | 90.11% |
| 1024×4096×4096 | `split_k` | 2.154650 | 2.046720 | 15.9468 | 94.99% |
| 4096×4096×4096 | `split_k` | 7.987763 | 7.420109 | 17.2062 | 92.89% |

吞吐为 `2*M*N*K / (median_ms*1e9)`；相对吞吐为 `cuBLAS median_ms / candidate median_ms`。4096³ 与 1024×4096×4096 的结果分别为 92.8934% 和 94.9909%，可概括为约 93% 和 95%，不是超过 cuBLAS 的百分比。

两项 `split_k` 数据的 `includes_output_clear=true`；其计时包括原子合并前必需的输出清零。最大绝对误差分别为 2.0226e-4 和 1.6900e-4。CSV 未附校验参考算法/容差，不据此补造数值标准。

## 实现解释

当前 [Split-K 源码](../csrc/gemm/gemm_splictK.cu) 对每个输出 tile 划分两个 K 分片，使用 FP32 累加、float4 搬运、cp.async 双缓冲及 atomicAdd 合并。适用正数 M/N 为 128 的倍数、K 为 16 的倍数，输入需满足布局与对齐条件。

当前 [cuBLAS 对照](../csrc/gemm/gemm_cublas.cu) 使用 `cublasGemmEx` / `CUBLAS_COMPUTE_32F_PEDANTIC`；[CUTLASS adaptive GEMM](../csrc/adaptive_gemm/README.md) 的 TF32 Tensor Core 基线独立报告。

## 历史数据与当前脚本

CSV 未包含源码 commit、显式 dtype 或 cuBLAS math mode 字段，因此不声称历史二进制与当前源码逐字一致。历史表每个 shape 包含 20 个变体，当前 `GEMM_STAGES` 注册 14 个入口；历史中的部分 contiguous8x8 预取/Split-K 变体不在当前默认集合中。

CSV 中 tflops / speedup 列与 **median_ms** 相符；当前 `bench_gemm_all.py` 终端及 JSON 汇总使用 **mean_ms**，同时保留 p50_ms。复测时应从同一统计量计算对照，不能直接把不同版本输出当成同口径优化。

```bash
bash scripts/bench_gemm_all.sh --shapes 4096,4096,4096 1024,4096,4096 \
  --timing graph --warmup 10 --repeat 50 --save results/raw/gemm_a800.json
```

当前 Graph 入口每次捕获 10 次调用，对 replay 用 CUDA Events 计时后除以 10；新结果保存 commit、环境、精度、layout、误差与统计口径。详见 [实现分层](gemm_optimization_ladder.md)。

原始 CSV SHA256：`9171ee30203540b1ca6704cc64ae67254b483ce13d77dc9297403a4e163a34d5`。
