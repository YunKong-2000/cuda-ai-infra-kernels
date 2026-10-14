# CUDA GEMM Kernels

面向 NVIDIA GPU 的 **GEMM 算子实现与性能优化项目**，覆盖手写 CUDA C++ kernel 与 CUTLASS 集成，重点包括 FP32 GEMM 的分层并行、向量化访存、共享内存/寄存器预取与 Split-K，以及 CUTLASS 多配置分派和 epilogue 融合。通过 PyTorch 扩展提供统一调用、正确性对照、CUDA Event / Graph 计时和 Nsight Compute 入口。

## 主要实现

- **手写 FP32 GEMM**：从 global-memory baseline 到 CTA/thread/warp tile、float4 搬运、双缓冲、异步预取和两路 Split-K；同输入布局下比较各阶段的资源与延迟。
- **CUTLASS GEMM**：预编译多组 warp tile、pipeline stage、alignment 和 epilogue 配置；支持显式选择与兼容性分派，以及 LinearCombination / ReLU 融合。
- **工程接口**：C++/CUDA Extension、输入约束、当前 stream、边界回归，以及独立的测量与 profiling 脚本。

## FP32 GEMM 性能

**NVIDIA A800 80GB PCIe，FP32，A 列主序，B/C 行主序。** 下表取自保存的 CUDA Graph 测量，使用中位耗时计算吞吐；Split-K 计时包含输出清零。

| M×N×K | 实现 | 自定义耗时 | cuBLAS 耗时 | 自定义吞吐 | 相对 cuBLAS 吞吐 |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096×4096×4096 | `split_k` | 7.9878 ms | 7.4201 ms | 17.206 TFLOP/s | **92.89%** |
| 1024×4096×4096 | `split_k` | 2.1546 ms | 2.0467 ms | 15.947 TFLOP/s | **94.99%** |

相对吞吐为 `cuBLAS 时间 / 自定义时间`，不表示超过 cuBLAS。结果限定于所列 shape、精度、布局与计时方式。历史 CSV 和当前 benchmark 的统计口径、变体集合有所不同，不能直接混合比较。

[完整形状汇总与数据来源](docs/gemm_optimization_log.md) · [原始 CSV](docs/results/gemm_1791702354072.csv) · [实现分层与运行方法](docs/gemm_optimization_ladder.md)

## 实现范围

| 模块 | 当前范围 | 说明 |
| --- | --- | --- |
| 手写 GEMM | FP32；当前统一链路 14 个入口（含 cuBLAS） | 完整 tile；A 列主序；SM80+ 预取路径 |
| [Adaptive GEMM](csrc/adaptive_gemm/README.md) | FP32 输入/输出，TF32 Tensor Core；LinearCombination / ReLU | 按注册顺序选兼容配置，不执行在线性能搜索 |
| RMSNorm | FP32 `naive` / `cache_x` | [实现与测试](docs/rmsnorm_optimization_log.md) |
| Softmax | FP32 `naive` | [实现与测试](docs/softmax_optimization_log.md) |
| RoPE / attention_decode | API 占位，尚无可调用计算实现 | Transformer 算子实现见相关项目 |

本仓库关注单 GPU 算子实现与测量，不包含训练框架、分布式执行或在线推理服务。Transformer 前反向与 Attention 的相关项目：[Tiny Transformer](https://github.com/YunKong-2000/Tiny-transformer)。

项目展示名称为 **CUDA GEMM Kernels**（`cuda-gemm-kernels`）。当前 GitHub 地址仍为 `YunKong-2000/cuda-ai-infra-kernels`；下方命令使用有效地址并克隆到新名称目录。Python 包 `cuda_ai_kernels` 和容器内部路径保持兼容。

## 环境与安装

需要 CUDA 版 PyTorch、与之兼容的 CUDA toolkit / nvcc、C++ 编译器、setuptools、Ninja，以及完整的 CUTLASS 子模块。异步拷贝 kernel 要求 SM80 或更新架构。

```bash
git clone --recurse-submodules https://github.com/YunKong-2000/cuda-ai-infra-kernels.git cuda-gemm-kernels
cd cuda-gemm-kernels
# A100 / A800；其他 GPU 按实际架构设置。
export TORCH_CUDA_ARCH_LIST=8.0
python -m pip install --no-build-isolation -e '.[dev]'
```

保留环境中与 CUDA 匹配的 PyTorch。修改 C++/CUDA 后重新构建：

```bash
python setup.py build_ext --inplace --force
```

容器配置见 [Docker 指南](docs/docker.md)。

## GEMM 快速运行

```bash
# 构建扩展，逐项校验后运行全部 FP32 GEMM 变体。
bash scripts/bench_gemm_all.sh --shapes 1024,4096,4096 4096,4096,4096 \
  --timing graph --warmup 10 --repeat 50 --save results/raw/gemm_all.json

# 正确性、布局/对齐、stream 与测量框架回归。
python -m pytest -q tests/test_gemm.py tests/test_gemm_split_k.py \
  tests/test_gemm_prefetch_smem.py tests/test_bench_gemm_all.py
```

所有变体使用同一组 A/B；在计时前与 FP64 matmul 转 FP32 的参考结果检查。当前脚本报告 mean/p50/min/max，其中终端吞吐与速度比使用 mean。Graph 包含设备计算及 Split-K 清零，排除捕获时的主机分配与输入准备。历史 CSV 的中位数统计另见结果页。

```python
import torch
from cuda_ai_kernels import gemm

# A 的逻辑 shape 为 [1024,4096]，stride 为 (1,1024)。
a = torch.randn(4096, 1024, device="cuda", dtype=torch.float32).t()
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
c = gemm(a, b, impl="split_k")
```

完整链路要求 M/N 为 128 的倍数、K 为 16 的倍数且均为正数；不会隐式转置或回退。其他算子的参数见各模块文档。

## CUTLASS 分派与融合

```bash
python benchmarks/bench_adaptive_gemm.py --kernel all --m 1024 --n 1024 --k 1024
python benchmarks/bench_adaptive_gemm.py --epilogue relu --kernel all \
  --m 1024 --n 1024 --k 1024
python -m pytest -q tests/test_adaptive_gemm.py tests/test_bench_adaptive_gemm.py
```

此路径使用 TF32 Tensor Core，与上方手写 FP32 SIMT 数据分开报告。ReLU 基线包含相同后处理。[API、注册表与边界](csrc/adaptive_gemm/README.md)。

## 文档与目录

| 路径 | 内容 |
| --- | --- |
| [docs/](docs/README.md) | 文档索引、优化记录及测量方法 |
| [csrc/gemm/](csrc/gemm/) | 手写 GEMM 与 cuBLAS 对照 |
| [csrc/adaptive_gemm/](csrc/adaptive_gemm/README.md) | CUTLASS 注册表、launcher 和 dispatcher |
| [python/cuda_ai_kernels/](python/cuda_ai_kernels/) | Python API、GEMM 阶段定义与公共计时 |
| [tests/](tests/) | PyTorch 对照及边界回归 |
| [benchmarks/](benchmarks/) | 算子基准与 JSON 输出 |
| [profiling/](profiling/) | Nsight Compute 入口 |

CUTLASS 为独立上游依赖，来源及许可保留在 `third_party/cutlass/`。
