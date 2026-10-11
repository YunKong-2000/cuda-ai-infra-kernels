# FP32 GEMM 优化链路

统一逻辑语义为 `C[M,N] = A[M,K] @ B[K,N]`。A 是列主序，stride 为 `(1,M)`；B、C 是行主序。前八阶段仅支持正数、完整 tile，host 检查尺寸、布局、设备、类型和 float4 地址对齐，kernel 内没有边界补零或不规则尺寸分支。Python 入口不隐式转换布局。

| 阶段 | impl | BM×BN×BK | threads/block | 每线程输出 |
| --- | --- | --- | --- | --- |
| 1 | `naive` | 16×16×1 | 256 | 一个元素，直接读 global |
| 2 | `block_tile` | 16×16×8 | 256 | 一个元素，标量搬运到 smem |
| 3 | `block_tile_float4_16x16` | 16×16×8 | 256 | 一个元素，float4 搬运 |
| 3 | `block_tile_float4_16x32` | 16×32×8 | 512 | 一个元素，float4 搬运 |
| 3 | `block_tile_float4_32x32` | 32×32×8 | 1024 | 一个元素，float4 搬运 |
| 4 | `thread_tile_4x4` | 128×128×8 | 1024 | 连续 4×4 |
| 4 | `thread_tile_4x8` | 128×128×8 | 512 | 连续 4×8 |
| 4 | `thread_tile_8x8` | 128×128×8 | 256 | 连续 8×8 |
| 5 | `warp_tile` | 128×128×8 | 256 | 8×8 拆成四个 4×4 子块 |
| 6 | `prefetch_smem` | 128×128×8 | 256 | 原有 smem 双缓冲 |
| 7 | `prefetch_reg` | 128×128×8 | 256 | 原有 smem + register 双缓冲 |
| 8 | `split_k` | 128×128×8，每个输出 tile 两个 CTA | 256 | 原有两路 K 切分、atomicAdd |
| 9 | `cublas` | 库选择 | 库选择 | 严格 FP32 对照 |

CUDA 每个 block 最多 1024 个线程。为保留第 3 阶段“一线程一元素”，按确认后的方案将原计划的 64×64 / 64×128 / 128×128 缩小为上述三种尺寸。第 4 阶段才扩展到 128×128，并复用 A/B 寄存器向量做外积。三种 thread tile 使用同一个 block tile，因此同时改变线程数与每线程寄存器需求；性能不保证随阶段递增。

第 2～5 阶段均只有一份 `As[BK][BM]`、`Bs[BK][BN]`，搬运后同步，计算完成再同步后覆盖。float4 沿 A 的 M 维、B 的 N 维搬运，均对应连续物理内存。第 4 阶段从 smem 读取连续 TM/TN 寄存器向量，第 5 阶段沿 M/N 各读两个相隔的 float4。

第 5 阶段与现有第 6～8 阶段采用相同映射：block 含 2×4 个 warp，每个 warp 覆盖 64×32；lane 按 8×4 排列，每线程持有两组连续 4 行、两组连续 4 列，行组间距 32、列组间距 16，组合出四个 4×4 子块。第 5 阶段没有预取或双缓冲。

`thread_tile` 保留为 `thread_tile_4x4` 的别名；全量测试不重复计入别名。旧 thread/warp 文件中混合的边界处理、额外 pipeline、未使用的 kernel 已删除。`gemm_prefetchSmem.cu`、`gemm_prefetchRegister.cu`、`gemm_splictK.cu` 原样保留；Tensor Core/CUTLASS 不参与这条 FP32 链路，保留原入口和实现。cuBLAS 保留对行主序 A 的兼容，但本脚本只向它传同一份列主序 A。

## 一键编译、校验和性能对比

在 NVIDIA CUDA 开发环境、仓库根目录执行（已安装 CUDA 版 PyTorch、setuptools、ninja，且仓库的 CUTLASS 子模块完整）：

```bash
bash scripts/bench_gemm_all.sh
```

脚本先 `setup.py build_ext --inplace`，再测试全部 13 个入口。现有 `cp.async` kernel 需要 SM80+；构建使用仓库 `setup.py`，由 PyTorch 检测目标 GPU，或通过 `TORCH_CUDA_ARCH_LIST` 显式指定，例如 A100 使用 `8.0`、RTX 4090 使用 `8.9`。无可见 GPU 时必须显式指定；CUDA toolkit 需支持目标架构。也可用 `make bench-gemm-all`。指定解释器与尺寸：

```bash
PYTHON=python3 bash scripts/bench_gemm_all.sh \
  --shapes 128,256,16 1024,1024,1024 2048,2048,2048 \
  --warmup 20 --repeat 100 --save results/raw/gemm_ladder.json
```

全量共同约束：`M % 128 == 0`、`N % 128 == 0`、`K % 16 == 0`，且均为正数。Split-K 的每个 K 分片需为 8 的倍数，因此共同 K 约束是 16。单独调用各个实现时，M/N/K 仅需满足表中对应倍数；naive 对 K 没有整除限制。向量化实现要求 A/B 起始地址 16 字节对齐。

默认规模与参考加速比热力图的七列一致，顺序为 `M×N×K`：`256×256×256`、`1024×1024×1024`、`2048×2048×2048`、`4096×4096×2048`、`4096×1024×4096`、`1024×4096×4096`、`4096×4096×4096`。每种规模测试全部 13 个实现，共 91 个“规模 × 实现”组合；`--shapes` 可覆盖默认列表。

每个尺寸生成一次 A/B，所有 kernel 和 cuBLAS 共用同一对象、相同物理布局。首先与独立 FP64 matmul 转 FP32 的结果逐项校验（`atol=rtol=1e-3`）；任一实现失败则退出，不为错误结果报性能。验证通过后才进行计时。

默认 `--timing graph` 在同一侧流预热并捕获 10 次调用，每次 replay 用 CUDA events 计时后除以 10，减少 Python launch 间隙。包含实际 GPU 运算及 Split-K 的清零；CPU 输入检查、分配工作和输入准备不计时。结果对应固定地址、重复输入的 steady-state 测量，不是冷缓存场景。`--timing event` 改用现有逐次 API 计时，较小尺寸可能受 host launch 间隙影响。不要混合两种计时方式的数值作直接比较。

终端输出 correctness、平均 ms、TFLOPS、`cuBLAS_ms / kernel_ms` 和 `naive_ms / kernel_ms`（大于 1 表示快于对照）。JSON 还包含误差、mean/p50/min/max、环境和 strides，默认保存到 `results/raw/gemm_all_*.json`。cuBLAS 使用 `CUBLAS_COMPUTE_32F_PEDANTIC`，不依赖全局 TF32 开关。

已编译时可直接运行 Python 入口：

```bash
PYTHONPATH=python python3 benchmarks/bench_gemm_all.py --shapes 128,256,16
PYTHONPATH=python python3 benchmarks/bench_gemm.py --impl thread_tile_4x8 --m 1024 --n 1024 --k 1024
```

## Python 调用与回归测试

```python
import torch
from cuda_ai_kernels import gemm

a = torch.randn(256, 128, device="cuda", dtype=torch.float32).t()  # [128,256], strides (1,128)
b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
c = gemm(a, b, impl="thread_tile_8x8")
```

原来给 naive/block/thread/warp 传行主序 A 或不完整 tile 的调用，需要迁移到以上列主序/完整 tile 接口；不会自动转置或回退。

```bash
python3 -m pytest -q tests/test_gemm.py tests/test_gemm_prefetch_smem.py \
  tests/test_gemm_split_k.py tests/test_bench_gemm_all.py tests/test_bench_gemm_prefetch_smem.py
compute-sanitizer --tool memcheck python3 -m pytest -q tests/test_gemm.py -k regular
compute-sanitizer --tool racecheck python3 -m pytest -q tests/test_gemm.py -k regular
```

新增 CUDA 用例覆盖最小 tile、多 block、矩形尺寸、单/多 K tile、错误布局/尺寸/地址对齐、非默认 stream 和 graph replay。CPU 测试检查同输入、先校验再计时、失败中止、性能比值以及 Python 分发不复制输入。
