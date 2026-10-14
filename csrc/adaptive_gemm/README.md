# CUTLASS 多配置 GEMM

为 SM80 提供预编译 CUTLASS GEMM 注册表、统一 launcher、兼容性分派和 epilogue 融合。当前支持 FP32 输入/输出与 FP32 累加，乘法使用 TF32 Tensor Core；不与手写 FP32 SIMT GEMM 的性能直接混比。

## API 与语义

```python
from cuda_ai_kernels import adaptive_gemm

d = adaptive_gemm(a, b, kernel="auto")
d = adaptive_gemm(a, b, c=c, alpha=0.75, beta=0.25,
                  epilogue="relu", kernel="auto")
```

- `linear`：`D = alpha * A @ B + beta * C`。
- `relu`：`D = relu(alpha * A @ B + beta * C)`。
- A/B/C/D 为 RowMajor CUDA FP32 张量，正数维度需满足当前包装器与 CUTLASS 的检查。
- 无 bias 广播、GELU/SiLU、低精度 dtype 或 reduction epilogue 的可用注册路径；类型别名和解析枚举不等于已实现能力。

## 注册配置

| kernel 参数 | CTA tile | Warp tile | Stages | A/B alignment（元素） | Epilogue |
| --- | --- | --- | ---: | ---: | --- |
| `fast` | 128×128×16 | 64×64×16 | 4 | 4 | LinearCombination |
| `fallback` | 128×128×16 | 64×64×16 | 4 | 1 | LinearCombination |
| `fast_mwarp` | 128×128×16 | 64×32×16 | 4 | 4 | LinearCombination |
| `fallback_mwarp` | 128×128×16 | 64×32×16 | 4 | 1 | LinearCombination |
| `fast_stage3` | 128×128×16 | 64×64×16 | 3 | 4 | LinearCombination |
| `fast_stage2` | 128×128×16 | 64×64×16 | 2 | 4 | LinearCombination |
| `fast_relu` | 128×128×16 | 64×64×16 | 4 | 4 | ReLU |
| `fallback_relu` | 128×128×16 | 64×64×16 | 4 | 1 | ReLU |

`auto` 按 [注册表顺序](kernel_dispatch.cu) 筛选 dtype、epilogue、指针与 leading-dimension 对齐条件，再调用 `can_implement` 和 launcher；首个成功项即返回。它是兼容性分派，不执行性能搜索，也没有保证某个 shape 选择最快配置。

显式指定 kernel 只尝试该项，不兼容就报错。fast 路径向量宽度为 4；fallback 为 scalar alignment，可处理非 4 倍数的 N/K。`fast_Relu` / `fallback_Relu` 保留为兼容别名。

## 构建、测试与测量

从仓库根目录执行：

```bash
python setup.py build_ext --inplace --force
python -m pytest -q tests/test_adaptive_gemm.py tests/test_bench_adaptive_gemm.py
python benchmarks/bench_adaptive_gemm.py --kernel all --m 1024 --n 1024 --k 1024
python benchmarks/bench_adaptive_gemm.py --epilogue relu --kernel all \
  --m 1024 --n 1024 --k 1024
python benchmarks/bench_adaptive_gemm.py --epilogue relu --kernel all --m 33 --n 65 --k 17
```

Linear `--kernel all` 包含 fast 路径，需要满足其对齐要求；ReLU 的 all 模式会对不兼容 fast 项记录跳过。ReLU 比较融合 kernel、CUTLASS 输出后 ReLU 和 cuBLAS 输出后 ReLU，所有计时包含等价后处理。

`kernel="cublas"` 仅支持 linear，采用 `CUBLAS_COMPUTE_32F_FAST_TF32`。benchmark 对照允许 TF32 的 PyTorch 结果，另记录完整 FP32 参考的误差。容差与结果字段以 [benchmark](../../benchmarks/bench_adaptive_gemm.py) 为准。允许 TF32 不保证 PyTorch 选择特定 Tensor Core kernel，也不要求逐 bit 相同。

当前 CUDA Event 计时包含输出分配和 host dispatch；小 shape 可能受主机提交间隙影响。非零 beta 使用独立 C 时，经典 cuBLAS 路径的 C→D 拷贝属于调用开销。

```bash
bash profiling/profile_adaptive_gemm.sh fast 4096 4096 4096 \
  profiling/adaptive_gemm_fast_4096.ncu-rep
```

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `adaptive_gemm.cu` | 参数检查、kernel/epilogue 解析、PyTorch stream 与输出 |
| `gemm_problem.*` | shape、stride、dtype 和标量参数 |
| `kernel_config.h` / `kernel_sm80.h` | 编译期 tile、stage、alignment、epilogue 类型 |
| `kernel_launcher.h` / `kernel_entry.h` | 统一 launcher 与元数据 |
| `kernel_dispatch.cu` | 8 项注册表与兼容性选择 |
| `cublas_reference.cu` | 同精度 TF32 cuBLAS 对照 |

返回 [项目首页](../../README.md) 或 [性能分析说明](../../docs/profiling_notes.md)。
