# A100 Shape-Adaptive Fused GEMM 学习项目

这个目录包含项目设计、FP32 LinearCombination baseline、kernel registry 和运行时
dispatcher。bias、activation、更多 tile/dtype、专项 benchmark 和 profiling 仍待完成。

## 1. 项目目标

最终实现一个面向 A100（SM80）的 CUTLASS GEMM 小型算子库：

- 支持多个数据类型、矩阵形状和对齐条件。
- 预编译多种 GEMM 配置，并在运行时选择合适的 kernel。
- 支持 bias 和逐元素 activation 的 epilogue fusion。
- 支持至少一种需要 reduction 的 GEMM 后处理。
- 提供 PyTorch 接口、正确性测试、可复现 benchmark 和 Nsight Compute 分析。
- 能解释每个模板参数、dispatch 规则以及性能变化的原因。

这个项目不要求全面超过 cuBLAS。项目价值在于形成完整的实验链路，并能够解释
什么形状有效、什么形状无效，以及原因是什么。

## 2. 最终模块边界

建议你逐步创建以下文件，而不是一开始全部建好：

```text
csrc/adaptive_gemm/
  adaptive_gemm.h          对外 C++ 接口和参数定义
  kernel_config.h          kernel ID、tile、stage、alignment 元数据
  kernel_sm80.h            CUTLASS kernel 类型
  kernel_launcher.h        类型擦除 launcher 和 metadata 构造
  kernel_dispatch.cu       kernel registry 和运行时选择逻辑
  epilogue_ops.cuh         自定义 activation 或 epilogue functor
  reduction.cu             partial/final reduction，最后阶段再增加
  README.md                本设计书和实验结论
```

项目外还需要逐步接入：

```text
csrc/bindings.cpp                    PyTorch binding
python/cuda_ai_kernels/ops.py        Python API
tests/test_adaptive_gemm.py          正确性和边界测试
benchmarks/bench_adaptive_gemm.py    性能测试
docs/adaptive_gemm_optimization.md   每轮实验记录
```

## 3. 阶段一：建立可信 baseline

### 需要实现

- 把现有 `gemm_cutlass.cu` 的单一 kernel 迁移为本模块的第一个 baseline。
- 暂时只支持 FP32 输入、FP32 accumulator、RowMajor A/B/D 和无融合输出。
- 保留 `can_implement()` 检查，并使用 PyTorch 当前 CUDA stream。
- 增加 PyTorch reference、cuBLAS 和当前 CUTLASS kernel 三组结果。

### 需要理解

- `ElementA/B/C` 和 `ElementAccumulator` 的区别。
- `OpClassTensorOp`、`Sm80`、instruction shape 分别控制什么。
- float 输入为什么会在 SM80 Tensor Core 路径中使用 TF32。
- `ThreadblockShape`、`WarpShape`、`InstructionShape` 的整除关系。
- leading dimension 如何由 PyTorch tensor layout 映射到 CUTLASS `TensorRef`。

### 验收标准

- 对齐和非对齐 shape 都有正确性测试。
- 明确记录 TF32 误差容限，不能直接沿用 FP32 SIMT 的容限。
- benchmark 使用 CUDA Event、有 warmup，不把首次初始化时间计入结果。
- 能画出 global memory、shared memory、register、MMA、epilogue 的数据通路。

## 4. 阶段二：手工建立 kernel registry

### 需要实现

定义多个编译期 CUTLASS kernel，并为每个类型编写统一 launcher。先测试以下候选，
不要预先假定它们一定最快：

| 场景 | Threadblock 候选 | Warp 候选 | Stage 候选 |
| --- | --- | --- | --- |
| M/N 都较大 | 128x128x16 | 64x64x16 | 3、4、5 |
| M 较小 | 64x128x16 | 32x64x16 | 3、4、5 |
| N 较小 | 128x64x16 | 64x32x16 | 3、4、5 |
| 较小方阵 | 64x64x16 | 32x32x16 | 3、4 |

每个编译期类型应被包装成相同签名的运行时 launcher。运行时 dispatcher 只能从
已编译的 launcher 中选择，不能在运行时改变 C++ 模板参数。

### 实现顺序

1. 固定 tile，只改变 stage，观察 shared memory 和 occupancy。
2. 固定 stage，只改变 CTA/warp tile，观察并行度和数据复用。
3. 增加 `Alignment=4` 的向量化版本。
4. 增加 `Alignment=1` 的通用 fallback。
5. 给每个配置稳定的 kernel ID，并让 API 可以返回本次选择的 ID。

### 验收标准

- 至少有四种 tile/stage 组合，不是简单复制类型别名。
- 测试覆盖 `M/N/K` 尾块、奇数尺寸和未对齐 leading dimension。
- 能解释为什么某个 tile 在大矩阵上快、在小 M 上反而慢。
- 使用 Nsight Compute 记录 registers/thread、shared memory/CTA、occupancy 和 Tensor Core 利用率。

## 5. 阶段三：shape-adaptive dispatch

### 需要实现

先做可解释的启发式选择器，输入至少包括：

- M、N、K。
- A/B/D 指针对齐和 leading dimension 对齐。
- dtype。
- 是否有 fused epilogue。
- 后续加入 Split-K 时的 workspace 限制。

第一版规则只需区分 large、small-M、small-N 和 fallback。之后编写离线扫描程序，
对代表性 shape 运行所有可用 kernel，把最快结果保存为 lookup table。不要把 benchmark
直接放进每次线上调用路径。

建议覆盖的 shape：

- M：1、8、32、128、512、2048、8192。
- N/K：768、1024、2048、4096、11008。
- 额外增加 17、65、130 等非对齐尺寸。

### 验收标准

- dispatcher 的决定可查询、可记录、可复现。
- 启发式和 lookup table 分开实现。
- 对每条规则都有 benchmark 数据，而不是只凭经验写阈值。
- fallback 永远保证正确性，不允许只覆盖理想尺寸。

## 6. 阶段四：epilogue fusion

按以下顺序实现，每完成一种就增加 reference、测试和性能对比。

### 4.1 LinearCombination

实现 `D = alpha * Acc + beta * C`，学习 epilogue vector width、alpha/beta 参数和
source-needed 判断。

### 4.2 Bias broadcast

把 `[N]` bias 作为 C operand，并令 RowMajor C 的行 stride 为 0，使所有输出行读取
同一个 bias 向量。需要验证 bias 指针和 N 是否满足 epilogue 向量宽度要求。

### 4.3 ReLU、GELU、SiLU

- 先使用 CUTLASS 已有的 `LinearCombinationRelu` 和 `LinearCombinationGELU`。
- 再用 `LinearCombinationGeneric` 实现一个自定义 activation。
- 比较普通 GELU 和近似 GELU 的精度、指令成本及 kernel latency。

### 4.4 Residual 和多输入 epilogue

普通 2.x output functor 主要接收 accumulator 和一个 source tensor。需要两个独立输入
时，先研究 CUTLASS example 13、45 和 SM80 epilogue visitor，再决定是扩展 epilogue、
限制输入布局，还是暂时使用第二个 kernel。不要通过提前把 bias 和 residual 相加来伪装
成融合实现。

### 验收标准

- 分别测量 GEMM、GEMM+bias、GEMM+bias+activation。
- PyTorch baseline 必须执行等价的非融合操作。
- 同时报告 kernel 数量、latency 和估算的中间显存流量。
- 说明 fusion 在计算密集大 GEMM 上可能收益较小，在小 batch 或 bandwidth-sensitive
  场景中可能收益更明显。

## 7. 阶段五：Batched、Grouped、Split-K 和 Stream-K

建议顺序：

1. Strided Batched GEMM。
2. Grouped GEMM，重点观察不同 problem shape 的调度问题。
3. Split-K serial，再实现或调用 parallel reduction。
4. 阅读 Ampere Stream-K example，比较它和 Split-K 的工作分配。

重点实验 small M/N、large K 的形状。记录 Split-K slice 数增加以后并行度提升与
workspace、额外 reduction、atomic 开销之间的平衡。

参考目录：

- `third_party/cutlass/examples/05_batched_gemm`
- `third_party/cutlass/examples/06_splitK_gemm`
- `third_party/cutlass/examples/24_gemm_grouped`
- `third_party/cutlass/examples/47_ampere_gemm_universal_streamk`

## 8. 阶段六：带 reduction 的后处理

不要一开始直接做完整 Softmax。按以下顺序推进：

1. GEMM 输出同时计算每个 CTA tile 的 partial row sum。
2. 把 partial sums 写入独立 workspace。
3. 编写 final reduction kernel，得到完整 row sum。
4. 扩展为 row max 或 square sum。
5. 最后选择 GEMM+RMSNorm 或 GEMM+Softmax 作为综合任务。

需要明确区分：

- CTA 内 reduction：可以使用 warp shuffle/shared memory，在当前 kernel 完成。
- 跨 CTA reduction：需要 atomic、workspace+第二个 kernel，或专门的 persistent 方案。
- Softmax/LayerNorm：reduction 结果还要反向作用于所有输出，普通 epilogue 通常无法
  在无 grid synchronization 的条件下直接完成整个流程。

参考目录：

- `third_party/cutlass/examples/23_ampere_gemm_operand_reduction_fusion`
- `third_party/cutlass/examples/35_gemm_softmax`
- `third_party/cutlass/examples/37_gemm_layernorm_gemm_fusion`

## 9. 阶段七：扩展 dtype

建议先完成 FP32/TF32 项目闭环，再依次增加：

1. FP16 input + FP32 accumulator。
2. BF16 input + FP32 accumulator。
3. INT8 input + INT32 accumulator + output scaling。
4. 可选的 2:4 structured sparse GEMM。

每增加一种 dtype，需要重新确定 instruction shape、alignment、epilogue compute type、
数值容限和候选 tile，不能只替换 `ElementA` 类型。

## 10. 测试矩阵

正确性测试至少包含：

- 所有支持的 dtype 和 activation。
- 有/无 bias，alpha/beta 的关键取值。
- M/N/K 小于 tile、等于 tile、非 tile 整数倍。
- 对齐路径和 fallback 路径。
- 非默认 CUDA stream。
- 多 GPU环境中的 device guard。
- 空 tensor 或非法 shape 的明确行为。
- Split-K、Grouped/Batched 的边界条件。

对误差同时报告最大绝对误差和最大相对误差。涉及 TF32、FP16、GELU、不同 reduction
顺序时，不要要求逐 bit 相同。

## 11. 性能实验规范

- 与 PyTorch、cuBLAS 和 `cutlass_profiler` 对比。
- 记录 GPU、driver、CUDA、PyTorch、CUTLASS commit 和 clock 状态。
- warmup 后使用 CUDA Event 测量，报告 mean、median 和 minimum。

现有 FP32 kernel 可以在相同 shape 下分别强制运行：

```bash
python benchmarks/bench_adaptive_gemm.py \
  --kernel all --m 1024 --n 1024 --k 1024
```

`all` 会依次测量 PyTorch、四个显式 kernel 和自动 dispatch：

| kernel 参数 | Warp tile | A/B alignment（元素） | Epilogue 每次处理元素数 |
| --- | --- | --- | --- |
| `fast` | 64×64×16 | 4 | 4 |
| `fallback` | 64×64×16 | 1 | 1 |
| `fast_mwarp` | 64×32×16 | 4 | 4 |
| `fallback_mwarp` | 64×32×16 | 4 | 1 |

四个 kernel 的 threadblock tile 都是 128×128×16，pipeline 都是 4 stages。
较小 warp tile 对应每个 CTA 从 4 个 warp 增加到 8 个 warp。
`fallback_mwarp` 当前仍使用向量化 A/B 访问，因此与 `fast`、`fast_mwarp` 一样要求
连续输入的 `N` 和 `K` 都是 4 的倍数；benchmark 会在运行前检查这个条件。
它与旧 `fallback` 的差异也包含 A/B alignment，比较时需要考虑这一点。

修改 C++/CUDA 后先重新编译扩展，再在新 Python 进程中运行：

```bash
python setup.py build_ext --inplace --force
python benchmarks/bench_adaptive_gemm.py --kernel fast_mwarp --m 4096 --n 4096 --k 4096
python benchmarks/bench_adaptive_gemm.py --kernel fallback_mwarp --m 4096 --n 4096 --k 4096
```

Python 中直接指定 `adaptive_gemm(a, b, kernel="fast_mwarp")` 或
`adaptive_gemm(a, b, kernel="fallback_mwarp")` 即可。显式指定时仅尝试选中的 kernel，
不兼容就报错，不会静默切换到其他实现。`auto` 按注册表顺序尝试，仍优先运行旧版
`fast` / `fallback`，不会根据性能自动选择较小 warp tile。

结果默认保存到
`results/raw/`，使用 `--no-save` 可只打印结果。

当前 benchmark 测量的是包含输出分配和 dispatch 的算子调用，CUDA Event 计时在
小 shape 下也可能受到主机提交间隔影响；需要具体 kernel 的指标时使用下方的 NCU 脚本。

- 避免把 tensor allocation 和 autotuning 混入 kernel latency；必要时提供 out variant。
- 报告 TFLOPS、峰值比例，以及 fused workload 的端到端 latency。
- 不只报告最有利的 shape，同时保留失败或退化结果。

Nsight Compute 至少关注：

- Tensor Core 指令和利用率。
- DRAM/L2/shared-memory throughput。
- global load/store efficiency。
- registers/thread 和 occupancy。
- shared-memory bank conflict。
- eligible/active warps 和主要 stall reason。

使用 Nsight Compute 收集完整指标并生成 `.ncu-rep` 报告：

```bash
bash profiling/profile_adaptive_gemm.sh \
  fast 4096 4096 4096 profiling/adaptive_gemm_fast_4096.ncu-rep
```

脚本只 profile `adaptive_gemm_profile` NVTX 范围内的一次目标 kernel，warmup 和输入
初始化不会出现在报告里。第一个参数也可以使用 `fallback`、`fast_mwarp`、
`fallback_mwarp` 或 `auto`。

## 12. 建议阅读顺序

1. `media/docs/cpp/efficient_gemm.md`
2. `examples/14_ampere_tf32_tensorop_gemm`
3. `include/cutlass/gemm/device/gemm.h`
4. `include/cutlass/gemm/kernel/default_gemm.h`
5. SM80 `threadblock::MmaMultistage` 和 warp-level MMA 实现
6. `examples/12_gemm_bias_relu`
7. examples 05、06、24、47
8. examples 23、35、37
9. CuTe layout、TiledCopy 和 TiledMMA 教程
10. 最后阅读 SM90 CollectiveBuilder/EVT，用于理解现代 CUTLASS 分层，不作为 A100
    项目的首要实现目标

## 13. 最终项目验收

完成时，你应该能够现场回答：

- 为什么选择当前 CTA/warp/instruction tile？
- stage 增加以后为什么可能变快或变慢？
- alignment 为什么影响可实现性和性能？
- small-M、small-N、large-K 分别应如何调整调度？
- epilogue fusion 实际减少了哪些 global-memory traffic？
- 哪些 reduction 能在 CTA 内完成，哪些必须跨 kernel？
- Split-K 和 Stream-K 各解决什么问题？
- 为什么你的 kernel 在某些 shape 上不如 cuBLAS？
- 如何把当前 SM80 设计迁移到 CUTLASS 3.x 的 mainloop/epilogue collective？

只有当实现、正确性、benchmark、profiling 和解释五部分都具备时，这个阶段才算完成。
