# 性能测量与 Nsight Compute

## 测量边界

正确性检查和 JIT/扩展构建在计时外完成。记录 GPU、驱动、CUDA/PyTorch/cuBLAS 版本、commit、shape、dtype、布局、预热、重复次数与原始样本。

- CUDA Events 围绕普通 API 调用时，区间可能包含主机提交间隙及设备清零/复制。
- CUDA Graph 固定输入与缓冲区，在捕获阶段完成主机工作，重放仍包含实际 GPU 调度与数据操作。
- `split_k` 依赖清零后原子累加，清零属于执行成本，不能从基准中省略。
- 手写 FP32 SIMT 的 PEDANTIC cuBLAS 对照与 adaptive GEMM 的 TF32 对照分开报告。
- 当前全阶段脚本使用 mean 生成吞吐；归档 A800 CSV 使用 median，详见 [结果说明](gemm_optimization_log.md)。

## Profiling 入口

```bash
bash profiling/profile_gemm.sh split_k 4096 4096 4096
bash profiling/profile_adaptive_gemm.sh fast 4096 4096 4096
```

RMSNorm / Softmax 的具体选项以对应 profiling 脚本和 benchmark 的 `--help` 为准。NCU 采集的运行时间可能受 replay/序列化影响；优化结论使用未挂 profiler 的同口径测量。

## 指标与解释

| 指标 | 关注的问题 |
| --- | --- |
| registers/thread、shared memory/CTA | tile 与预取缓冲带来的驻留限制 |
| achieved occupancy、active/eligible warps | 是否具备足够并行任务和可发射 warp |
| DRAM / L2 / shared throughput | 数据复用及搬运压力，结合实际访存量解释 |
| shared-memory bank conflict | lane→地址映射及向量访问冲突 |
| global transaction / sector 利用率 | 连续访问和线程输出布局 |
| stall reasons | 依赖与同步等待，不能直接解释为整卡空闲比例 |

先提出可测的瓶颈假设，再修改单个因素，保存正确性、资源指标与延迟。更高 occupancy、更少 atomic 或更合并的访存不必然降低总耗时。退化变体保留在同输入对照中，避免只展示最优数字。
