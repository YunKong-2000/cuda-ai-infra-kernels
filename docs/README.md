# 文档导航

- [项目首页](../README.md)：实现范围、性能摘要和安装。
- [FP32 GEMM 实现分层](gemm_optimization_ladder.md)：tile、线程布局、输入契约和统一测试。
- [A800 GEMM 性能记录](gemm_optimization_log.md)：完整形状汇总、CSV 和统计口径。
- [共享内存预取](gemm_prefetch_smem.md)：列主序 A / 行主序 B,C 接口及对照。
- [CUTLASS 多配置 GEMM](../csrc/adaptive_gemm/README.md)：注册项、对齐分派、epilogue 与 TF32 基线。
- [RMSNorm](rmsnorm_optimization_log.md) / [Softmax](softmax_optimization_log.md)：归约实现、测试与未测量范围。
- [Profiling 方法](profiling_notes.md)：Nsight Compute 入口及结果解释。
- [Docker 环境](docker.md)：构建及 GPU profiling 环境。

`docs/results/` 保存本文引用的历史数据；`results/raw/` 用于运行时生成结果，默认不纳入版本控制。文档中的历史结果与当前源码支持范围分别描述。
