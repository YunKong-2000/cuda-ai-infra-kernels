# RMSNorm 实现与测量

当前提供 FP32 `naive` 和 `cache_x` 前向路径，通过 [Python API](../python/cuda_ai_kernels/ops.py) 调用。源文件位于 [csrc/rmsnorm](../csrc/rmsnorm/)，数值对照位于 [test_rmsnorm.py](../tests/test_rmsnorm.py)。

计算为 `y = x * rsqrt(mean(x²) + eps) * weight`。两个实现用于比较输入重复读取与缓存策略，涉及 warp/block 归约及寄存器/共享内存占用的取舍。

```bash
python -m pytest -q tests/test_rmsnorm.py
python benchmarks/bench_rmsnorm.py --impl naive --rows 4096 --hidden 4096
python benchmarks/bench_rmsnorm.py --impl cache_x --rows 4096 --hidden 4096
```

本仓库尚未归档该算子的完整性能结果，不声明相对 PyTorch 的加速比。比较需固定 shape、精度、epsilon、基线和计时范围；有效带宽使用明确的读写字节估算，不与实际 HBM 流量混同。
