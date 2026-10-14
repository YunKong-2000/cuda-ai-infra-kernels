# Softmax 实现与测量

当前提供 FP32 `naive` 前向路径。每个 CTA 处理一行，通过 warp shuffle 和共享内存完成最大值及求和归约；采用减去行最大值的稳定 softmax，包含 float4 搬运与尾部处理。

源码：[softmax.cu](../csrc/softmax/softmax.cu)；测试：[test_softmax.py](../tests/test_softmax.py)。

```bash
python -m pytest -q tests/test_softmax.py
python benchmarks/bench_softmax.py --impl naive --rows 4096 --cols 2048
```

本仓库尚未归档 Softmax 的完整性能结果，不声明相对融合库的加速比。测试覆盖应按源码支持范围解释；宽度变化涉及向量对齐、归约工作量和并发度，不能把单一形状结论推广到全部行宽。
