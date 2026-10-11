"""CPU checks for identical-input benchmarking and layout-preserving dispatch."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bench():
    return load_module("bench_prefetch", ROOT / "benchmarks/bench_gemm_prefetch_smem.py")


@pytest.mark.parametrize("impl", ["prefetch_smem", "prefetch_reg"])
def test_benchmark_uses_same_storage_and_validates_before_timing(bench, monkeypatch, impl):
    a, b = bench.make_inputs(128, 256, 24, device="cpu")
    assert a.shape == (128, 24) and a.stride() == (1, 128)
    assert b.is_contiguous()
    calls = []
    timed = []

    def fake_gemm(actual_a, actual_b, *, impl):
        assert actual_a is a and actual_b is b
        calls.append(impl)
        return a @ b

    def fake_timer(run, *, warmup, repeat):
        assert calls[:2] == [impl, "cublas"]
        assert (warmup, repeat) == (2, 3)
        run()
        timed.append(calls[-1])
        return {"mean_ms": 1.0 if calls[-1] == impl else 2.0}

    monkeypatch.setattr(bench, "gemm", fake_gemm)
    monkeypatch.setattr(bench, "cuda_event_benchmark", fake_timer)
    result = bench.benchmark_pair(a, b, impl=impl, warmup=2, repeat=3)
    assert timed == [impl, "cublas"]
    assert result["prefetch_speedup_over_cublas"] == 2.0
    assert result["results"][impl]["tflops"] == pytest.approx(2 * 128 * 256 * 24 / 1e9)


@pytest.mark.parametrize("impl", ["prefetch_smem", "prefetch_reg"])
def test_wrong_result_prevents_timing(bench, monkeypatch, impl):
    a, b = torch.ones(4, 4), torch.ones(4, 4)
    monkeypatch.setattr(bench, "gemm", lambda *args, **kwargs: torch.zeros(4, 4))
    def must_not_time(*args, **kwargs):
        pytest.fail("incorrect results must not be benchmarked")
    monkeypatch.setattr(bench, "cuda_event_benchmark", must_not_time)
    with pytest.raises(AssertionError):
        bench.benchmark_pair(a, b, impl=impl)


@pytest.mark.parametrize("args,message", [
    (["--m", "127"], "multiples of 128"),
    (["--n", "129"], "multiples of 128"),
    (["--k", "7"], "multiple of 8"),
    (["--k", "0"], "positive"),
    (["--repeat", "0"], "repeat must be positive"),
    (["--warmup", "-1"], "warmup must be nonnegative"),
])
def test_bad_benchmark_arguments(bench, monkeypatch, capsys, args, message):
    monkeypatch.setattr(sys, "argv", ["bench", *args])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("impl", ["prefetch_smem", "prefetch_reg", "split_k", "cublas"])
def test_python_dispatch_preserves_column_major_input(monkeypatch, impl):
    import cuda_ai_kernels
    calls = []
    extension = types.ModuleType("cuda_ai_kernels._C")
    def fake_gemm(a, b):
        calls.append((a, b))
        return a @ b
    # Load the dispatch module against a stub, without a CUDA extension on CPU.
    from cuda_ai_kernels.gemm_stages import GEMM_STAGE_IMPLS
    for name in tuple("gemm_" + name for name in GEMM_STAGE_IMPLS) + ("gemm_cublas", "gemm_naive", "gemm_block_tile", "gemm_thread_tile",
                 "gemm_warp_tile", "gemm_prefetch_smem", "gemm_prefetch_reg", "gemm_split_k", "gemm_tensor_core", "gemm_cutlass",
                 "rmsnorm_naive", "rmsnorm_cache_x", "softmax_naive"):
        setattr(extension, name, fake_gemm)
    monkeypatch.setitem(sys.modules, "cuda_ai_kernels._C", extension)
    monkeypatch.setattr(cuda_ai_kernels, "_C", extension, raising=False)
    ops = load_module("prefetch_dispatch_test", ROOT / "python/cuda_ai_kernels/ops.py")
    a, b = torch.randn(32, 128).t(), torch.randn(32, 256)
    ops.gemm(a, b, impl=impl)
    assert calls[0][0] is a and calls[0][1] is b
