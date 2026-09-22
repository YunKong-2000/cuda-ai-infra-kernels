"""CPU checks for benchmark semantics; CUDA correctness lives in test_adaptive_gemm."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_adaptive_gemm.py"
    spec = importlib.util.spec_from_file_location("bench_adaptive_gemm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kernel", ["torch", "cublas", "fast+relu", "fallback+relu", "fast_relu", "fallback_relu", "auto"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (-0.5, 0.25)])
def test_relu_runner_computes_equivalent_operation(bench, monkeypatch, kernel, alpha, beta):
    calls = []

    def fake_gemm(a, b, *, c, alpha, beta, epilogue, kernel):
        calls.append((kernel, epilogue))
        result = alpha * (a @ b)
        if beta:
            result = result + beta * c
        return result.relu() if epilogue == "relu" else result

    monkeypatch.setattr(bench, "adaptive_gemm", fake_gemm)
    a = torch.eye(2)
    b = torch.tensor([[-2.0, 2.0], [4.0, -4.0]])
    c = torch.ones_like(b) if beta else None
    expected = (alpha * b + (beta * c if c is not None else 0)).relu()
    actual = bench.make_runner(kernel, a, b, "relu", c, alpha, beta)()
    torch.testing.assert_close(actual, expected)
    if kernel == "torch":
        assert calls == []
    elif kernel in ("cublas", "fast+relu", "fallback+relu"):
        # These baselines must request linear GEMM and apply a separate ReLU.
        assert calls == [(kernel.removesuffix("+relu"), "linear")]
    else:
        assert calls == [(kernel, "relu")]


@pytest.mark.parametrize("kernel", ["torch", "cublas", "fast", "auto"])
def test_linear_runner_preserves_negative_outputs(bench, monkeypatch, kernel):
    def fake_gemm(a, b, **kwargs):
        assert kwargs["epilogue"] == "linear"
        return a @ b

    monkeypatch.setattr(bench, "adaptive_gemm", fake_gemm)
    a = torch.eye(2)
    b = -torch.ones(2, 2)
    torch.testing.assert_close(bench.make_runner(kernel, a, b)(), b)


@pytest.mark.parametrize("arguments,message", [
    (["--kernel", "fast_relu"], "require --epilogue relu"),
    (["--epilogue", "relu", "--kernel", "fast"], "use fast_relu/fallback_relu"),
    (["--epilogue", "relu", "--kernel", "fast_relu", "--n", "65"], "divisible by 4"),
    (["--epilogue", "relu", "--kernel", "fast_relu", "--k", "17"], "divisible by 4"),
    (["--profile", "--kernel", "all"], "requires one kernel"),
])
def test_benchmark_rejects_invalid_combinations(bench, monkeypatch, capsys, arguments, message):
    monkeypatch.setattr("sys.argv", ["bench_adaptive_gemm.py", *arguments])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("fail_fp32", [False, True])
def test_references_use_both_precisions_and_restore_setting(bench, monkeypatch, previous, fail_fp32):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", previous)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    modes = []

    def run_torch():
        enabled = torch.backends.cuda.matmul.allow_tf32
        modes.append(enabled)
        if not enabled and fail_fp32:
            raise RuntimeError("reference failed")
        return torch.tensor([0.1 if enabled else 0.0])

    if fail_fp32:
        with pytest.raises(RuntimeError, match="reference failed"):
            bench.make_references(run_torch)
    else:
        expected, fp32_expected = bench.make_references(run_torch)
        torch.testing.assert_close(expected, torch.tensor([0.1]))
        torch.testing.assert_close(fp32_expected, torch.tensor([0.0]))
    assert modes == [True, False]
    assert torch.backends.cuda.matmul.allow_tf32 == previous


def test_fp32_precision_loss_does_not_block_benchmark(bench, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    expected = torch.tensor([[0.1]])
    fp32_expected = torch.zeros_like(expected)
    events = []

    def run():
        events.append("run")
        return expected.clone()

    def time_runner(fn, *, warmup, repeat):
        # Validation must finish before the timed runner is invoked.
        assert events == ["run"]
        assert (warmup, repeat) == (2, 3)
        fn()
        return {"mean_ms": 1.0, "min_ms": 1.0}

    monkeypatch.setattr(bench, "make_runner", lambda *args: run)
    monkeypatch.setattr(bench, "cuda_event_benchmark", time_runner)
    result = bench.benchmark_kernel(
        "fast", expected, expected, expected, fp32_expected, warmup=2, repeat=3,
    )
    correctness = result["correctness"]
    assert correctness["reference"] == "torch_tf32_allowed"
    assert correctness["max_absolute_error"] == 0.0
    assert correctness["vs_fp32"]["max_absolute_error"] == pytest.approx(0.1)
    assert events == ["run", "run"]


@pytest.mark.parametrize("value", [0.1, float("nan"), float("inf")])
def test_tf32_reference_mismatch_still_fails(bench, monkeypatch, value):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    actual = torch.tensor([value])
    with pytest.raises(AssertionError):
        bench.check_correctness(lambda: actual, torch.zeros(1), actual)
