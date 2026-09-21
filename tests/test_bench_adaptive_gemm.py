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
