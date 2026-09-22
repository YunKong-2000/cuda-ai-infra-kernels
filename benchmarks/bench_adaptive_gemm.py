from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import torch

from cuda_ai_kernels import adaptive_gemm
from cuda_ai_kernels.benchmark import cuda_event_benchmark, now_tag, save_json
from cuda_ai_kernels.env import collect_env


KERNELS = (
    "fast",
    "fallback",
    "fast_mwarp",
    "fallback_mwarp",
    "fast_stage3",
    "fast_stage2",
)
RELU_KERNELS = ("fast_relu", "fallback_relu")
UNFUSED_RELU_KERNELS = ("fast+relu", "fallback+relu")
CUBLAS_BASELINE = "cublas"
# Only the fast variants require vectorized A/B loads (alignment=4).
ALIGNED_KERNELS = ("fast", "fast_mwarp", "fast_stage3", "fast_stage2", "fast_relu", "fast+relu")


def make_runner(
    kernel: str,
    a: torch.Tensor,
    b: torch.Tensor,
    epilogue: str = "linear",
    c: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Callable[[], torch.Tensor]:
    if kernel == "torch":
        def run_torch() -> torch.Tensor:
            result = torch.matmul(a, b)
            if alpha != 1.0:
                result = alpha * result
            if beta != 0.0:
                result = result + beta * c
            return torch.relu(result) if epilogue == "relu" else result
        return run_torch
    if epilogue == "relu" and kernel in (CUBLAS_BASELINE, *UNFUSED_RELU_KERNELS):
        linear_kernel = kernel.removesuffix("+relu")
        return lambda: torch.relu(adaptive_gemm(
            a, b, c=c, alpha=alpha, beta=beta, epilogue="linear", kernel=linear_kernel,
        ))
    return lambda: adaptive_gemm(
        a, b, c=c, alpha=alpha, beta=beta, epilogue=epilogue, kernel=kernel,
    )


def make_references(
    run_torch: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use TF32-allowed PyTorch for validation and full FP32 for diagnostics."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        expected = run_torch()
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = False
        fp32_expected = run_torch()
        torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return expected, fp32_expected


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    absolute_error = (actual - expected).abs()
    relative_error = absolute_error / expected.abs().clamp_min(1e-7)
    return {
        "max_absolute_error": absolute_error.max().item(),
        "max_relative_error": relative_error.max().item(),
    }


def check_correctness(
    run: Callable[[], torch.Tensor],
    expected: torch.Tensor,
    fp32_expected: torch.Tensor,
) -> dict[str, Any]:
    actual = run()
    torch.cuda.synchronize()

    # Keep validation against the performance baseline. FP32 error is reported
    # separately and does not gate TF32 performance measurements.
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    return {
        "reference": "torch_tf32_allowed",
        "atol": 3e-2,
        "rtol": 3e-2,
        **error_metrics(actual, expected),
        "vs_fp32": error_metrics(actual, fp32_expected),
    }


def benchmark_kernel(
    kernel: str,
    a: torch.Tensor,
    b: torch.Tensor,
    expected: torch.Tensor,
    fp32_expected: torch.Tensor,
    warmup: int,
    repeat: int,
    epilogue: str = "linear",
    c: torch.Tensor | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> dict[str, Any]:
    run = make_runner(kernel, a, b, epilogue, c, alpha, beta)
    correctness = check_correctness(run, expected, fp32_expected)
    stats = cuda_event_benchmark(run, warmup=warmup, repeat=repeat)
    operations = 2.0 * a.size(0) * b.size(1) * a.size(1)

    return {
        "kernel": kernel,
        "epilogue": epilogue,
        "fused_relu": epilogue == "relu" and kernel in (*RELU_KERNELS, "auto"),
        "tflops_mean": operations / (stats["mean_ms"] * 1e-3) / 1e12,
        "tflops_min_latency": operations / (stats["min_ms"] * 1e-3) / 1e12,
        "stats": stats,
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FP32/TF32 adaptive GEMM using CUDA Events. Validate against "
            "TF32-allowed PyTorch and report full FP32 error separately."
        )
    )
    parser.add_argument(
        "--kernel",
        choices=[*KERNELS, *RELU_KERNELS, *UNFUSED_RELU_KERNELS, CUBLAS_BASELINE, "auto", "torch", "all"],
        default="all",
        help=(
            "'all' benchmarks compatible kernels for --epilogue, torch, cuBLAS, "
            "and auto. ReLU includes unfused cuBLAS/CUTLASS + torch.relu baselines."
        ),
    )
    parser.add_argument("--epilogue", choices=["linear", "relu"], default="linear")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Warm up, then run one kernel inside the adaptive_gemm_profile NVTX range.",
    )
    args = parser.parse_args()

    if min(args.m, args.n, args.k) <= 0:
        parser.error("--m, --n, and --k must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    if args.epilogue == "linear" and args.kernel in (*RELU_KERNELS, *UNFUSED_RELU_KERNELS):
        parser.error("ReLU kernels require --epilogue relu")
    if args.epilogue == "relu" and args.kernel in KERNELS:
        parser.error("use fast_relu/fallback_relu for fusion or fast+relu/fallback+relu for an unfused baseline")
    if args.kernel in ALIGNED_KERNELS and (args.k % 4 != 0 or args.n % 4 != 0):
        parser.error("fast kernels require both K and N to be divisible by 4")
    if args.profile and args.kernel == "all":
        parser.error("--profile requires one kernel, not --kernel all")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float32)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float32)
    c = torch.randn((args.m, args.n), device="cuda", dtype=torch.float32) if args.beta != 0 else None

    # Both references are generated outside timing/profiling; timed torch keeps
    # TF32 enabled. Allowing TF32 does not force a particular PyTorch algorithm.
    expected, fp32_expected = make_references(
        make_runner("torch", a, b, args.epilogue, c, args.alpha, args.beta)
    )

    if args.profile:
        run = make_runner(args.kernel, a, b, args.epilogue, c, args.alpha, args.beta)
        correctness = check_correctness(run, expected, fp32_expected)
        print(f"correctness (torch TF32 allowed): {correctness}")
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()

        torch.cuda.nvtx.range_push("adaptive_gemm_profile")
        try:
            run()
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return

    candidates = KERNELS if args.epilogue == "linear" else (*RELU_KERNELS, *UNFUSED_RELU_KERNELS)
    kernels = (
        ["torch", CUBLAS_BASELINE, *candidates, "auto"]
        if args.kernel == "all"
        else [args.kernel]
    )
    skipped = []
    if args.kernel == "all" and (args.k % 4 != 0 or args.n % 4 != 0):
        skipped = [kernel for kernel in kernels if kernel in ALIGNED_KERNELS]
        kernels = [kernel for kernel in kernels if kernel not in skipped]
        print(f"skipping unaligned kernels: {', '.join(skipped)}")
    results = [
        benchmark_kernel(kernel, a, b, expected, fp32_expected, args.warmup, args.repeat,
                         args.epilogue, c, args.alpha, args.beta)
        for kernel in kernels
    ]
    cublas_result = next(
        (result for result in results if result["kernel"] == CUBLAS_BASELINE),
        None,
    )
    if cublas_result is not None:
        cublas_mean_ms = cublas_result["stats"]["mean_ms"]
        for result in results:
            result["speedup_vs_cublas"] = cublas_mean_ms / result["stats"]["mean_ms"]
    payload = {
        "benchmark": "adaptive_gemm",
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "dtype": "fp32_tf32",
        "epilogue": args.epilogue,
        "alpha": args.alpha,
        "beta": args.beta,
        "skipped_kernels": skipped,
        "cublas_baseline": "cublas+torch.relu" if args.epilogue == "relu" else "cublas",
        "tflops_basis": "2*M*N*K (GEMM FLOPs / full operation latency)",
        "cublas_compute_type": "CUBLAS_COMPUTE_32F_FAST_TF32",
        "torch_allow_tf32": True,
        "correctness_reference": "torch_tf32_allowed",
        "precision_reference": "torch_fp32_tf32_disabled",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": results,
        "env": collect_env(),
    }

    for result in results:
        stats = result["stats"]
        correctness = result["correctness"]
        speedup = result.get("speedup_vs_cublas")
        baseline = "cublas+relu" if args.epilogue == "relu" else "cublas"
        speedup_text = f", vs_{baseline}={speedup:.3f}x" if speedup is not None else ""
        label = result["kernel"]
        if args.epilogue == "relu" and label in ("torch", "cublas"):
            label += "+relu"
        print(
            f"{label:>14}: "
            f"mean={stats['mean_ms']:.4f} ms, "
            f"p50={stats['p50_ms']:.4f} ms, "
            f"min={stats['min_ms']:.4f} ms, "
            f"TFLOPS={result['tflops_mean']:.2f}"
            f"{speedup_text}"
            f", max_abs_vs_torch_tf32_allowed={correctness['max_absolute_error']:.6g}"
            f", max_abs_vs_fp32={correctness['vs_fp32']['max_absolute_error']:.6g}"
        )

    if args.no_save:
        return
    output_path = args.save or f"results/raw/adaptive_gemm_{args.epilogue}_{args.kernel}_{now_tag()}.json"
    save_json(output_path, payload)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
