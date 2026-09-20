import argparse
from collections.abc import Callable
from typing import Any

import torch

from cuda_ai_kernels import adaptive_gemm
from cuda_ai_kernels.benchmark import cuda_event_benchmark, now_tag, save_json
from cuda_ai_kernels.env import collect_env


KERNELS = ("fast", "fallback", "fast_mwarp", "fallback_mwarp")
# Both MWarp variants currently use vectorized A/B loads (alignment=4).
ALIGNED_KERNELS = ("fast", "fast_mwarp", "fallback_mwarp")


def make_runner(
    kernel: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    if kernel == "torch":
        return lambda: torch.matmul(a, b)
    return lambda: adaptive_gemm(a, b, kernel=kernel)


def check_correctness(
    run: Callable[[], torch.Tensor],
    expected: torch.Tensor,
) -> dict[str, float]:
    actual = run()
    torch.cuda.synchronize()
    absolute_error = (actual - expected).abs()
    relative_error = absolute_error / expected.abs().clamp_min(1e-7)
    max_absolute_error = absolute_error.max().item()
    max_relative_error = relative_error.max().item()

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    return {
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }


def benchmark_kernel(
    kernel: str,
    a: torch.Tensor,
    b: torch.Tensor,
    expected: torch.Tensor,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    run = make_runner(kernel, a, b)
    correctness = check_correctness(run, expected)
    stats = cuda_event_benchmark(run, warmup=warmup, repeat=repeat)
    operations = 2.0 * a.size(0) * b.size(1) * a.size(1)

    return {
        "kernel": kernel,
        "tflops_mean": operations / (stats["mean_ms"] * 1e-3) / 1e12,
        "tflops_min_latency": operations / (stats["min_ms"] * 1e-3) / 1e12,
        "stats": stats,
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the FP32 adaptive GEMM kernels with CUDA Events."
    )
    parser.add_argument(
        "--kernel",
        choices=[*KERNELS, "auto", "torch", "all"],
        default="all",
        help="'all' benchmarks torch, all four compiled kernels, and automatic dispatch.",
    )
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
    if args.kernel in (*ALIGNED_KERNELS, "all") and (args.k % 4 != 0 or args.n % 4 != 0):
        parser.error("fast, fast_mwarp, and fallback_mwarp require both K and N to be divisible by 4")
    if args.profile and args.kernel == "all":
        parser.error("--profile requires one kernel, not --kernel all")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float32)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float32)

    if args.profile:
        run = make_runner(args.kernel, a, b)
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

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()

    kernels = ["torch", *KERNELS, "auto"] if args.kernel == "all" else [args.kernel]
    results = [
        benchmark_kernel(kernel, a, b, expected, args.warmup, args.repeat)
        for kernel in kernels
    ]
    payload = {
        "benchmark": "adaptive_gemm",
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "dtype": "fp32_tf32",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": results,
        "env": collect_env(),
    }

    for result in results:
        stats = result["stats"]
        print(
            f"{result['kernel']:>14}: "
            f"mean={stats['mean_ms']:.4f} ms, "
            f"p50={stats['p50_ms']:.4f} ms, "
            f"min={stats['min_ms']:.4f} ms, "
            f"TFLOPS={result['tflops_mean']:.2f}"
        )

    if args.no_save:
        return
    output_path = args.save or f"results/raw/adaptive_gemm_{args.kernel}_{now_tag()}.json"
    save_json(output_path, payload)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
