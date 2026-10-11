"""Compare a prefetch GEMM and strict-FP32 cuBLAS on identical A/B storage."""

import argparse
import json

import torch

from cuda_ai_kernels import gemm
from cuda_ai_kernels.benchmark import cuda_event_benchmark, now_tag, save_json
from cuda_ai_kernels.env import collect_env


PREFETCH_IMPLEMENTATIONS = ("prefetch_smem", "prefetch_reg")


def make_inputs(m, n, k, device="cuda"):
    # Transpose is a view: A is logical [M,K], physically column-major.
    a = torch.randn((k, m), device=device, dtype=torch.float32).t()
    b = torch.randn((k, n), device=device, dtype=torch.float32)
    return a, b


def make_runner(impl, a, b):
    return lambda: gemm(a, b, impl=impl)


def benchmark_pair(a, b, *, impl="prefetch_smem", warmup=50, repeat=200):
    if impl not in PREFETCH_IMPLEMENTATIONS:
        raise ValueError(f"unsupported prefetch implementation: {impl}")
    # Validate both implementations before measuring either. FP64 provides an
    # independent reference and does not depend on PyTorch's TF32 setting.
    expected = (a.double() @ b.double()).float()
    runners = {name: make_runner(name, a, b) for name in (impl, "cublas")}
    results = {}
    for name, run in runners.items():
        actual = run()
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
        results[name] = {
            "max_absolute_error": (actual - expected).abs().max().item(),
            "correct": True,
        }
    m, k = a.shape
    n = b.size(1)
    for name, run in runners.items():
        stats = cuda_event_benchmark(run, warmup=warmup, repeat=repeat)
        results[name].update(
            stats=stats,
            tflops=2.0 * m * n * k / (stats["mean_ms"] * 1e9),
        )
    return {
        "results": results,
        # >1 means the selected prefetch implementation is faster; no performance threshold is asserted.
        "prefetch_speedup_over_cublas": (
            results["cublas"]["stats"]["mean_ms"]
            / results[impl]["stats"]["mean_ms"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=PREFETCH_IMPLEMENTATIONS, default="prefetch_smem")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0 or args.m % 128 or args.n % 128 or args.k % 8:
        parser.error("M/N must be positive multiples of 128; K must be a positive multiple of 8")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be nonnegative and repeat must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    a, b = make_inputs(args.m, args.n, args.k)
    payload = {
        "kernel": f"gemm_{args.impl}_vs_cublas",
        "impl": args.impl,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "layout": {"a": "column_major", "b": "row_major", "c": "row_major"},
        "a_stride": list(a.stride()),
        "b_stride": list(b.stride()),
        "dtype": "fp32",
        "cublas_compute": "CUBLAS_COMPUTE_32F_PEDANTIC",
        "reference": "torch_fp64_cast_to_fp32",
        "timing": "CUDA events around allocating API calls; excludes input preparation and validation",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "device": args.device,
        "env": collect_env(),
        **benchmark_pair(a, b, impl=args.impl, warmup=args.warmup, repeat=args.repeat),
    }
    print(json.dumps(payload, indent=2))
    save_json(args.save or f"results/raw/gemm_{args.impl}_vs_cublas_{now_tag()}.json", payload)


if __name__ == "__main__":
    main()
