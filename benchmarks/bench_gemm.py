import argparse

import torch

from cuda_ai_kernels import gemm
from cuda_ai_kernels.benchmark import cuda_event_benchmark, now_tag, save_json
from cuda_ai_kernels.env import collect_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["torch", "naive", "tiled", "thread_tile", "warp_tile"], default="naive")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float32)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float32)

    def run():
        if args.impl == "torch":
            return torch.matmul(a, b)
        return gemm(a, b, impl=args.impl)

    stats = cuda_event_benchmark(run, warmup=args.warmup, repeat=args.repeat)
    tflops = 2.0 * args.m * args.n * args.k / (stats["mean_ms"] * 1e-3) / 1e12
    payload = {
        "kernel": "gemm",
        "impl": args.impl,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "dtype": "fp32",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "tflops": tflops,
        "stats": stats,
        "env": collect_env(),
    }

    print(payload)
    if args.save:
        save_json(args.save, payload)
    elif not args.profile:
        save_json(f"results/raw/gemm_{args.impl}_{now_tag()}.json", payload)


if __name__ == "__main__":
    main()

