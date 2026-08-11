import argparse

import torch

from cuda_ai_kernels import rmsnorm
from cuda_ai_kernels.benchmark import cuda_event_benchmark, now_tag, save_json
from cuda_ai_kernels.env import collect_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn((args.rows, args.hidden), device="cuda", dtype=torch.float32)
    weight = torch.randn((args.hidden,), device="cuda", dtype=torch.float32)

    def run():
        if args.impl == "torch":
            inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + args.eps)
            return x * inv_rms * weight
        return rmsnorm(x, weight, args.eps)

    stats = cuda_event_benchmark(run, warmup=args.warmup, repeat=args.repeat)
    bytes_moved = args.rows * args.hidden * 3 * 4 + args.hidden * 4
    bandwidth_gbs = bytes_moved / (stats["mean_ms"] * 1e-3) / 1e9
    payload = {
        "kernel": "rmsnorm",
        "impl": args.impl,
        "shape": {"rows": args.rows, "hidden": args.hidden},
        "dtype": "fp32",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "bandwidth_gbs": bandwidth_gbs,
        "stats": stats,
        "env": collect_env(),
    }

    print(payload)
    if args.save:
        save_json(args.save, payload)
    elif not args.profile:
        save_json(f"results/raw/rmsnorm_{args.impl}_{now_tag()}.json", payload)


if __name__ == "__main__":
    main()

