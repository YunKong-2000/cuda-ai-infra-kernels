import argparse

import torch

from cuda_ai_kernels import softmax
from cuda_ai_kernels.benchmark import cuda_event_benchmark, now_tag, save_json
from cuda_ai_kernels.env import collect_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.float32)

    def run():
        if args.impl == "torch":
            return torch.softmax(x, dim=-1)
        return softmax(x)

    stats = cuda_event_benchmark(run, warmup=args.warmup, repeat=args.repeat)
    bytes_moved = args.rows * args.cols * 3 * 4
    bandwidth_gbs = bytes_moved / (stats["mean_ms"] * 1e-3) / 1e9
    payload = {
        "kernel": "softmax",
        "impl": args.impl,
        "shape": {"rows": args.rows, "cols": args.cols},
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
        save_json(f"results/raw/softmax_{args.impl}_{now_tag()}.json", payload)


if __name__ == "__main__":
    main()

