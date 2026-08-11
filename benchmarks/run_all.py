import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["gemm", "rmsnorm", "softmax"], required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    shapes_path = ROOT / "benchmarks" / "shapes" / f"{args.kernel}.yaml"
    shapes = yaml.safe_load(shapes_path.read_text(encoding="utf-8"))["shapes"]

    for shape in shapes:
        if args.kernel == "gemm":
            cmd = [
                sys.executable,
                "benchmarks/bench_gemm.py",
                "--impl",
                args.impl,
                "--m",
                str(shape["m"]),
                "--n",
                str(shape["n"]),
                "--k",
                str(shape["k"]),
            ]
        elif args.kernel == "rmsnorm":
            cmd = [
                sys.executable,
                "benchmarks/bench_rmsnorm.py",
                "--impl",
                args.impl,
                "--rows",
                str(shape["rows"]),
                "--hidden",
                str(shape["hidden"]),
            ]
        else:
            cmd = [
                sys.executable,
                "benchmarks/bench_softmax.py",
                "--impl",
                args.impl,
                "--rows",
                str(shape["rows"]),
                "--cols",
                str(shape["cols"]),
            ]

        cmd += ["--warmup", str(args.warmup), "--repeat", str(args.repeat)]
        run(cmd)


if __name__ == "__main__":
    main()

