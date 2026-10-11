"""Validate and benchmark the entire column-major-A FP32 GEMM ladder."""

import argparse
import statistics

import torch

from cuda_ai_kernels import gemm
from cuda_ai_kernels.benchmark import cuda_event_benchmark, now_tag, save_json
from cuda_ai_kernels.env import collect_env
from cuda_ai_kernels.gemm_stages import GEMM_STAGES


# M,N,K order; matches the seven columns in the reference speedup heatmap.
DEFAULT_SHAPES = (
    "256,256,256",
    "1024,1024,1024",
    "2048,2048,2048",
    "4096,4096,2048",
    "4096,1024,4096",
    "1024,4096,4096",
    "4096,4096,4096",
)


def parse_shape(value):
    try:
        m, n, k = (int(x) for x in value.lower().replace("x", ",").split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be M,N,K or MxNxK") from error
    if min(m, n, k) <= 0 or m % 128 or n % 128 or k % 16:
        raise argparse.ArgumentTypeError(
            "all-stage shapes require positive M/N multiples of 128 and K multiples of 16")
    # Naive / scalar stages use BM=16.
    if m // 16 > 65535:
        raise argparse.ArgumentTypeError("M/16 must not exceed grid.y limit 65535")
    return m, n, k


def make_inputs(m, n, k, device="cuda"):
    a = torch.randn((k, m), device=device, dtype=torch.float32).t()
    b = torch.randn((k, n), device=device, dtype=torch.float32)
    return a, b


def cuda_graph_benchmark(run, *, warmup, repeat, batch=10):
    # Capture ten API calls to amortize Python launch gaps. GPU operations
    # (including Split-K's output zeroing) remain in the captured graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup)):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(batch):
            output = run()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / batch)
    # Keep the final captured tensor alive until every replay completes.
    del output
    return {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_shape(a, b, *, warmup=10, repeat=50, timing="graph"):
    # Independent FP64 reference; TF32 flags cannot weaken validation.
    expected = (a.double() @ b.double()).float()
    results = []
    runners = {}
    # Every variant sees exactly the same tensors and must pass before timing.
    for stage, name, tile in GEMM_STAGES:
        runners[name] = lambda name=name: gemm(a, b, impl=name)
        actual = runners[name]()
        if not actual.is_contiguous():
            raise AssertionError(f"{name}: C must be row-major contiguous")
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3,
                                   msg=lambda msg: f"{name}: {msg}")
        results.append({
            "stage": stage, "impl": name, "tile_multiples": tile,
            "correct": True,
            "max_absolute_error": (actual - expected).abs().max().item(),
        })
    m, k = a.shape
    n = b.size(1)
    timer = cuda_graph_benchmark if timing == "graph" else cuda_event_benchmark
    for row in results:
        stats = timer(runners[row["impl"]], warmup=warmup, repeat=repeat)
        row.update(stats=stats, tflops=2.0 * m * n * k / (stats["mean_ms"] * 1e9))
    cublas_ms = results[-1]["stats"]["mean_ms"]
    naive_ms = results[0]["stats"]["mean_ms"]
    for row in results:
        row["speedup_over_cublas"] = cublas_ms / row["stats"]["mean_ms"]
        row["speedup_over_naive"] = naive_ms / row["stats"]["mean_ms"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=parse_shape,
                        default=[parse_shape(s) for s in DEFAULT_SHAPES], metavar="M,N,K")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--timing", choices=("graph", "event"), default="graph")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be nonnegative and repeat must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required (SM80+ for the existing prefetch kernels)")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] < 8:
        parser.error("the existing prefetch kernels require SM80 or newer")
    torch.manual_seed(args.seed)
    payload = {
        "kernel": "gemm_optimization_ladder", "dtype": "fp32",
        "layout": {"a": "column_major", "b": "row_major", "c": "row_major"},
        "cublas_compute": "CUBLAS_COMPUTE_32F_PEDANTIC",
        "reference": "torch_fp64_cast_to_fp32",
        "timing": args.timing,
        "timing_note": (
            "CUDA events per replay / 10 captured calls; includes Split-K zeroing; "
            "excludes CPU allocation/validation and input preparation"
            if args.timing == "graph" else
            "CUDA events around allocating API calls; includes host launch gaps and Split-K zeroing; "
            "excludes input preparation and correctness validation"
        ),
        "warmup": args.warmup, "repeat": args.repeat,
        "seed": args.seed, "device": args.device, "env": collect_env(), "shapes": [],
    }
    for m, n, k in args.shapes:
        print(f"\nM={m} N={n} K={k} | A column-major, B/C row-major | {args.timing}", flush=True)
        a, b = make_inputs(m, n, k)
        rows = benchmark_shape(a, b, warmup=args.warmup, repeat=args.repeat, timing=args.timing)
        print(f"{'stage':>5}  {'implementation':<28} {'ms':>10} {'TFLOPS':>10} {'vs cuBLAS':>10} {'vs naive':>10}  check")
        for row in rows:
            print(f"{row['stage']:>5}  {row['impl']:<28} {row['stats']['mean_ms']:>10.4f} "
                  f"{row['tflops']:>10.3f} {row['speedup_over_cublas']:>9.3f}x "
                  f"{row['speedup_over_naive']:>9.3f}x  PASS", flush=True)
        payload["shapes"].append({
            "m": m, "n": n, "k": k, "a_stride": list(a.stride()),
            "b_stride": list(b.stride()), "results": rows,
        })
    path = args.save or f"results/raw/gemm_all_{now_tag()}.json"
    save_json(path, payload)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
