# cuda-ai-infra-kernels

Native CUDA kernels for AI infrastructure interview preparation and performance engineering practice.

The project is organized around a repeatable optimization loop:

```text
kernel -> correctness test -> benchmark -> profiling -> bottleneck -> optimization note
```

## Scope

Current v0.1 focus:

- Buildable PyTorch C++/CUDA extension layout
- Python API wrappers for GEMM, RMSNorm, Softmax, RoPE, and attention decode
- Correctness test templates against PyTorch baselines
- Benchmark templates with CUDA Event timing
- Profiling workflow with Nsight Compute

The FP32 GEMM optimization ladder is implemented under `csrc/gemm/`; other operators have their own implementation status.

Planned kernels:

- RoPE
- Attention decode toy kernel
- Fused bias + activation

## Layout

```text
csrc/                  C++/CUDA extension sources
python/cuda_ai_kernels Python package and benchmark helpers
tests/                 PyTest correctness tests against PyTorch
benchmarks/            Reproducible benchmark entrypoints
profiling/             Nsight Compute launch scripts
docs/                  Optimization logs and profiling notes
results/               Raw outputs, tables, and figures
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Requirements:

- NVIDIA GPU
- CUDA toolkit compatible with your PyTorch build
- PyTorch with CUDA support
- Nsight Compute for profiling scripts

For quick extension compile checks during CUDA development:

```bash
make check-skeleton
make build-ext
```

`make check-skeleton` performs static checks for Python syntax, `setup.py` source paths, project-local includes, and pybind declarations. `make build-ext` compiles the extension in place without doing a full editable pip install.

## Docker

Start a development container from a locally pulled NVIDIA image:

```bash
scripts/docker_run.sh nvcr.io/nvidia/pytorch:<tag>
```

The container is started with all GPUs, host IPC shared memory, Nsight Compute-friendly permissions, the current user's UID/GID, and this repository mounted as the working directory. More details are in [docs/docker.md](docs/docker.md).

## Correctness

```bash
pytest tests
```

CUDA tests require a CUDA-capable GPU and a rebuilt extension.

## Benchmarks

Run the complete 9-stage FP32 GEMM ladder (13 variants including cuBLAS):

```bash
bash scripts/bench_gemm_all.sh
```

This builds the extension, checks every implementation, and prints latency,
TFLOPS, and speedups against cuBLAS and naive using identical column-major A
and row-major B. See [GEMM optimization ladder](docs/gemm_optimization_ladder.md)
for tile variants, shape requirements, timing methods, and regression tests.


For the column-major A / row-major B,C prefetch kernel, see
[interface, tests and same-input cuBLAS comparison](docs/gemm_prefetch_smem.md).

The `split_k` implementation uses the same FP32 column-major A / row-major B
layout, with positive M/N divisible by 128 and K divisible by 16. It computes
two K slices per output tile and atomically sums them into a zeroed output.
Call `gemm(a, b, impl="split_k")` or benchmark with
`python benchmarks/bench_gemm.py --impl split_k --m 128 --n 256 --k 1024`.
Run its CUDA regression tests with `pytest tests/test_gemm_split_k.py` after
rebuilding the extension.

```bash
python benchmarks/bench_gemm.py --impl naive --m 1024 --n 1024 --k 1024
python benchmarks/bench_rmsnorm.py --impl cuda --rows 4096 --hidden 4096
python benchmarks/bench_softmax.py --impl cuda --rows 4096 --cols 2048
```

Benchmark scripts are ready to use after the selected kernel implementation is filled in.

Batch benchmark by shape list:

```bash
python benchmarks/bench_gemm_all.py --shapes 128,256,16 1024,1024,1024
python benchmarks/run_all.py --kernel rmsnorm --impl cuda
python benchmarks/run_all.py --kernel softmax --impl cuda
```

## Profiling

```bash
bash profiling/profile_gemm.sh thread_tile_8x8 4096 4096 4096
bash profiling/profile_adaptive_gemm.sh fast 4096 4096 4096
bash profiling/profile_rmsnorm.sh cuda 4096 4096
bash profiling/profile_softmax.sh cuda 4096 2048
```

## Reporting Checklist

For each kernel, record:

- GPU model, CUDA version, driver version, PyTorch version
- Shapes and dtype
- Warmup and repeat counts
- Latency, speedup, TFLOPS or effective bandwidth
- Nsight Compute observations
- Bottleneck and optimization rationale
- Known limitations
