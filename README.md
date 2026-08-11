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

Kernel implementations are intentionally left as TODO stubs. Fill in the `.cu` files under `csrc/` and then enable the corresponding tests.

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

## Correctness

```bash
pytest tests
```

The initial tests are marked as skipped because the kernels are TODOs. After implementing one kernel, remove the skip marker from the corresponding test file.

## Benchmarks

```bash
python benchmarks/bench_gemm.py --impl naive --m 1024 --n 1024 --k 1024
python benchmarks/bench_rmsnorm.py --impl cuda --rows 4096 --hidden 4096
python benchmarks/bench_softmax.py --impl cuda --rows 4096 --cols 2048
```

Benchmark scripts are ready to use after the selected kernel implementation is filled in.

Batch benchmark by shape list:

```bash
python benchmarks/run_all.py --kernel gemm --impl tiled
python benchmarks/run_all.py --kernel rmsnorm --impl cuda
python benchmarks/run_all.py --kernel softmax --impl cuda
```

## Profiling

```bash
bash profiling/profile_gemm.sh tiled 4096 4096 4096
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
