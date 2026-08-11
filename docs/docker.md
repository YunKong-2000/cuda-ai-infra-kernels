# Docker Runtime

Use `scripts/docker_run.sh` to start an interactive development container from a locally available NVIDIA image.

The script intentionally does not pull images. Pull the NVIDIA image you want first, then pass the exact local image tag.

```bash
docker pull nvcr.io/nvidia/pytorch:<tag>
scripts/docker_run.sh nvcr.io/nvidia/pytorch:<tag>
```

Equivalent environment-variable form:

```bash
CUDA_KERNELS_IMAGE=nvcr.io/nvidia/pytorch:<tag> scripts/docker_run.sh
```

Run a one-off command:

```bash
scripts/docker_run.sh nvcr.io/nvidia/pytorch:<tag> nvidia-smi
scripts/docker_run.sh nvcr.io/nvidia/pytorch:<tag> ncu --version
```

## Docker Flags

The script uses:

- `--gpus all`: expose all visible NVIDIA GPUs to the container.
- `--ipc=host`: use the host IPC namespace so the container is not limited by Docker's small default `/dev/shm`.
- `--cap-add=SYS_ADMIN`: grant the capability commonly needed by Nsight Compute performance counter collection.
- `--security-opt seccomp=unconfined`: avoid seccomp blocking profiler-related system calls.
- `--user "$(id -u):$(id -g)"`: write mounted project files with the same UID/GID as the current host user.
- `--volume "$PROJECT_ROOT:/workspace/cuda-ai-infra-kernels"`: mount this repository into the container.
- `--workdir /workspace/cuda-ai-infra-kernels`: start inside the mounted repository.

If Nsight Compute still reports permission errors, the host driver may restrict GPU performance counters. In that case, ask the server administrator to enable profiling access on the host, or run with additional site-approved Docker permissions.

