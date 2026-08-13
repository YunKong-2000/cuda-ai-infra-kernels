#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/docker_run.sh <local-nvidia-image> [command...]
  CUDA_KERNELS_IMAGE=<local-nvidia-image> scripts/docker_run.sh [-- command...]

Examples:
  scripts/docker_run.sh nvcr.io/nvidia/pytorch:<tag>
  scripts/docker_run.sh nvcr.io/nvidia/pytorch:<tag> nvidia-smi
  CUDA_KERNELS_IMAGE=nvcr.io/nvidia/pytorch:<tag> scripts/docker_run.sh -- bash

Environment variables:
  CUDA_KERNELS_IMAGE                  Docker image to use when not passed as the first argument.
  CUDA_KERNELS_CONTAINER_NAME         Container name. Default: cuda-ai-infra-kernels-<uid>.
  CUDA_KERNELS_CONTAINER_WORKDIR      Container workdir. Default: /workspace/cuda-ai-infra-kernels.
  CUDA_KERNELS_DOCKER_EXTRA_ARGS      Extra docker run args, split on whitespace.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DOCKER_IMAGE="${CUDA_KERNELS_IMAGE:-}"
if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  DOCKER_IMAGE="$1"
  shift
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ -z "${DOCKER_IMAGE}" ]]; then
  usage >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker command not found" >&2
  exit 127
fi

if ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
  echo "error: Docker image is not available locally: ${DOCKER_IMAGE}" >&2
  echo "Pull an NVIDIA image first, for example: docker pull nvcr.io/nvidia/pytorch:<tag>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
CONTAINER_NAME="${CUDA_KERNELS_CONTAINER_NAME:-cuda-ai-infra-kernels-${USER_ID}}"
CONTAINER_WORKDIR="${CUDA_KERNELS_CONTAINER_WORKDIR:-/workspace/cuda-ai-infra-kernels}"

CONTAINER_CMD=("$@")
if [[ ${#CONTAINER_CMD[@]} -eq 0 ]]; then
  CONTAINER_CMD=("bash")
fi

EXTRA_ARGS=()
if [[ -n "${CUDA_KERNELS_DOCKER_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${CUDA_KERNELS_DOCKER_EXTRA_ARGS}"
fi

exec docker run \
  -it \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --user "${USER_ID}:${GROUP_ID}" \
  --workdir "${CONTAINER_WORKDIR}" \
  --volume "${PROJECT_ROOT}:${CONTAINER_WORKDIR}" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env HOME=/tmp \
  --env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  --env CUDA_CACHE_PATH=/tmp/cuda_cache \
  "${EXTRA_ARGS[@]}" \
  "${DOCKER_IMAGE}" \
  "${CONTAINER_CMD[@]}"
