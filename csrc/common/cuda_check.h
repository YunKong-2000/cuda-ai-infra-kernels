#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                        \
    cudaError_t status = (expr);                                              \
    TORCH_CHECK(status == cudaSuccess, "CUDA error: ", cudaGetErrorString(status)); \
  } while (0)

