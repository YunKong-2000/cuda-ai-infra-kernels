#pragma once

#include <cuda_runtime.h>

template <typename T>
__device__ T warp_reduce_sum(T value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename T>
__device__ T warp_reduce_max(T value) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    T other = __shfl_down_sync(0xffffffff, value, offset);
    value = value > other ? value : other;
  }
  return value;
}

