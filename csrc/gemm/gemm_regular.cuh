#pragma once

#include "gemm/gemm_layout.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

// Stages 1-5: validate on the host, then use unconditional full-tile accesses.
inline void check_regular_gemm_inputs(
    const torch::Tensor& a, const torch::Tensor& b,
    int BM, int BN, int BK, bool vectorized = true)
{
  check_column_major_gemm_inputs(a, b);
  const auto M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(M > 0 && N > 0 && K > 0, "GEMM requires positive M, N and K");
  TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0,
              "GEMM requires M divisible by ", BM, ", N divisible by ", BN,
              " and K divisible by ", BK);
  TORCH_CHECK(M / BM <= 65535, "GEMM grid.y exceeds 65535");
  if (vectorized) {
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(a.data_ptr<float>()) % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(b.data_ptr<float>()) % 16 == 0,
                "GEMM requires 16-byte aligned A and B pointers");
  }
}

__forceinline__ __device__ void VectorizedAccess(const float* src, float* dst) {
  *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}

// A: [K,M] storage, As: [BK,BM]; B: [K,N], Bs: [BK,BN].
// Synchronous float4 loads: no prefetch and only one shared-memory buffer.
template<int SD, int BK, int THREADS>
__forceinline__ __device__ void loadSmem(
    const float* src, float* dst, const int GD, const int tid)
{
  #pragma unroll
  for (int i = tid; i < BK * SD / 4; i += THREADS) {
    const int row = i / (SD / 4);
    const int col = (i % (SD / 4)) * 4;
    VectorizedAccess(src + int64_t(row) * GD + col, dst + row * SD + col);
  }
}
