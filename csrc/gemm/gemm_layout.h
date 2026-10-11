#pragma once

#include "gemm/gemm.h"

// Dense column-major A[M,K] has the same storage as contiguous A_storage[K,M],
// but callers pass A_storage.t() so the logical operation remains A @ B.
inline bool is_column_major_gemm_a(const torch::Tensor& a) {
  return a.dim() == 2 && a.stride(0) == 1 && a.stride(1) == a.size(0);
}

inline void check_column_major_gemm_inputs(
    const torch::Tensor& a, const torch::Tensor& b) {
  CHECK_CUDA(a);
  CHECK_CUDA(b);
  CHECK_FLOAT32(a);
  CHECK_FLOAT32(b);
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same CUDA device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: A[M,K] @ B[K,N]");
  TORCH_CHECK(is_column_major_gemm_a(a),
              "A must be column-major [M,K] with strides (1,M)");
  TORCH_CHECK(b.is_contiguous(), "B must be row-major contiguous");
  TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max(), "M is too large");
  TORCH_CHECK(a.size(1) <= std::numeric_limits<int>::max(), "K is too large");
  TORCH_CHECK(b.size(1) <= std::numeric_limits<int>::max(), "N is too large");
}
