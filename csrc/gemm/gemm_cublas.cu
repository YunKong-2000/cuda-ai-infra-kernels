#include "gemm/gemm.h"
#include "gemm/gemm_layout.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

torch::Tensor gemm_cublas(torch::Tensor a, torch::Tensor b) {
  const bool column_major_a = is_column_major_gemm_a(a);
  if (column_major_a) {
    check_column_major_gemm_inputs(a, b);
  } else {
    check_gemm_inputs(a, b);
  }

  const c10::cuda::CUDAGuard device_guard(a.device());
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));
  auto c = torch::empty({m, n}, a.options());
  if (m == 0 || n == 0) {
    return c;
  }
  if (k == 0) {
    return c.zero_();
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // cuBLAS is column-major. Swapping A/B computes
  // C^T[N,M] = B^T[N,K] * A^T[K,M]. Row-major A already stores A^T;
  // column-major A needs OP_T with lda=M. Neither path copies the inputs.
  // PyTorch's handle is bound to the current device and current CUDA stream.
  // PEDANTIC keeps this FP32 baseline independent of global TF32 settings.
  TORCH_CHECK(
      cublasGemmEx(
          handle,
          CUBLAS_OP_N,
          column_major_a ? CUBLAS_OP_T : CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          b.data_ptr<float>(),
          CUDA_R_32F,
          n,
          a.data_ptr<float>(),
          CUDA_R_32F,
          column_major_a ? m : k,
          &beta,
          c.data_ptr<float>(),
          CUDA_R_32F,
          n,
          CUBLAS_COMPUTE_32F_PEDANTIC,
          CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
      "cublasGemmEx (FP32) failed");

  return c;
}
