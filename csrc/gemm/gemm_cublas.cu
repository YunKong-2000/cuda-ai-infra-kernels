#include "gemm/gemm.h"

#include <ATen/cuda/CublasHandlePool.h>
#include <cublas_v2.h>

torch::Tensor gemm_cublas(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);

  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));
  auto c = torch::empty({m, n}, a.options());

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // cuBLAS is column-major. Swapping A/B computes
  // C^T[N,M] = B^T[N,K] * A^T[K,M] for row-major PyTorch tensors.
  TORCH_CHECK(
      cublasSgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          k,
          &alpha,
          b.data_ptr<float>(),
          n,
          a.data_ptr<float>(),
          k,
          &beta,
          c.data_ptr<float>(),
          n) == CUBLAS_STATUS_SUCCESS,
      "cublasSgemm failed");

  return c;
}
