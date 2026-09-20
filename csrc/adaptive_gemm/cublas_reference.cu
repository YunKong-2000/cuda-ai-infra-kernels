#include "cublas_reference.h"

#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

#include <cstddef>

#include "common/cuda_check.h"

namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
    status == CUBLAS_STATUS_SUCCESS,
    operation, " failed with cuBLAS status ", static_cast<int>(status));
}

}  // namespace

void launch_cublas_reference(const GemmProblem& problem) {
  // The classic cuBLAS API uses C as both source and destination. Preserve the
  // adaptive_gemm D = alpha * A * B + beta * C contract when C is separate.
  if (problem.epilogue.beta != 0.0f && problem.c != problem.d) {
    const std::size_t output_bytes =
      static_cast<std::size_t>(problem.M) *
      static_cast<std::size_t>(problem.N) * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(
      problem.d,
      problem.c,
      output_bytes,
      cudaMemcpyDeviceToDevice,
      problem.stream));
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  // cuBLAS is column-major. Swapping A/B computes
  // D^T[N,M] = alpha * B^T[N,K] * A^T[K,M] + beta * C^T[N,M].
  check_cublas(
    cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      problem.N,
      problem.M,
      problem.K,
      &problem.epilogue.alpha,
      problem.b,
      CUDA_R_32F,
      static_cast<int>(problem.ldb),
      problem.a,
      CUDA_R_32F,
      static_cast<int>(problem.lda),
      &problem.epilogue.beta,
      problem.d,
      CUDA_R_32F,
      static_cast<int>(problem.ldd),
      CUBLAS_COMPUTE_32F_FAST_TF32,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
    "cublasGemmEx");
}
