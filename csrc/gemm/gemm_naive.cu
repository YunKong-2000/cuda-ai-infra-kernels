#include "common/tensor_check.h"
#include "gemm/gemm.h"

namespace {

void check_gemm_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_FLOAT32(a);
  CHECK_FLOAT32(b);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: A[M,K] @ B[K,N]");
}

}  // namespace

torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);
  TORCH_CHECK(false, "gemm_naive is a TODO: implement this kernel in csrc/gemm/gemm_naive.cu");
  return torch::Tensor();
}
