#include "common/tensor_check.h"
#include "gemm/gemm.h"

torch::Tensor gemm_thread_tile(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_FLOAT32(a);
  CHECK_FLOAT32(b);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: A[M,K] @ B[K,N]");
  TORCH_CHECK(false, "gemm_thread_tile is a TODO: implement this kernel in csrc/gemm/gemm_thread_tile.cu");
  return torch::Tensor();
}
