#include "common/tensor_check.h"
#include "softmax/softmax.h"

torch::Tensor softmax_forward(torch::Tensor x) {
  CHECK_INPUT(x);
  CHECK_FLOAT32(x);
  TORCH_CHECK(x.dim() == 2, "softmax expects a 2D tensor [rows, cols]");
  TORCH_CHECK(false, "softmax_forward is a TODO: implement this kernel in csrc/softmax/softmax.cu");
  return torch::Tensor();
}
