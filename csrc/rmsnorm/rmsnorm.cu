#include "common/tensor_check.h"
#include "rmsnorm/rmsnorm.h"

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_FLOAT32(x);
  CHECK_FLOAT32(weight);
  TORCH_CHECK(x.dim() == 2, "x must be [rows, hidden]");
  TORCH_CHECK(weight.dim() == 1, "weight must be [hidden]");
  TORCH_CHECK(x.size(1) == weight.size(0), "hidden size mismatch");
  TORCH_CHECK(false, "rmsnorm_forward is a TODO: implement this kernel in csrc/rmsnorm/rmsnorm.cu");
  return torch::Tensor();
}
