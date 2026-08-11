#include "common/tensor_check.h"
#include "rope/rope.h"

torch::Tensor rope_forward(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
  CHECK_INPUT(x);
  CHECK_INPUT(cos);
  CHECK_INPUT(sin);
  TORCH_CHECK(false, "rope_forward is planned for v0.2");
}

