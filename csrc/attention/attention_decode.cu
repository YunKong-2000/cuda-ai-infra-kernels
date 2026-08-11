#include "attention/attention_decode.h"
#include "common/tensor_check.h"

torch::Tensor attention_decode_forward(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache) {
  CHECK_INPUT(q);
  CHECK_INPUT(k_cache);
  CHECK_INPUT(v_cache);
  TORCH_CHECK(false, "attention_decode_forward is planned for v0.3");
}

