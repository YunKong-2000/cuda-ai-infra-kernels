#pragma once

#include <torch/extension.h>

torch::Tensor attention_decode_forward(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache);

