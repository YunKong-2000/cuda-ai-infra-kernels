#pragma once

#include <torch/extension.h>

torch::Tensor softmax_forward(torch::Tensor x);

