#pragma once

#include <torch/extension.h>

torch::Tensor softmax_naive(torch::Tensor x);

