#pragma once

#include <torch/extension.h>

torch::Tensor rope_forward(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);

