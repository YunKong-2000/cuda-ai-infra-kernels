#pragma once

#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps);

