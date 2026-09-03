#pragma once

#include <torch/extension.h>

torch::Tensor rmsnorm_naive(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rmsnorm_cache_x(torch::Tensor x, torch::Tensor weight, double eps);

