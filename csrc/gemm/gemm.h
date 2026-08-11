#pragma once

#include <torch/extension.h>

torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_tiled(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_thread_tile(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_warp_tile(torch::Tensor a, torch::Tensor b);

