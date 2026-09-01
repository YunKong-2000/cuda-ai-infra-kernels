#pragma once

#include <torch/extension.h>
#include <limits>
#include "common/cuda_check.h"
#include "common/launch_utils.h"
#include "common/tensor_check.h"

torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_cublas(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_block_tile(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_thread_tile(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_warp_tile(torch::Tensor a, torch::Tensor b);
void check_gemm_inputs(const torch::Tensor& a, const torch::Tensor& b);
