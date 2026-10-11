#pragma once

#include <torch/extension.h>
#include <limits>
#include "common/cuda_check.h"
#include "common/launch_utils.h"
#include "common/tensor_check.h"

torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_cublas(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_cutlass(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_block_tile(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_thread_tile(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_warp_tile(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_tensor_core(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_prefetchSmem(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_prefetchReg(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_splictK(torch::Tensor a, torch::Tensor b);
void check_gemm_inputs(const torch::Tensor& a, const torch::Tensor& b);

torch::Tensor gemm_block_tile_float4_16x16(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_block_tile_float4_16x32(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_block_tile_float4_32x32(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_thread_tile_4x4(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_thread_tile_4x8(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_thread_tile_8x8(torch::Tensor a, torch::Tensor b);
