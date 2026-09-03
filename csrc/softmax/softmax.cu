#include "common/tensor_check.h"
#include "softmax/softmax.h"
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 256;
const int VEC = 4;


__global__ void softmax_one_row_kernel
(
  const float* __restrict__ input,
  float* __restrict__ output,
  const int rows,
  const int cols
)
{
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  __shared__ float warp_result[BLOCK_SIZE / WARP_SIZE]; // max for each warp
  __shared__ float block_result; // max for the block
  float local_max = -FLT_MAX;
  // Compute the maximum value in the row
  for (int i = tid; i * VEC < cols; i += BLOCK_SIZE) {
    int offset = i * VEC;
    if (offset + VEC <= cols) {
      float4 data = reinterpret_cast<const float4*>(input + row * cols)[i];
      local_max = fmaxf(fmaxf(fmaxf(data.x, data.y), fmaxf(data.z, data.w)), local_max);
    } else {
      for (int j = 0; j < VEC && offset + j < cols; ++j) {
        float val = input[row * cols + offset + j];
        local_max = fmaxf(local_max, val);
      }
    }
  }

  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
  }

  if (lane_id == 0) {
    warp_result[warp_id] = local_max;
  }

  __syncthreads();

  if (warp_id == 0) {
    local_max = (lane_id < BLOCK_SIZE / WARP_SIZE) ? warp_result[lane_id] : -FLT_MAX;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (lane_id == 0) {
      block_result = local_max;
    }
  }
  __syncthreads();
  local_max = block_result;

  // Compute the sum of exponentials
  float local_sum = 0.0f;
  for (int i = tid; i * VEC < cols; i += BLOCK_SIZE) {
    int offset = i * VEC;
    if (offset + VEC <= cols) { 
      float4 data = reinterpret_cast<const float4*>(input + row * cols)[i];
      float4 exp_data;
      exp_data.x = expf(data.x - local_max);
      exp_data.y = expf(data.y - local_max);
      exp_data.z = expf(data.z - local_max);
      exp_data.w = expf(data.w - local_max);
      local_sum += (exp_data.x + exp_data.y + exp_data.z + exp_data.w);
      reinterpret_cast<float4*>(output + row * cols)[i] = exp_data;
    } else {
      for (int j = 0; j < VEC && offset + j < cols; ++j) {
        float val = input[row * cols + offset + j];
        float exp_val = expf(val - local_max);
        local_sum += exp_val;
        output[row * cols + offset + j] = exp_val;
      }
    }
  }
    
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
  }

  if (lane_id == 0) {
    warp_result[warp_id] = local_sum;
  }

  __syncthreads();

  if (warp_id == 0) {
    local_sum = (lane_id < BLOCK_SIZE / WARP_SIZE) ? warp_result[lane_id] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0) {
      block_result = local_sum;
    }
  }
  __syncthreads();

  local_sum = block_result;

    // Normalize the output
  for (int i = tid; i * VEC < cols; i += BLOCK_SIZE) {
    int offset = i * VEC;
    if (offset + VEC <= cols) {
      float4 data = reinterpret_cast<float4*>(output + row * cols)[i];
      data.x /= local_sum;
      data.y /= local_sum;
      data.z /= local_sum;
      data.w /= local_sum;
      reinterpret_cast<float4*>(output + row * cols)[i] = data;
    } else {
      for (int j = 0; j < VEC && offset + j < cols; ++j) {
        output[row * cols + offset + j] /= local_sum;
      }
    }
  }
}

torch::Tensor softmax_naive(torch::Tensor x) {
  CHECK_INPUT(x);
  CHECK_FLOAT32(x);
  TORCH_CHECK(x.dim() == 2, "softmax expects a 2D tensor [rows, cols]");
  auto output = torch::empty_like(x);
  dim3 block(BLOCK_SIZE);
  dim3 grid(x.size(0));
  softmax_one_row_kernel<<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), x.size(0), x.size(1));
  return output;
}
