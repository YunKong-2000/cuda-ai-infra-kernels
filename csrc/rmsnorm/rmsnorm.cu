#include "common/tensor_check.h"
#include "rmsnorm/rmsnorm.h"
const int VEC = 4;
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 256;

__device__ __force_inline__ float warp_reduce_sum(float val)
{
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void rmsnorm_one_row_kernel
(
  const float* __restrict__ x,
  const float* __restrict__ weight,
  float* __restrict__ out,
  int rows,
  int hidden
)
{
  int row = blockIdx.x;
  if (row >= rows) return;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];
  float thread_sum = 0;
  float row_sum = 0;
  float rms = 0;
  for (int i = tid; i < hidden; i += BLOCK_SIZE * VEC) {
    if (i + VEC < hidden) {
      float4 x_vec = *reinterpret_cast<const float4*>(x + row * hidden + i);
      thread_sum += x_vec.x * x_vec.x + x_vec.y * x_vec.y + x_vec.z * x_vec.z + x_vec.w * x_vec.w;
    }
    else {
      for (int j = i; j < hidden; j++) {
        float x_val = x[row * hidden + j];
        thread_sum += x_val * x_val;
      }
    }
  }
  thread_num = warp_reduce_sum(thread_sum);
  if (lane_id == 0) {
    shared_sum[warp_id] = thread_sum;
  }
  __syncthreads();
  if (warp_id == 0) {

    thread_sum = (lane_id < BLOCK_SIZE / WARP_SIZE) ? shared_sum[lane_id] : 0;
    thread_sum = warp_reduce_sum(thread_sum);
  }
  if (tid == 0) {
    row_sum = thread_sum;
    rms = rsqrtf(row_sum / hidden + 1e-8);
  }
  __syncthreads();
  for (int i = tid; i < hidden; i += BLOCK_SIZE * VEC) {
    if (i + VEC < hidden) {
      float4 x_vec = *reinterpret_cast<const float4*>(x + row * hidden + i);
      float4 w_vec = *reinterpret_cast<const float4*>(weight + i);
      float4 out_vec;
      out_vec.x = x_vec.x * w_vec.x * rms;
      out_vec.y = x_vec.y * w_vec.y * rms;
      out_vec.z = x_vec.z * w_vec.z * rms;
      out_vec.w = x_vec.w * w_vec.w * rms;
      *reinterpret_cast<float4*>(out + row * hidden + i) = out_vec;
    }
    else {
      for (int j = i; j < hidden; j++) {
        float x_val = x[row * hidden + j];
        float w_val = weight[j];
        out[row * hidden + j] = x_val * w_val * rms;
      }
    }
  }
}



torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_FLOAT32(x);
  CHECK_FLOAT32(weight);
  TORCH_CHECK(x.dim() == 2, "x must be [rows, hidden]");
  TORCH_CHECK(weight.dim() == 1, "weight must be [hidden]");
  TORCH_CHECK(x.size(1) == weight.size(0), "hidden size mismatch");
  auto out = torch::empty_like(x);
  int rows = x.size(0);
  int hidden = x.size(1);
  dim3 grid(rows);
  dim3 block(BLOCK_SIZE);
  rmsnorm_one_row_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), rows, hidden);
  return out;
}
