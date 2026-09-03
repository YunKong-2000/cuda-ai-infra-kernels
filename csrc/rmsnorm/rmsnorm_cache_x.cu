#include "common/tensor_check.h"
#include "rmsnorm/rmsnorm.h"
const int MAX_ELEMENTS_PER_THREAD = 16;
const int VEC = 4;
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 256;

__forceinline__  __device__ float warp_reduce_sum(float val)
{
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void rmsnorm_cache_x_kernel
(
  const float* __restrict__ x,
  const float* __restrict__ weight,
  float* __restrict__ out,
  int rows,
  int hidden,
  double eps
)
{
  int row = blockIdx.x;
  if (row >= rows) return;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;
  __shared__ float shared_sum[BLOCK_SIZE / WARP_SIZE];
  __shared__ float shared_rms;
  __shared__ __align__(16) float x_vals[BLOCK_SIZE * MAX_ELEMENTS_PER_THREAD];
  float thread_sum = 0;
  float row_sum = 0;
  float rms = 0;
  // Compute the sum of squares in local thread
  for (int i = tid; i * VEC < hidden; i += BLOCK_SIZE) {
    int offset = i * VEC;
    if (offset + VEC <= hidden) {
      float4 x_vec = *reinterpret_cast<const float4*>(x + row * hidden + offset);
      *reinterpret_cast<float4*>(x_vals + offset) = x_vec;
      thread_sum += x_vec.x * x_vec.x + x_vec.y * x_vec.y + x_vec.z * x_vec.z + x_vec.w * x_vec.w;
    }
    else {
      for (int j = offset; j < hidden; j++) {
        float x_v = x[row * hidden + j];
        x_vals[j] = x_v;
        thread_sum += x_v * x_v;
      }
    }
  }
  // Reduce the sum of squares across threads in the block
  thread_sum = warp_reduce_sum(thread_sum);
  if (lane_id == 0) {
    shared_sum[warp_id] = thread_sum;
  }
  __syncthreads();

  // Reduce the sum of squares across warps in the block
  if (warp_id == 0) {
    thread_sum = (lane_id < BLOCK_SIZE / WARP_SIZE) ? shared_sum[lane_id] : 0;
    thread_sum = warp_reduce_sum(thread_sum);
  }

  // The first thread in the block computes the final sum and rms
  if (tid == 0) {
    row_sum = thread_sum;
    shared_rms = rsqrtf(row_sum / hidden + eps);
  }
  __syncthreads();

  rms = shared_rms;
  // Broadcast the rms value to all threads in the block
  for (int i = tid; i * VEC < hidden; i += BLOCK_SIZE) {
    int offset = i * VEC;
    if (offset + VEC <= hidden) {
      float4 w_vec = *reinterpret_cast<const float4*>(weight + offset);
      float4 out_vec;
      float4 x_vec = *reinterpret_cast<const float4*>(x_vals + offset);
      out_vec.x = x_vec.x * w_vec.x * rms;
      out_vec.y = x_vec.y * w_vec.y * rms;
      out_vec.z = x_vec.z * w_vec.z * rms;
      out_vec.w = x_vec.w * w_vec.w * rms;
      *reinterpret_cast<float4*>(out + row * hidden + offset) = out_vec;
    }
    else {
      for (int j = offset; j < hidden; j++) {
        float w_val = weight[j];
        out[row * hidden + j] = x_vals[j] * w_val * rms;
      }
    }
  }
}



torch::Tensor rmsnorm_cache_x(torch::Tensor x, torch::Tensor weight, double eps) {
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
  if (hidden <= BLOCK_SIZE * MAX_ELEMENTS_PER_THREAD) {
    rmsnorm_cache_x_kernel<<<grid, block>>>(x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), rows, hidden, eps);
  }
  else {
    TORCH_CHECK(false, "hidden size too large for naive implementation");
  }
  return out;
}