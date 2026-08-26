#include "common/tensor_check.h"
#include "gemm/gemm.h"
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 4;
const int TN = 4;

__global__ void gemm_thread_tile_kernel
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  __shared__ float A_tile[BM][BK];
  __shared__ float B_tile[BK][BN];
  float A_vec[TM];
  float B_vec[TN];
  



}


torch::Tensor gemm_thread_tile(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);

  return torch::Tensor();
}
