#include "gemm/gemm.h"
const int BN = 16;
const int BM = 16;
const int BK = 16;
constexpr int VEC = 4;
constexpr int A_VEC_COLS = BK / VEC;
constexpr int A_VEC_COUNT = BM * A_VEC_COLS;
constexpr int B_VEC_COLS = BN / VEC;
constexpr int B_VEC_COUNT = BK * B_VEC_COLS;


__global__ void gemm_block_tile_kernel_scalar
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) 
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int by = row / BM;
  const int bx = col / BN;
  const int y_offset = by * BM;
  const int x_offset = bx * BN;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int idx = threadIdx.y * blockDim.x + threadIdx.x;
  __shared__ float __align__(16) A_tile[BM][BK];
  __shared__ float B_tile[BK][BN];
  float acc = 0;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {

    int ix = 0;
    int iy = 0;
    //load A from global to shared memory
    for (int t = idx; t < BM * BK; t += blockDim.x * blockDim.y) {
      ix = t % BK;
      iy = t / BK;
      A_tile[iy][ix] =
      ((iy + y_offset) < M && (ix + k_iter) < K)
        ? A[(iy + y_offset) * K + ix + k_iter]
        : 0.0f;
    }
    //load A from global to shared memory
    for (int t = idx; t < BN * BK; t += blockDim.x * blockDim.y) {
      ix = t % BN;
      iy = t / BN;
      B_tile[iy][ix] =
      ((iy + k_iter) < K && (ix + x_offset) < N)
        ? B[(iy + k_iter) * N + ix + x_offset]
        : 0.0f;
    }
    __syncthreads();
    
    //outer product
    for (int k = 0; k < BK; k++){
      acc += A_tile[ty][k] * B_tile[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}


__global__ void gemm_block_tile_kernel_float4
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) 
{
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int by = row / BM;
  const int bx = col / BN;
  const int y_offset = by * BM;
  const int x_offset = bx * BN;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  __shared__ float __align__(16) A_tile[BM][BK];
  __shared__ float __align__(16) B_tile[BK][BN];
  float acc = 0;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    int ix = 0;
    int iy = 0;
    int global_col;
    int global_row;
    if (tid < A_VEC_COUNT) {
      ix = tid % A_VEC_COLS * 4;
      iy = tid / A_VEC_COLS;
      global_col = ix + k_iter;
      global_row = iy + y_offset;
      if (global_row < M && global_col + 3 < K) {
        float4 value = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
        *reinterpret_cast<float4*>(&A_tile[iy][ix]) = value;
      }
      else {
#pragma unroll
        for (int i = 0; i < VEC; i++) {
          A_tile[iy][ix + i] = 
          (global_row < M && global_col + i < K) ?
          A[global_row * K + global_col + i] 
          : 0;
        }
      }
    }

    if (tid < B_VEC_COUNT) {
      ix = tid % B_VEC_COLS * 4;
      iy = tid / B_VEC_COLS;
      global_col = ix + x_offset;
      global_row = iy + k_iter;
      if (global_row < K && global_col + 3 < N) {
        float4 value = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
        *reinterpret_cast<float4*>(&B_tile[iy][ix]) = value;
      }
      else {
#pragma unroll
        for (int i = 0; i < VEC; i++) {
          B_tile[iy][ix + i] = 
          (global_row < K && global_col + i < N) ?
          B[global_row * N + global_col + i] 
          : 0;
        }
      }
    }
    
    __syncthreads();
    
    //out product
#pragma unroll
    for (int k = 0; k < BK; k++){
      acc += A_tile[ty][k] * B_tile[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}


torch::Tensor gemm_block_tile(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);

  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  auto c = torch::zeros({m, n}, a.options());
  dim3 block(16, 16);
  dim3 grid(ceil_div(n, static_cast<int>(block.x)), ceil_div(m, static_cast<int>(block.y)));
  if (k % VEC == 0 && n % VEC == 0) {
    gemm_block_tile_kernel_float4<<<grid, block>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      m,
      n,
      k);
  }
  else {
    gemm_block_tile_kernel_scalar<<<grid, block>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      m,
      n,
      k);
  }
  
  CUDA_CHECK(cudaGetLastError());
  return c;
}