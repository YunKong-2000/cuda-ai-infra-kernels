#include "gemm/gemm.h"

#include <mma.h>

using namespace nvcuda;

const int BM = 64;
const int BN = 64;
const int BK = 32;
const int VEC = 4;
constexpr int W_M = 16;
constexpr int W_N = 32;
constexpr int W_K = 8;
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 8;

__global__ __launch_bounds__(256, 4)
__global__ void gemm_tensor_core_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K) {
  const int tid = threadIdx.x;
  const int warp_id = threadIdx.x / 32;
  const int row_base = blockIdx.y * BM;
  const int col_base = blockIdx.x * BN;
  const int WARPS_M = BM / W_M;
  const int WARPS_N = BN / W_N;
  const int warp_row = warp_id / WARPS_N;
  const int warp_col = warp_id % WARPS_N;

  const int AS = BK + 8;
  const int BS = BN + 8;
  __shared__ __align__(32) float A_tile[BM][AS];
  __shared__ __align__(32) float B_tile[BK][BS];

  wmma::fragment<
      wmma::matrix_a,
      MMA_M,
      MMA_N,
      MMA_K,
      wmma::precision::tf32,
      wmma::row_major> a_frag;

  wmma::fragment<
      wmma::matrix_b,
      MMA_M,
      MMA_N,
      MMA_K,
      wmma::precision::tf32,
      wmma::row_major> b_frag;

  wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> c_frag[2];
  for (int i = 0; i < 2; i ++) { 
    wmma::fill_fragment(c_frag[i], 0.0f);
  }

  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    for (int idx = tid; idx < BM * BK; idx += blockDim.x) {
      const int tile_row = idx / BK;
      const int tile_col = idx % BK;
      const int global_row = row_base + tile_row;
      const int global_col = k_iter + tile_col;

      A_tile[tile_row][tile_col] =
          (global_row < M && global_col < K)
              ? A[global_row * K + global_col]
              : 0.0f;
    }

    for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
      const int tile_row = idx / BN;
      const int tile_col = idx % BN;
      const int global_row = k_iter + tile_row;
      const int global_col = col_base + tile_col;

      B_tile[tile_row][tile_col] =
          (global_row < K && global_col < N)
              ? B[global_row * N + global_col]
              : 0.0f;
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk += MMA_K) {
      wmma::load_matrix_sync(a_frag, &A_tile[warp_row * W_M][kk], AS);
      wmma::load_matrix_sync(b_frag, &B_tile[kk][warp_col * W_N], BS);
      wmma::mma_sync(c_frag[0], a_frag, b_frag, c_frag[0]);

      wmma::load_matrix_sync(b_frag, &B_tile[kk][warp_col * W_N + MMA_N], BS);
      wmma::mma_sync(c_frag[1], a_frag, b_frag, c_frag[1]);
    }
    __syncthreads();
  }

  const int warp_global_row = row_base + warp_row * W_M;
  const int warp_global_col = col_base + warp_col * W_N;
  #pragma unroll
  for (int nj = 0; nj < 2; ++nj) {
    wmma::store_matrix_sync(
        &C[(warp_global_row) * N + (warp_global_col + nj * MMA_N)],
        c_frag[nj],
        N,
        wmma::mem_row_major);
  }
}



torch::Tensor gemm_tensor_core(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);

  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  auto c = torch::zeros({m, n}, a.options());
  if (m == 0 || n == 0) {
    return c;
  }

  dim3 block(256);
  dim3 grid(ceil_div(n, BN), ceil_div(m, BM));

  gemm_tensor_core_kernel<<<grid, block>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      m,
      n,
      k);
  CUDA_CHECK(cudaGetLastError());

  return c;
}
