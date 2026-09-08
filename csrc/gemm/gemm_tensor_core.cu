#include "gemm/gemm.h"

#include <mma.h>

using namespace nvcuda;

namespace {
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

__global__ void gemm_onewarp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K) {
  const int row = blockIdx.y * WMMA_M;
  const int col = blockIdx.x * WMMA_N;
  const int tid = threadIdx.x;

  __shared__ __align__(32) float A_tile[WMMA_M][WMMA_K];
  __shared__ __align__(32) float B_tile[WMMA_K][WMMA_N];
  __shared__ __align__(32) float C_tile[WMMA_M][WMMA_N];

  wmma::fragment<
      wmma::matrix_a,
      WMMA_M,
      WMMA_N,
      WMMA_K,
      wmma::precision::tf32,
      wmma::row_major>
      a_frag;
  wmma::fragment<
      wmma::matrix_b,
      WMMA_M,
      WMMA_N,
      WMMA_K,
      wmma::precision::tf32,
      wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k_iter = 0; k_iter < K; k_iter += WMMA_K) {
    for (int idx = tid; idx < WMMA_M * WMMA_K; idx += blockDim.x) {
      const int tile_row = idx / WMMA_K;
      const int tile_col = idx % WMMA_K;
      const int global_row = row + tile_row;
      const int global_col = k_iter + tile_col;

      A_tile[tile_row][tile_col] =
          (global_row < M && global_col < K)
              ? A[global_row * K + global_col]
              : 0.0f;
    }

    for (int idx = tid; idx < WMMA_K * WMMA_N; idx += blockDim.x) {
      const int tile_row = idx / WMMA_N;
      const int tile_col = idx % WMMA_N;
      const int global_row = k_iter + tile_row;
      const int global_col = col + tile_col;

      B_tile[tile_row][tile_col] =
          (global_row < K && global_col < N)
              ? B[global_row * N + global_col]
              : 0.0f;
    }

    __syncthreads();

    wmma::load_matrix_sync(a_frag, &A_tile[0][0], WMMA_K);
    wmma::load_matrix_sync(b_frag, &B_tile[0][0], WMMA_N);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
  }

  wmma::store_matrix_sync(&C_tile[0][0], c_frag, WMMA_N, wmma::mem_row_major);
  __syncthreads();

  for (int idx = tid; idx < WMMA_M * WMMA_N; idx += blockDim.x) {
    const int tile_row = idx / WMMA_N;
    const int tile_col = idx % WMMA_N;
    const int global_row = row + tile_row;
    const int global_col = col + tile_col;

    if (global_row < M && global_col < N) {
      C[global_row * N + global_col] = C_tile[tile_row][tile_col];
    }
  }
}
}  // namespace

torch::Tensor gemm_tensor_core(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);

  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  auto c = torch::zeros({m, n}, a.options());
  if (m == 0 || n == 0) {
    return c;
  }

  dim3 block(32);
  dim3 grid(ceil_div(n, WMMA_N), ceil_div(m, WMMA_M));

  gemm_onewarp_kernel<<<grid, block>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      m,
      n,
      k);
  CUDA_CHECK(cudaGetLastError());

  return c;
}
