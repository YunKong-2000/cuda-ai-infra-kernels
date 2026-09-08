#include "gemm.h"
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_onewarp_kernel
(
  const float* __restrict__ a,
  const float* __restrict__ b,
  float* __restrict__ c,
  const int M,
  const int N,
  const int K
) {
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int row = by * 16;
  int col = bx * 16;
  
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, float> c_frag;

  for (int k = 0; k < K; k += 8) {
    float* A_tile = A + row * K + k;
    float* B_tile = B + k * N + col;

    wmma::load_matrix_sync(a_frag, A_tile, K);
    wmma::load_matrix_sync(b_frag, B_tile, N);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float* C_tile = C + row * N + col;
  wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
  
}




torch::Tensor gemm_tensor_core(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);

  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  auto c = torch::zeros({m, n}, a.options());
  bool one_warp = true;
  if (one_warp) {
    dim3 block(32);
    dim3 grid(ceil_div(n, static_cast<int>(16)), ceil_div(m, static_cast<int>(16)));
    gemm_onewarp_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
    CUDA_CHECK(cudaGetLastError()); 
  }
  return c;
}