#include "gemm/gemm_regular.cuh"

namespace {
constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 8;
constexpr int THREADS = BM * BN;

__global__ void gemm_block_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  const int tid = threadIdx.y * BN + threadIdx.x;
  const int by = blockIdx.y, bx = blockIdx.x;
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  float acc = 0;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    for (int i = tid; i < BK * BM; i += THREADS) {
      As[i / BM][i % BM] = A[int64_t(k_iter + i / BM) * M + by * BM + i % BM];
    }
    for (int i = tid; i < BK * BN; i += THREADS) {
      Bs[i / BN][i % BN] = B[int64_t(k_iter + i / BN) * N + bx * BN + i % BN];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      acc += As[k][threadIdx.y] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  C[int64_t(by * BM + threadIdx.y) * N + bx * BN + threadIdx.x] = acc;
}
} // namespace

torch::Tensor gemm_block_tile(torch::Tensor a, torch::Tensor b) {
  check_regular_gemm_inputs(a, b, BM, BN, BK, false);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  const c10::cuda::CUDAGuard device_guard(a.device());
  auto c = torch::empty({M, N}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  gemm_block_tile_kernel<<<dim3(N / BN, M / BM), dim3(BN, BM), 0, stream.stream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
