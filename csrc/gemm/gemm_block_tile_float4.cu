#include "gemm/gemm_regular.cuh"

namespace {
constexpr int BK = 8;

template<int BM, int BN>
__global__ void gemm_block_tile_float4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  constexpr int THREADS = BM * BN;
  static_assert(THREADS <= 1024);
  const int tid = threadIdx.x;
  const int ty = tid / BN, tx = tid % BN;
  const int by = blockIdx.y, bx = blockIdx.x;
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  float acc = 0;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    loadSmem<BM, BK, THREADS>(A + int64_t(k_iter) * M + by * BM, &As[0][0], M, tid);
    loadSmem<BN, BK, THREADS>(B + int64_t(k_iter) * N + bx * BN, &Bs[0][0], N, tid);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      acc += As[k][ty] * Bs[k][tx];
    }
    __syncthreads();
  }
  // Exactly one scalar accumulator and one C element per thread.
  C[int64_t(by * BM + ty) * N + bx * BN + tx] = acc;
}

template<int BM, int BN>
torch::Tensor launch_block_tile_float4(torch::Tensor a, torch::Tensor b) {
  check_regular_gemm_inputs(a, b, BM, BN, BK);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  const c10::cuda::CUDAGuard device_guard(a.device());
  auto c = torch::empty({M, N}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  gemm_block_tile_float4_kernel<BM, BN><<<dim3(N / BN, M / BM), BM * BN, 0, stream.stream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
} // namespace

torch::Tensor gemm_block_tile_float4_16x16(torch::Tensor a, torch::Tensor b) {
  return launch_block_tile_float4<16, 16>(a, b);
}
torch::Tensor gemm_block_tile_float4_16x32(torch::Tensor a, torch::Tensor b) {
  return launch_block_tile_float4<16, 32>(a, b);
}
torch::Tensor gemm_block_tile_float4_32x32(torch::Tensor a, torch::Tensor b) {
  return launch_block_tile_float4<32, 32>(a, b);
}
