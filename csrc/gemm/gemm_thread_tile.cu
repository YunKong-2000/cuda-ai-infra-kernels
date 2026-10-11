#include "gemm/gemm_regular.cuh"

namespace {
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

template<int TM, int TN>
__global__ void gemm_thread_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  constexpr int THREADS = (BM / TM) * (BN / TN);
  static_assert(THREADS <= 1024);
  const int tid = threadIdx.x;
  const int ty = tid / (BN / TN), tx = tid % (BN / TN);
  const int by = blockIdx.y, bx = blockIdx.x;
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  alignas(16) float rAcc[TM][TN] = {};
  alignas(16) float rA[TM], rB[TN];

  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    loadSmem<BM, BK, THREADS>(A + int64_t(k_iter) * M + by * BM, &As[0][0], M, tid);
    loadSmem<BN, BK, THREADS>(B + int64_t(k_iter) * N + bx * BN, &Bs[0][0], N, tid);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      #pragma unroll
      for (int i = 0; i < TM; i += 4) {
        VectorizedAccess(&As[k][ty * TM + i], rA + i);
      }
      #pragma unroll
      for (int j = 0; j < TN; j += 4) {
        VectorizedAccess(&Bs[k][tx * TN + j], rB + j);
      }
      #pragma unroll
      for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
          rAcc[i][j] += rA[i] * rB[j];
        }
      }
    }
    __syncthreads();
  }
  // Each thread owns one contiguous TM x TN rectangle.
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const int row = by * BM + ty * TM + i;
      const int col = bx * BN + tx * TN + j;
      VectorizedAccess(&rAcc[i][j], C + int64_t(row) * N + col);
    }
  }
}

template<int TM, int TN>
torch::Tensor launch_thread_tile(torch::Tensor a, torch::Tensor b) {
  check_regular_gemm_inputs(a, b, BM, BN, BK);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  const c10::cuda::CUDAGuard device_guard(a.device());
  auto c = torch::empty({M, N}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  gemm_thread_tile_kernel<TM, TN><<<dim3(N / BN, M / BM), (BM / TM) * (BN / TN), 0, stream.stream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
} // namespace

torch::Tensor gemm_thread_tile_4x4(torch::Tensor a, torch::Tensor b) {
  return launch_thread_tile<4, 4>(a, b);
}
torch::Tensor gemm_thread_tile_4x8(torch::Tensor a, torch::Tensor b) {
  return launch_thread_tile<4, 8>(a, b);
}
torch::Tensor gemm_thread_tile_8x8(torch::Tensor a, torch::Tensor b) {
  return launch_thread_tile<8, 8>(a, b);
}
torch::Tensor gemm_thread_tile(torch::Tensor a, torch::Tensor b) {
  return gemm_thread_tile_4x4(a, b);
}
