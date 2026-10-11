#include "gemm/gemm_regular.cuh"

void check_gemm_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_FLOAT32(a);
  CHECK_FLOAT32(b);
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same CUDA device");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: A[M,K] @ B[K,N]");
  TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max(), "M is too large for this reference kernel");
  TORCH_CHECK(a.size(1) <= std::numeric_limits<int>::max(), "K is too large for this reference kernel");
  TORCH_CHECK(b.size(1) <= std::numeric_limits<int>::max(), "N is too large for this reference kernel");
}

namespace {
constexpr int BM = 16;
constexpr int BN = 16;

__global__ void gemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K)
{
  const int row = blockIdx.y * BM + threadIdx.y;
  const int col = blockIdx.x * BN + threadIdx.x;
  float acc = 0;
  for (int k = 0; k < K; k++) {
    acc += A[int64_t(k) * M + row] * B[int64_t(k) * N + col];
  }
  C[int64_t(row) * N + col] = acc;
}
} // namespace

torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b) {
  check_regular_gemm_inputs(a, b, BM, BN, 1, false);
  const int M = a.size(0), K = a.size(1), N = b.size(1);
  const c10::cuda::CUDAGuard device_guard(a.device());
  auto c = torch::empty({M, N}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  gemm_naive_kernel<<<dim3(N / BN, M / BM), dim3(BN, BM), 0, stream.stream()>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
