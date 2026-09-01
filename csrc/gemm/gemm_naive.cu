#include "gemm/gemm.h"



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
__global__ void gemm_naive_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc += a[row * k + i] * b[i * n + col];
  }

  c[row * n + col] = acc;
}

}  // namespace

torch::Tensor gemm_naive(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);

  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  auto c = torch::empty({m, n}, a.options());

  dim3 block(16, 16);
  dim3 grid(ceil_div(n, static_cast<int>(block.x)), ceil_div(m, static_cast<int>(block.y)));

  gemm_naive_kernel<<<grid, block>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      m,
      n,
      k);
  CUDA_CHECK(cudaGetLastError());

  return c;
}
