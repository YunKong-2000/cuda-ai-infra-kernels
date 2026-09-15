#include "cutlass/gemm/device/gemm.h"
#include "gemm.h"

cudaError_t cutlassSgemm
(
  int M,
  int N,
  int K,
  const float* A,
  const float* B,
  float* C
)
{
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm = cutlass::gemm::device::Gemm<float,
                                                 RowMajor,
                                                 float,
                                                 RowMajor,
                                                 float,
                                                 RowMajor>;
  CutlassGemm gemmop;
  CutlassGemm::Arguments args = {{M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1, 0}};
  cutlass::Status status = gemmop(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

torch::Tensor gemm_cutlass(torch::Tensor a, torch::Tensor b){
  check_gemm_inputs(a, b);
  const int M = static_cast<int>(a.size(0));
  const int N = static_cast<int>(b.size(1));
  const int K = static_cast<int>(b.size(0));
  torch::Tensor c = torch::zeros({M, N}, a.options());
  cudaError_t status = cutlassSgemm(
    M, 
    N, 
    K, 
    a.data_ptr<float>(),
    b.data_ptr<float>(),
    c.data_ptr<float>()
  );
  return c;
}