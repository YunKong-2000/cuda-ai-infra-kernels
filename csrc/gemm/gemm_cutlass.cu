#include "cutlass/gemm/device/gemm.h"
#include "gemm.h"
#include <c10/cuda/CUDAStream.h>
namespace
{
using RowMajor = cutlass::layout::RowMajor;
using ThreadBlockShape = cutlass::gemm::GemmShape<128,128,16>;
using WarpShape = cutlass::gemm::GemmShape<64,64,16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
using FastEpilogueOp = cutlass::epilogue::thread::LinearCombination<float, 4, float, float>;
using FallbackEpilogueOp = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using FastGemm = cutlass::gemm::device::Gemm<float,
                                                 RowMajor,
                                                 float,
                                                 RowMajor,
                                                 float,
                                                 RowMajor,
                                                 float,
                                                 cutlass::arch::OpClassTensorOp,
                                                 cutlass::arch::Sm80,
                                                 ThreadBlockShape,
                                                 WarpShape,
                                                 InstructionShape,
                                                 FastEpilogueOp,
                                                 Swizzle,
                                                 4,
                                                 4,
                                                 4
                                                >;

using FallbackGemm = cutlass::gemm::device::Gemm<float,
                                                 RowMajor,
                                                 float,
                                                 RowMajor,
                                                 float,
                                                 RowMajor,
                                                 float,
                                                 cutlass::arch::OpClassTensorOp,
                                                 cutlass::arch::Sm80,
                                                 ThreadBlockShape,
                                                 WarpShape,
                                                 InstructionShape,
                                                 FallbackEpilogueOp,
                                                 Swizzle,
                                                 4,
                                                 1,
                                                 1
                                                >;

}

template<typename Gemm>
cutlass::Status lauch_gemm
(
  int M,
  int N,
  int K,
  const float* A,
  const float* B,
  float* C,
  cudaStream_t stream
)
{
  Gemm gemmop;
  typename Gemm::Arguments args = {{M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1, 0}};
  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }
  return gemmop(args, nullptr, stream);
}

cudaError_t cutlassSgemm
(
  int M,
  int N,
  int K,
  const float* A,
  const float* B,
  float* C,
  cudaStream_t stream
)
{
  cutlass::Status status;
  if (K % 4 == 0 && N % 4 == 0) {
    status = lauch_gemm<FastGemm>(M, N, K, A, B, C, stream);
  } 
  else {
    status = lauch_gemm<FallbackGemm>(M, N, K, A, B, C, stream);
  }

  if (status == cutlass::Status::kErrorMisalignedOperand) {
        return cudaErrorInvalidValue;
  }

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
  cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  cudaError_t status = cutlassSgemm(
    M, 
    N, 
    K, 
    a.data_ptr<float>(),
    b.data_ptr<float>(),
    c.data_ptr<float>(),
    stream
  );
  TORCH_CHECK(
        status == cudaSuccess,
        "CUTLASS GEMM failed: ",
        cudaGetErrorString(status));
  return c;
}