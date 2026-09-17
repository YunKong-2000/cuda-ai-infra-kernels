#pragma once

#include "kernel_entry.h"
#include "traits.h"


template <typename Gemm>
cutlass::Status launch_gemm(const GemmProblem& problem) {
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  typename Gemm::Arguments args{
    {problem.M, problem.N, problem.K},
    {static_cast<const ElementA*>(problem.a), static_cast<int>(problem.lda)},
    {static_cast<const ElementB*>(problem.b), static_cast<int>(problem.ldb)},
    {static_cast<const ElementC*>(problem.c), static_cast<int>(problem.ldc)},
    {static_cast<ElementC*>(problem.d), static_cast<int>(problem.ldd)},
    {problem.epilogue.alpha, problem.epilogue.beta},
  };

  auto status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  Gemm op;
  return op(args, nullptr, problem.stream);
}

template <typename Gemm>
constexpr KernelMeta make_kernel_meta() {
  return KernelMeta{
    TorchScalarType<typename Gemm::ElementA>::value,
    TorchScalarType<typename Gemm::ElementB>::value,
    TorchScalarType<typename Gemm::ElementC>::value,

    Gemm::ThreadblockShape::kM,
    Gemm::ThreadblockShape::kN,
    Gemm::ThreadblockShape::kK,

    Gemm::WarpShape::kM,
    Gemm::WarpShape::kN,
    Gemm::WarpShape::kK,

    Gemm::InstructionShape::kM,
    Gemm::InstructionShape::kN,
    Gemm::InstructionShape::kK,

    Gemm::kStages,
    Gemm::kAlignmentA,
    Gemm::kAlignmentB,
    Gemm::kAlignmentC,

    Gemm::kSplitKSerial,
  };
}
