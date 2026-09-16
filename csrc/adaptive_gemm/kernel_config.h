#include "cutlass\gemm\device\gemm.h"
#include "gemm.h"

enum class KernelId {
  Balanced128x128Stage3,
  Balanced128x128Stage4,
  SmallM64x128,
  SmallN128x64,
  ScalarFallback,
  SplitK,
};

using ElementAccumulator = float;
using ElementEpilogue = float;
using ElementA = float;
using ElementB = float;
using ElementC = float;

using RowMajor = cutlass::layout::RowMajor;
using ColMajor = cutlass::layout::ColumnMajor;
using LayoutA = RowMajor;
using LayoutB = RowMajor;
using LayoutC = RowMajor;

using ArchTag = cutlass::arch::Sm80;

using OpClass = cutlass::arch::OpClassTensorOp;
using OpMMA = cutlass::arch::OpMultiplyAdd;

using LargeThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using MediumThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 16>;

using LargeWarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using MediumWarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
using SmallWarpShape = cutlass::gemm::GemmShape<32, 32, 16>;

using LargeMutiplyAddShape = cutlass::gemm::GemmShape<16, 8, 8>;
using SmallMutiplyAddShape = cutlass::gemm::GemmShape<16, 8, 4>;

const int kAlignmentA = 128 / sizeof_bits<ElementA>::value;
const int kAlignmentB = 128 / sizeof_bits<ElementB>::value;

using FastFP32EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, kAlignmentA, ElementAccumilator, ElementAccumilator>;
using FallbackFP32EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumilator, ElementAccumilator>;

using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

const int kstages_num = 4;








