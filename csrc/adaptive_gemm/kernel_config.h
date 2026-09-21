#pragma once
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"

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
using LargeThreadBlockShapeLargeK = cutlass::gemm::GemmShape<128, 128, 32>;
using MediumThreadBlockShape = cutlass::gemm::GemmShape<128, 64, 16>;

using LargeWarpShape = cutlass::gemm::GemmShape<64, 64, 16>;
using LargeWarpShapeLargeK = cutlass::gemm::GemmShape<64, 64, 32>;
using MediumWarpShape = cutlass::gemm::GemmShape<64, 32, 16>;
using SmallWarpShape = cutlass::gemm::GemmShape<32, 32, 16>;

using LargeMultiplyAddShape = cutlass::gemm::GemmShape<16, 8, 8>;
using SmallMultiplyAddShape = cutlass::gemm::GemmShape<16, 8, 4>;

inline constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
inline constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using FastFP32EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, 4, ElementAccumulator, ElementEpilogue>;
using FastFP32ReluEpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<ElementC, 4, ElementAccumulator, ElementEpilogue>;
using FastFP32SiluEpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<ElementC, 4, ElementAccumulator, ElementEpilogue>;
using FastFP32GeluEpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<ElementC, 4, ElementAccumulator, ElementEpilogue>;
using FallbackFP32EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementEpilogue>;
using FallbackFP32ReluEpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<ElementC, 1, ElementAccumulator, ElementEpilogue>;
using FallbackFP32SiluEpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<ElementC, 1, ElementAccumulator, ElementEpilogue>;
using FallbackFP32GeluEpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<ElementC, 1, ElementAccumulator, ElementEpilogue>;


using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

inline constexpr int kStages = 4;
inline constexpr int k2Stages = 2;
inline constexpr int k3Stages = 3;
