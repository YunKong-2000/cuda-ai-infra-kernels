#include "kernel_config.h"

using cutlassSgemmLargeFast = cutlass::gemm::device::Gemm<ElementA,
                                                 LayoutA,
                                                 ElementB,
                                                 LayoutB,
                                                 ElementC,
                                                 LayoutC,
                                                 ElementAccumulator,
                                                 Opclass,
                                                 ArchTag,
                                                 LargeThreadBlockShape,
                                                 LargeWarpShape,
                                                 LargeMutiplyAddShape,
                                                 FastFP32EpilogueOp,
                                                 Swizzle,
                                                 kstages_num,
                                                 kAlignmentA,
                                                 kAlignmentB,
                                                 false,
                                                 OpMMA
                                                 >


using cutlassSgemmLargeFallback = cutlass::gemm::device::Gemm<ElementA,
                                                 LayoutA,
                                                 ElementB,
                                                 LayoutB,
                                                 ElementC,
                                                 LayoutC,
                                                 ElementAccumulator,
                                                 Opclass,
                                                 ArchTag,
                                                 LargeThreadBlockShape,
                                                 LargeWarpShape,
                                                 LargeMutiplyAddShape,
                                                 FallbackFP32EpilogueOp,
                                                 Swizzle,
                                                 kstages_num,
                                                 1,
                                                 1,
                                                 false,
                                                 OpMMA
                                                 >
                
enum class KernelId {
  Balanced128x128Stage3,
  Balanced128x128Stage4,
  SmallM64x128,
  SmallN128x64,
  ScalarFallback,
  SplitK,
};
                                                 