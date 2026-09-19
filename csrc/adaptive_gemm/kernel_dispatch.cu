#include "kernel_dispatch.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "kernel_entry.h"
#include "kernel_launcher.h"
#include "kernel_sm80.h"

namespace {

bool is_aligned(
  const void* pointer,
  int64_t leading_dimension,
  int alignment_elements,
  std::size_t element_size) {
  if (alignment_elements <= 1) {
    return true;
  }

  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  const auto alignment_bytes =
    static_cast<std::uintptr_t>(alignment_elements) * element_size;
  return address % alignment_bytes == 0 &&
         leading_dimension % alignment_elements == 0;
}

bool is_compatible(const GemmProblem& problem, const KernelMeta& meta) {
  const bool has_same_datatype =
    problem.dtype_a == meta.dtype_a &&
    problem.dtype_b == meta.dtype_b &&
    problem.dtype_c == meta.dtype_d &&
    problem.dtype_d == meta.dtype_d;

  if (!has_same_datatype) {
    return false;
  }

  return is_aligned(problem.a, problem.lda, meta.alignment_a, sizeof(float)) &&
         is_aligned(problem.b, problem.ldb, meta.alignment_b, sizeof(float)) &&
         is_aligned(problem.c, problem.ldc, meta.alignment_c, sizeof(float)) &&
         is_aligned(problem.d, problem.ldd, meta.alignment_c, sizeof(float));
}

const std::array<KernelEntry, 2> kKernelRegistry{{
  {
    KernelId::Fast128x128Stage4FP32,
    "blocktile128x128stage4vector4",
    EpilogueKind::Linear,
    make_kernel_meta<Fast128x128Stage4FP32>(),
    &launch_gemm<Fast128x128Stage4FP32>,
  },
  {
    KernelId::Fallback128x128Stage4FP32,
    "blocktile128x128stage4scalar",
    EpilogueKind::Linear,
    make_kernel_meta<Fallback128x128Stage4FP32>(),
    &launch_gemm<Fallback128x128Stage4FP32>,
  },
}};

}  // namespace

cutlass::Status dispatch_gemm(
  const GemmProblem& problem,
  KernelId requested_kernel) {
  for (const auto& entry : kKernelRegistry) {
    if (requested_kernel != KernelId::Auto && entry.id != requested_kernel) {
      continue;
    }

    if (entry.epilogue_kind != problem.epilogue.kind) {
      continue;
    }

    if (!is_compatible(problem, entry.meta)) {
      continue;
    }

    const cutlass::Status status = entry.launch(problem);

    if (status == cutlass::Status::kSuccess) {
      return status;
    }

    if (status != cutlass::Status::kErrorMisalignedOperand &&
        status != cutlass::Status::kErrorInvalidProblem) {
      return status;
    }
  }

  return cutlass::Status::kErrorNotSupported;
}
