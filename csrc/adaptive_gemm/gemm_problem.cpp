#include "gemm_problem.h"

#include <limits>

namespace {

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(
    value >= 0 && value <= std::numeric_limits<int>::max(),
    name, " exceeds the CUTLASS int32 range: ", value);
  return static_cast<int>(value);
}

}  // namespace

GemmProblem::GemmProblem(
  const torch::Tensor& a,
  const torch::Tensor& b,
  torch::Tensor& d,
  cudaStream_t current_stream)
    : M(checked_int(a.size(0), "M")),
      N(checked_int(b.size(1), "N")),
      K(checked_int(a.size(1), "K")),
      a(a.const_data_ptr()),
      b(b.const_data_ptr()),
      d(d.mutable_data_ptr()),
      lda(a.stride(0)),
      ldb(b.stride(0)),
      ldd(d.stride(0)),
      dtype_a(a.scalar_type()),
      dtype_b(b.scalar_type()),
      dtype_d(d.scalar_type()),
      stream(current_stream) {
  checked_int(lda, "lda");
  checked_int(ldb, "ldb");
  checked_int(ldd, "ldd");
}
