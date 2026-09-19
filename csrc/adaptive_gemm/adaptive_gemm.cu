#include "adaptive_gemm.h"

#include <limits>

#include <c10/cuda/CUDAGuard.h>

#include "common/tensor_check.h"
#include "gemm_problem.h"
#include "kernel_dispatch.h"

namespace {

void check_input(const torch::Tensor& a, const torch::Tensor& b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_FLOAT32(a);
  CHECK_FLOAT32(b);
  TORCH_CHECK(
    a.dim() == 2 && b.dim() == 2,
    "A and B must be 2D tensors"
  );
  TORCH_CHECK(
    a.size(1) == b.size(0),
    "Matrix multiplication shape mismatch: A has ", a.size(1),
    " columns, but B has ", b.size(0), " rows"
  );
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same CUDA device");
  TORCH_CHECK(
    a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0,
    "M, N, and K must be positive");
  TORCH_CHECK(
    a.size(0) <= std::numeric_limits<int>::max() &&
    a.size(1) <= std::numeric_limits<int>::max() &&
    b.size(1) <= std::numeric_limits<int>::max(),
    "M, N, and K must fit in the CUTLASS int32 problem size");
}

EpilogueKind parse_epilogue(const std::string& value) {
  if (value == "linear") {
    return EpilogueKind::Linear;
  }
  if (value == "bias") {
    return EpilogueKind::Bias;
  }
  if (value == "relu") {
    return EpilogueKind::Relu;
  }
  if (value == "gelu") {
    return EpilogueKind::Gelu;
  }
  if (value == "bias_relu") {
    return EpilogueKind::BiasRelu;
  }
  if (value == "bias_gelu") {
    return EpilogueKind::BiasGelu;
  }
  TORCH_CHECK(false, "unsupported epilogue: ", value);
}

KernelId parse_kernel(const std::string& value) {
  if (value == "auto") {
    return KernelId::Auto;
  }
  if (value == "fast") {
    return KernelId::Fast128x128Stage4FP32;
  }
  if (value == "fallback") {
    return KernelId::Fallback128x128Stage4FP32;
  }
  TORCH_CHECK(
    false,
    "unsupported adaptive_gemm kernel: ", value,
    "; expected one of: auto, fast, fallback");
}

void check_c(const torch::Tensor& c, const torch::Tensor& a, int64_t m, int64_t n) {
  CHECK_INPUT(c);
  CHECK_FLOAT32(c);
  TORCH_CHECK(c.device() == a.device(), "C must be on the same CUDA device as A");
  TORCH_CHECK(c.dim() == 2, "C must be a 2D tensor");
  TORCH_CHECK(
    c.size(0) == m && c.size(1) == n,
    "C must have shape [", m, ", ", n, "] but got ", c.sizes());
}

}  // namespace

torch::Tensor adaptive_gemm(
  const torch::Tensor& a,
  const torch::Tensor& b,
  const c10::optional<at::Tensor>& c,
  const c10::optional<at::Tensor>& bias,
  double alpha,
  double beta,
  const std::string& epilogue,
  const std::string& kernel) {
  check_input(a, b);

  const EpilogueKind epilogue_kind = parse_epilogue(epilogue);
  const KernelId requested_kernel = parse_kernel(kernel);
  TORCH_CHECK(
    epilogue_kind == EpilogueKind::Linear,
    "adaptive_gemm currently supports only the linear epilogue");

  const bool has_bias = bias.has_value() && bias->defined();
  TORCH_CHECK(!has_bias, "adaptive_gemm bias fusion is not implemented yet");

  const int64_t m = a.size(0);
  const int64_t n = b.size(1);
  const bool has_c = c.has_value() && c->defined();
  TORCH_CHECK(has_c || beta == 0.0, "C is required when beta is non-zero");
  if (has_c) {
    check_c(*c, a, m, n);
  }

  c10::cuda::CUDAGuard device_guard(a.device());
  auto d = torch::empty({m, n}, a.options());

  const cudaStream_t current_stream =
    c10::cuda::getCurrentCUDAStream(a.get_device()).stream();

  GemmProblem problem(a, b, d, current_stream);

  problem.has_c = has_c;
  problem.c = has_c ? c->const_data_ptr() : d.const_data_ptr();
  problem.dtype_c = has_c ? c->scalar_type() : d.scalar_type();
  problem.ldc = has_c ? c->stride(0) : d.stride(0);

  problem.has_bias = false;
  problem.dtype_bias = at::ScalarType::Undefined;
  problem.bias = nullptr;

  problem.epilogue.kind = epilogue_kind;
  problem.epilogue.alpha = static_cast<float>(alpha);
  problem.epilogue.beta = static_cast<float>(beta);

  const cutlass::Status status = dispatch_gemm(problem, requested_kernel);
  TORCH_CHECK(
    status == cutlass::Status::kSuccess,
    "adaptive GEMM dispatch failed: ", cutlassGetStatusString(status));
  return d;
}
