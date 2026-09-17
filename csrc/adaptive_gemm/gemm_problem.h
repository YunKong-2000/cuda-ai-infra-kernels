#pragma once

#include <cuda_runtime_api.h>
#include <torch/extension.h>

enum class EpilogueKind {
  Relu,
  Gelu,
  Bias,
  Linear,
  BiasRelu,
  BiasGelu,
  Residual,
  BiasResidualGelu,
};

struct EpilogueDesc {
  EpilogueKind kind{EpilogueKind::Linear};
  float alpha{1.0f};
  float beta{0.0f};
  const void* bias{nullptr};
  const void* residual{nullptr};
};

struct GemmProblem {
  int M{0};
  int N{0};
  int K{0};

  const void* a{nullptr};
  const void* b{nullptr};
  bool has_c{false};
  const void* c{nullptr};
  bool has_bias{false};
  const void* bias{nullptr};
  void* d{nullptr};

  int64_t lda{0};
  int64_t ldb{0};
  int64_t ldc{0};
  int64_t ldd{0};

  at::ScalarType dtype_a{at::ScalarType::Undefined};
  at::ScalarType dtype_b{at::ScalarType::Undefined};
  at::ScalarType dtype_c{at::ScalarType::Undefined};
  at::ScalarType dtype_d{at::ScalarType::Undefined};
  at::ScalarType dtype_bias{at::ScalarType::Undefined};


  EpilogueDesc epilogue;
  cudaStream_t stream{nullptr};
  GemmProblem(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& d,
    cudaStream_t current_stream);
};
