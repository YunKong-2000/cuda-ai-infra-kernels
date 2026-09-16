#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include "common/tensor_check.h"

enum class KernelId {
  Balanced128x128Stage3,
  Balanced128x128Stage4,
  SmallM64x128,
  SmallN128x64,
  ScalarFallback,
  SplitK,
};

torch::Tensor adaptive_gemm(torch::Tensor a, 
  torch::Tensor b, 
  const c10::optional<at::Tensor>& c,
  const c10::optional<at::Tensor>& bias, 
  double alpha,
  double beta,
  const std::string& epilogue);

bool check_input(torch::Tensor a, torch::Tensor b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  TORCH_CHECK(
    a.dim() == 2 && b.dim() == 2,
    "A and B must be 2D tensors"
  );
  TORCH_CHECK(
    a.size(1) == b.size(0),
    "Matrix multiplication shape mismatch: A has ", A.size(1),
    " columns, but B has ", B.size(0), " rows"
  );
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