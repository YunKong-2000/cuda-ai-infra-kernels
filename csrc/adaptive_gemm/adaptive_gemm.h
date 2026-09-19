#pragma once

#include <string>

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor adaptive_gemm(
  const torch::Tensor& a,
  const torch::Tensor& b,
  const c10::optional<at::Tensor>& c,
  const c10::optional<at::Tensor>& bias,
  double alpha,
  double beta,
  const std::string& epilogue,
  const std::string& kernel);
