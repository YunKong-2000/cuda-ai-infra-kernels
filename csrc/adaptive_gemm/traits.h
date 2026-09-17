#pragma once

#include <ATen/core/ScalarType.h>

template <typename T>
struct TorchScalarType;

template <>
struct TorchScalarType<float> {
  static constexpr at::ScalarType value = at::ScalarType::Float;
};
