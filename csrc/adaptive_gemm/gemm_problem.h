#pragma once
#include "adaptive_gemm.h"
#include <ATen/core/ScalarType.h>

enum class EpilogueKind{
  ReLu,
  GeLu,
  Bias,
  Linear,
  BiasRelu,
  BiasGelu,
  Residual,
  BiasResidualGelu,
}

struct EpilogueDesc{
  EpilogueKind kind;
  float alpha;
  float beta;
  const void* bias;
  const void* residual;
}

struct GemmProblem{
  int M;
  int N;
  int K;

  const void* a;
  const void* b;
  bool has_c;
  const void* c;
  bool has_bias;
  const void* bias;
  void* d;

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t ldd;

  at::ScalarType dtype_a;
  at::ScalarType dtype_b;
  at::ScalarType dtype_c;
  at::ScalarType dtype_d;

  EpilogueDesc epilogue;
  cudaStream_t stream;
  GemmProblem(torch::Tensor a, torch::Tensor a);
}

GemmProblem(torch::Tensor a, torch::Tensor b, torch::Tensor d, cudaStream_t cur_stream) {
  this->M = a.size(0);
  this->N = b.size(1);
  this->K = a.size(1);
  this->dtype_a = a.dtype;
  this->dtype_b = b.dtype;
  this->dtype_d = d.dtype;
  this->a = a.data_ptr<a.dtype>();
  this->b = b.data_ptr<b.dtype>();
  this->d = d.data_ptr<d.dtype>();
  this->lda = a.size(1);
  this->ldb = b.size(1);
  this->ldd = d.size(1);
  this->stream = cur_stream;
}