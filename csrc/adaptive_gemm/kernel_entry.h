#pragma once

#include <cutlass/cutlass.h>

#include "gemm_problem.h"

enum class KernelId {
  Auto,
  Cublas,
  Fast128x128Stage4FP32,
  Fallback128x128Stage4FP32,
  Fast128x128Stage4FP32MWarp,
  Fallback128x128Stage4FP32MWarp,
  Fast128x128Stage3FP32,
  Fast128x128Stage2FP32,
  Fast128x128Stage4FP32Relu,
  Fallback128x128Stage4FP32Relu,
};

struct KernelMeta {
  // element data type
  at::ScalarType dtype_a;
  at::ScalarType dtype_b;
  at::ScalarType dtype_d;

  // Threadblock tile
  int threadblock_m;
  int threadblock_n;
  int threadblock_k;

  // Warp tile
  int warp_m;
  int warp_n;
  int warp_k;

  // MMA instruction
  int instruction_m;
  int instruction_n;
  int instruction_k;

  // Pipeline
  int stages;

  // number of elements
  int alignment_a;
  int alignment_b;
  int alignment_c;

  // capacity
  bool supports_split_k;
};

using LaunchFn = cutlass::Status (*)(const GemmProblem&);

struct KernelEntry {
  KernelId id;
  const char* name;
  EpilogueKind epilogue_kind;
  KernelMeta meta;
  LaunchFn launch;
};
