#pragma once

#include <cutlass/cutlass.h>

#include "kernel_entry.h"

cutlass::Status dispatch_gemm(
  const GemmProblem& problem,
  KernelId requested_kernel = KernelId::Auto);
