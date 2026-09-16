#include "adaptive_gemm.h"
#include "gemm_problem.h"
#include "kernel_sm80.h"


struct KernelEntry{
  KernelId id;
  const char* name;
  KernelMeta meta;
  LaunchFn launch;
};