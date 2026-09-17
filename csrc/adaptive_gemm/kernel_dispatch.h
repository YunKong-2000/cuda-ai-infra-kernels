#pragma once

#include <cutlass/cutlass.h>

#include "gemm_problem.h"

cutlass::Status dispatch_gemm(const GemmProblem& problem);
