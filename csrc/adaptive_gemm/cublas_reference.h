#pragma once

#include "gemm_problem.h"

void launch_cublas_reference(const GemmProblem& problem);
