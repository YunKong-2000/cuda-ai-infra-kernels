#include <torch/extension.h>

#include "attention/attention_decode.h"
#include "gemm/gemm.h"
#include "rmsnorm/rmsnorm.h"
#include "rope/rope.h"
#include "softmax/softmax.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_naive", &gemm_naive, "Naive FP32 GEMM");
  m.def("gemm_cublas", &gemm_cublas, "cuBLAS FP32 GEMM");
  m.def("gemm_block_tile", &gemm_block_tile, "Shared-memory tiled FP32 GEMM");
  m.def("gemm_thread_tile", &gemm_thread_tile, "Thread-tiled FP32 GEMM");
  m.def("gemm_warp_tile", &gemm_warp_tile, "Warp-level FP32 GEMM");
  m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward");
  m.def("softmax_forward", &softmax_forward, "Softmax forward");
  m.def("rope_forward", &rope_forward, "RoPE forward");
  m.def("attention_decode_forward", &attention_decode_forward, "Attention decode forward");
}
