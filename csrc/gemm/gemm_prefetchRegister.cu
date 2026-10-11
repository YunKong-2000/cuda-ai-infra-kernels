#include "gemm/gemm.h"
#include "gemm/gemm_layout.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {
constexpr int THREADS = 256;
constexpr int WARPSIZE = 32;
constexpr int WARP_NUM_M = 2;
constexpr int WARP_NUM_N = 4;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int WM = BM / WARP_NUM_M;
constexpr int WN = BN / WARP_NUM_N;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THR_NUM_N = 4;
constexpr int THR_NUM_M = 8;
constexpr int VEC_LEN = 4;
constexpr int WM_SUB = THR_NUM_M * VEC_LEN;
constexpr int WN_SUB = THR_NUM_N * VEC_LEN;
constexpr int STAGES = 2;
using Element = float;
using ElementVec = float4;

static_assert(TM == 8 && TN == 8);
static_assert(THREADS == WARPSIZE * WARP_NUM_M * WARP_NUM_N);
static_assert(THR_NUM_M * THR_NUM_N == WARPSIZE);
static_assert(WM == THR_NUM_M * TM && WN == THR_NUM_N * TN);

__inline__ __device__ void VectorizedAccess(const Element* src, Element* dst)
{
  *reinterpret_cast<ElementVec*>(dst) = *reinterpret_cast<const ElementVec*>(src);
}

__inline__ __device__ void LDS2R(
  Element* sA,
  Element* sB,
  Element* rA,
  Element* rB,
  const int lx,
  const int ly,
  const int k
)
{
  const int M0 = k * BM + ly * VEC_LEN;
  const int N0 = k * BN + lx * VEC_LEN;
  VectorizedAccess(sA + M0, rA);
  VectorizedAccess(sB + N0, rB);
  VectorizedAccess(sA + M0 + WM_SUB , rA + TM / 2);
  VectorizedAccess(sB + N0 + WN_SUB, rB + TN / 2);
}

__inline__ __device__ void innerProduct(const Element* rA, const Element* rB, Element* rAcc) {
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      rAcc[i * TN + j] += rA[i] * rB[j];
    }
  }
}

// Register double buffering: preload row 0, then overlap loading row k+1
// with computing row k. The final row is consumed outside the loop.
__inline__ __device__ void compute
(
  Element* sA,
  Element* sB,
  Element rAcc[TM][TN],
  const int lane_id
)
{
  alignas(16) Element rA[STAGES][TM];
  alignas(16) Element rB[STAGES][TN];
  const int lx = lane_id % THR_NUM_N;
  const int ly = lane_id / THR_NUM_N;
  LDS2R(sA, sB, &rA[0][0], &rB[0][0], lx, ly, 0);
  int stage = 0;
  #pragma unroll
  for (int k = 0; k < BK - 1; k++) {
    LDS2R(sA, sB, &rA[stage ^ 1][0], &rB[stage ^ 1][0], lx, ly, k + 1);
    innerProduct(&rA[stage][0], &rB[stage][0], &rAcc[0][0]);
    stage ^= 1;
  }
  innerProduct(&rA[stage][0], &rB[stage][0], &rAcc[0][0]);
}

__forceinline__ __device__ void CopyAsync16(const Element* src, Element* dst) {
  const unsigned int shared_addr = static_cast<unsigned int>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :: "r"(shared_addr), "l"(src) : "memory");
}

__forceinline__ __device__ void CommitAsyncCopies() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__forceinline__ __device__ void WaitAsyncCopies() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
// global A:(K, M), As:(BK, BM), GD=M, SD=BM
// global B:(K, N), Bs:(BK, BN), GD=N, SD=BN
template<int SD>
inline __device__  void loadSmem(
  const Element* src,
  Element* dst,
  const int GD,
  const int tid) {
  const int VEC_PER_ROW = SD / VEC_LEN;
  const int VEC_NUM = (BK * SD) / VEC_LEN;
  #pragma unroll
  for (int i = tid; i < VEC_NUM; i += THREADS) {
    const int row = i / VEC_PER_ROW;
    const int col = (i % VEC_PER_ROW) * VEC_LEN;
    CopyAsync16(src + int64_t(row) * GD + col, dst + row * SD + col);
  }
}

// Host validation guarantees full tiles and aligned global pointers. All
// device loads/stores are unconditional; there are no element bounds checks.
__global__ void gemm_kernel(
  const int M,
  const int N,
  const int K,
  const Element* A,
  const Element* B,
  Element* C
)
{
  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARPSIZE;
  const int warp_x = warp_id % WARP_NUM_N;
  const int warp_y = warp_id / WARP_NUM_N;
  const int lane_id = tid % WARPSIZE;

  // Compile-time storage alignment, not runtime alignment checks.
  __shared__ __align__(16) Element As[STAGES][BK][BM];
  __shared__ __align__(16) Element Bs[STAGES][BK][BN];
  alignas(16) Element rAcc[TM][TN];

  #pragma unroll
  for (int i = 0; i < TM; i++) {
    #pragma unroll
    for (int j = 0; j < TN; j++) {
      rAcc[i][j] = 0;
    }
  }

  loadSmem<BM>(A + by * BM, &As[0][0][0], M, tid);
  loadSmem<BN>(B + bx * BN, &Bs[0][0][0], N, tid);
  CommitAsyncCopies();
  WaitAsyncCopies();
  __syncthreads();
  int stage = 0;
  const int k_tile_num = K / BK;
  #pragma unroll
  for (int k = 0; k < k_tile_num - 1; k++) {
    loadSmem<BM>(A + int64_t(k + 1) * BK * M + by * BM, &As[stage ^ 1][0][0], M, tid);
    loadSmem<BN>(B + int64_t(k + 1) * BK * N + bx * BN, &Bs[stage ^ 1][0][0], N, tid);
    CommitAsyncCopies();
    compute(&As[stage][0][0] + WM * warp_y, &Bs[stage][0][0] + WN * warp_x, rAcc, lane_id);
    WaitAsyncCopies();
    __syncthreads();
    stage ^= 1;
  }
  // Drain the final tile without a prefetch or a final barrier. For K == BK,
  // the main loop is empty and this computes the initially loaded tile.
  compute(&As[stage][0][0] + WM * warp_y, &Bs[stage][0][0] + WN * warp_x, rAcc, lane_id);

  const int lane_x = lane_id % THR_NUM_N;
  const int lane_y = lane_id / THR_NUM_N;
  #pragma unroll
  for (int a = 0; a < 2; a++) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int b = 0; b < 2; b++) {
        int row = by * BM + warp_y * WM + lane_y * VEC_LEN + a * WM_SUB + i;
        int col = bx * BN + warp_x * WN + lane_x * VEC_LEN + b * WN_SUB;
        VectorizedAccess(&rAcc[a * 4 + i][b * 4], C + int64_t(row) * N + col);
      }
    }
  }
}// kernel

} // namespace

// A is a logical [M,K] column-major tensor; B and C are row-major.
// This fast path intentionally supports full tiles only. Validate all runtime
// layout, size and pointer-alignment requirements here, before launching.
torch::Tensor gemm_prefetchReg(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
  TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0,
              "prefetch_reg requires positive M, N and K");
  check_column_major_gemm_inputs(a, b);
  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));
  TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0,
              "prefetch_reg requires M and N divisible by 128 and K divisible by 8");
  TORCH_CHECK(M / BM <= 65535, "prefetch_reg grid.y exceeds 65535");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(a.data_ptr<float>()) % 16 == 0 &&
              reinterpret_cast<std::uintptr_t>(b.data_ptr<float>()) % 16 == 0,
              "prefetch_reg requires 16-byte aligned A and B pointers");

  const c10::cuda::CUDAGuard device_guard(a.device());
  auto c = torch::empty({M, N}, a.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
  const dim3 blocks(N / BN, M / BM);
  gemm_kernel<<<blocks, THREADS, 0, stream.stream()>>>(
      M, N, K, a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}
