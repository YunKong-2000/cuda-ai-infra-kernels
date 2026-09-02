#include "common/tensor_check.h"
#include "gemm/gemm.h"

const int BM = 64;
const int BN = 64;
const int BK = 16;
const int WM = 16;
const int WN = 32;
const int WX = 2;
constexpr int WARPS_M = BM / WM;
constexpr int WARPS_N = BN / WN;
const int TM = 4;
const int TN = 4;
const int TX = 8;//warp size: 4*8
const int VEC = 4;
constexpr int A_VEC_COL = BK / VEC;
constexpr int B_VEC_COL = BN / VEC;

__global__ void gemm_warp_tile_kernel
(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const int M,
  const int N,
  const int K
)
{
  const int by = blockIdx.y;
  const int bx = blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / warpSize;
  const int warp_y = warp_id / WX;
  const int warp_x = warp_id % WX;
  const int lane_id = tid % warpSize;
  const int lane_y = lane_id / TX;
  const int lane_x = lane_id % TX;
  __shared__ float __align__(16) A_smem[BM][BK];
  __shared__ float __align__(16) B_smem[BK][BN];
  float A_vec[TM];
  float B_vec[TN];
  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; i++) {
    for (int j =0; j < TN; j++) {
      acc[i][j] = 0;
    }
  }
  int A_VEC_NUM = BM * BK / VEC;
  int total_VEC_NUM = A_VEC_NUM + BK * BN / VEC;

  #pragma unroll
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    // GTS
    if (tid < A_VEC_NUM) {
      int iy = tid / A_VEC_COL;
      int ix = (tid % A_VEC_COL) * VEC;
      if (k_iter + ix + VEC <= K && by * BM + iy < M) {
        float4 value = *reinterpret_cast<const float4*>(&A[(by * BM + iy) * K + k_iter + ix]);
        *reinterpret_cast<float4*>(&A_smem[iy][ix]) = value;
      }
      else {
        for (int i = 0; i < VEC; i++) {
          A_smem[iy][ix + i] = (by * BM + iy < M && k_iter + ix + i < K) ? A[(by * BM + iy) * K + k_iter + ix + i] : 0;
        }
      }
    }
    else if (tid < total_VEC_NUM) {
      int idx = tid - A_VEC_NUM;
      int iy = idx / B_VEC_COL;
      int ix = (idx % B_VEC_COL) * VEC;
      if (k_iter + iy < K && bx * BN + ix + VEC <= N) {
        float4 value = *reinterpret_cast<const float4*>(&B[(k_iter + iy) * N + bx * BN + ix]);
        *reinterpret_cast<float4*>(&B_smem[iy][ix]) = value;
      }
      else {
        for (int i = 0; i < VEC; i++) {
          B_smem[iy][ix + i] = (bx * BN + ix + i < N && k_iter + iy < K) ? B[(k_iter + iy) * N + bx * BN + ix + i] : 0;
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; k++) {
      #pragma unroll
      for (int i = 0; i < TM; i++) {
        A_vec[i] = A_smem[warp_y * WM + i * (WM / TM) + lane_y][k];
      }
      #pragma unroll
      for (int j = 0; j < TN; j++) {
        B_vec[j] = B_smem[k][warp_x * WN + lane_x * TN + j];
      }
      #pragma unroll
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          acc[i][j] += A_vec[i] * B_vec[j];
        }
      }
    }
    __syncthreads();
  }

  const int row_base = by * BM + warp_y * WM;
  const int col = bx * BN + warp_x * WN + lane_x * TN;

  if (col < N) {
    #pragma unroll
    for (int i = 0; i < TM; i++) {
      const int output_row =
      row_base + i * (WM / TM) + lane_y;
      if (output_row >= M) {
        break;
      }

      if (col + TN <= N) {
        float4 value = make_float4(
            acc[i][0],
            acc[i][1],
            acc[i][2],
            acc[i][3]);

        *reinterpret_cast<float4*>(
            &C[output_row * N + col]) = value;
      } else {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
          if (col + j < N) {
            C[output_row * N + col + j] = acc[i][j];
          }
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 4)
void gemm_warp_tile1_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K)
{
  // 要求 blockDim = dim3(256, 1, 1)
  const int tid = threadIdx.x;

  const int warp_id = tid >> 5;
  const int warp_y = warp_id >> 1;  // WX == 2
  const int warp_x = warp_id & 1;

  const int lane_id = tid & 31;
  const int lane_y = lane_id >> 3;  // TX == 8
  const int lane_x = lane_id & 7;

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  __shared__ __align__(16) float A_smem[BM][BK + 4];
  __shared__ __align__(16) float B_smem[BK][BN];

  // 删除 A_vec[TM]，只保留B的4个值
  float b0, b1, b2, b3;
  float acc[TM][TN];

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  constexpr int A_VEC_NUM = BM * BK / VEC;
  constexpr int B_VEC_NUM = BK * BN / VEC;

  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    #pragma unroll
    for (int idx = tid; idx < A_VEC_NUM; idx += blockDim.x) {
      const int iy = idx / A_VEC_COL;
      const int ix = (idx % A_VEC_COL) * VEC;
      const int global_row = block_row + iy;

      // K为4的倍数时，所有A行都满足float4对齐
      if ((K % VEC == 0) &&
          global_row < M &&
          k_iter + ix + VEC <= K) {
        const float4 value =
            *reinterpret_cast<const float4*>(
                &A[global_row * K + k_iter + ix]);

        *reinterpret_cast<float4*>(
            &A_smem[iy][ix]) = value;
      } else {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const int global_col = k_iter + ix + i;

          A_smem[iy][ix + i] =
              (global_row < M && global_col < K)
                  ? A[global_row * K + global_col]
                  : 0.0f;
        }
      }
    }

    for (int idx = tid; idx < B_VEC_NUM; idx += blockDim.x) {
      const int iy = idx / B_VEC_COL;
      const int ix = (idx % B_VEC_COL) * VEC;
      const int global_row = k_iter + iy;
      const int global_col = block_col + ix;

      // N为4的倍数时，所有B行都满足float4对齐
      if ((N % VEC == 0) &&
          global_row < K &&
          global_col + VEC <= N) {
        const float4 value =
            *reinterpret_cast<const float4*>(
                &B[global_row * N + global_col]);

        *reinterpret_cast<float4*>(
            &B_smem[iy][ix]) = value;
      } else {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
          B_smem[iy][ix + i] =
              (global_row < K && global_col + i < N)
                  ? B[global_row * N + global_col + i]
                  : 0.0f;
        }
      }
    }

    __syncthreads();

    const int warp_row = warp_y * WM;
    const int warp_col = warp_x * WN;
    const int thread_col = warp_col + lane_x * TN;

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      // B只读取一次，供4个A标量复用
      b0 = B_smem[k][thread_col + 0];
      b1 = B_smem[k][thread_col + 1];
      b2 = B_smem[k][thread_col + 2];
      b3 = B_smem[k][thread_col + 3];

      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        // A只保留一个标量，不再同时保存A_vec[4]
        const float a =
            A_smem[
                warp_row +
                i * (WM / TM) +
                lane_y][k];

        acc[i][0] = fmaf(a, b0, acc[i][0]);
        acc[i][1] = fmaf(a, b1, acc[i][1]);
        acc[i][2] = fmaf(a, b2, acc[i][2]);
        acc[i][3] = fmaf(a, b3, acc[i][3]);
      }
    }

    __syncthreads();
  }

  const int row_base =
      block_row + warp_y * WM;

  const int col =
      block_col + warp_x * WN + lane_x * TN;

  if (col < N) {
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int output_row =
          row_base + i * (WM / TM) + lane_y;

      if (output_row >= M) {
        break;
      }

      // N为4的倍数时，C的每一行均满足float4对齐
      if ((N % VEC == 0) && col + TN <= N) {
        const float4 value = make_float4(
            acc[i][0],
            acc[i][1],
            acc[i][2],
            acc[i][3]);

        *reinterpret_cast<float4*>(
            &C[output_row * N + col]) = value;
      } else {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
          if (col + j < N) {
            C[output_row * N + col + j] =
                acc[i][j];
          }
        }
      }
    }
  }
}

__device__ __forceinline__
void cp_async_16(void* smem_ptr, const void* global_ptr, bool valid)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  const unsigned smem_addr =
      static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));

  const int src_size = valid ? 16 : 0;

  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      :
      : "r"(smem_addr),
        "l"(global_ptr),
        "r"(src_size));
#endif
}

__device__ __forceinline__
void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__
void cp_async_wait_all()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;\n");
#endif
}

template <int BM_, int BN_, int BK_, int WM_, int WN_, int TM_, int TN_>
struct GemmConfig {
  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int BK = BK_;
  static constexpr int WM = WM_;
  static constexpr int WN = WN_;
  static constexpr int TM = TM_;
  static constexpr int TN = TN_;
  static constexpr int VEC = 4;
  static constexpr int WARPS_M = BM / WM;
  static constexpr int WARPS_N = BN / WN;
  static constexpr int LANES_M = WM / TM;
  static constexpr int LANES_N = WN / TN;
  static constexpr int THREADS = WARPS_M * WARPS_N * 32;
};

// Change only this alias to benchmark another block/warp/thread layout.
using ActiveGemmConfig = GemmConfig<128, 64, 16, 32, 32, 8, 4>;

template <typename Config>
__device__ __forceinline__
void prefetch_gemm_tile(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* A_stage,
    float* B_stage,
    int tid,
    int block_row,
    int block_col,
    int k_begin,
    int M,
    int N,
    int K)
{
  constexpr int A_VEC_COL = Config::BK / Config::VEC;
  constexpr int B_VEC_COL = Config::BN / Config::VEC;
  constexpr int A_VEC_COUNT = Config::BM * Config::BK / Config::VEC;
  constexpr int B_VEC_COUNT = Config::BK * Config::BN / Config::VEC;

  for (int index = tid; index < A_VEC_COUNT; index += Config::THREADS) {
    const int iy = index / A_VEC_COL;
    const int ix = (index % A_VEC_COL) * Config::VEC;

    const int global_row = block_row + iy;
    const int global_col = k_begin + ix;

    const bool valid =
        global_row < M &&
        global_col + Config::VEC <= K;

    const int swizzle_mask =
        ((iy & 3) * Config::VEC) & (Config::BK - 1);

    const int swizzled_ix =
        ix ^ swizzle_mask;

    const float* global_src = valid
        ? &A[static_cast<size_t>(global_row) * K + global_col]
        : A;

    float* shared_dst =
        &A_stage[iy * Config::BK + swizzled_ix];

    cp_async_16(
        shared_dst,
        global_src,
        valid ? 16 : 0);
  }

  for (int index = tid; index < B_VEC_COUNT; index += Config::THREADS) {
    const int iy = index / B_VEC_COL;
    const int ix = (index % B_VEC_COL) * Config::VEC;

    const int global_row = k_begin + iy;
    const int global_col = block_col + ix;

    const bool valid =
        global_row < K &&
        global_col + Config::VEC <= N;

    const float* global_src = valid
        ? &B[static_cast<size_t>(global_row) * N + global_col]
        : B;

    float* shared_dst =
        &B_stage[iy * Config::BN + ix];

    cp_async_16(
        shared_dst,
        global_src,
        valid ? 16 : 0);
  }

  cp_async_commit();
}

template <typename Config>
__global__
void gemm_warp_tile_double_buffer_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
  static_assert(Config::BM % Config::WM == 0, "BM must be divisible by WM");
  static_assert(Config::BN % Config::WN == 0, "BN must be divisible by WN");
  static_assert(Config::WM % Config::TM == 0, "WM must be divisible by TM");
  static_assert(Config::WN % Config::TN == 0, "WN must be divisible by TN");
  static_assert(Config::LANES_M * Config::LANES_N == 32,
                "one warp must map to exactly 32 thread tiles");
  static_assert(Config::BK >= Config::VEC && (Config::BK & (Config::BK - 1)) == 0,
                "BK must be a power of two and at least 4");
  static_assert(Config::BN % Config::VEC == 0 && Config::TN % Config::VEC == 0,
                "BN and TN must be multiples of float4");
  static_assert(Config::THREADS <= 1024, "CUDA blocks support at most 1024 threads");

  const int tid = threadIdx.x;

  const int warp_id = tid >> 5;
  const int warp_y = warp_id / Config::WARPS_N;
  const int warp_x = warp_id % Config::WARPS_N;

  const int lane_id = tid & 31;
  const int lane_y = lane_id / Config::LANES_N;
  const int lane_x = lane_id % Config::LANES_N;

  const int block_row = blockIdx.y * Config::BM;
  const int block_col = blockIdx.x * Config::BN;

  __shared__ __align__(16)
      float A_smem[2][Config::BM][Config::BK];

  __shared__ __align__(16)
      float B_smem[2][Config::BK][Config::BN];

  float acc[Config::TM][Config::TN];

  #pragma unroll
  for (int i = 0; i < Config::TM; ++i) {
    #pragma unroll
    for (int j = 0; j < Config::TN; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  prefetch_gemm_tile<Config>(
      A,
      B,
      &A_smem[0][0][0],
      &B_smem[0][0][0],
      tid,
      block_row,
      block_col,
      0,
      M,
      N,
      K);

  cp_async_wait_all();
  __syncthreads();

  const int warp_row =
      warp_y * Config::WM;

  const int thread_col =
      warp_x * Config::WN + lane_x * Config::TN;

  int current_buffer = 0;

  for (int k_iter = 0; k_iter < K; k_iter += Config::BK) {
    const int next_k =
        k_iter + Config::BK;

    const int next_buffer =
        current_buffer ^ 1;

    const bool has_next =
        next_k < K;

    if (has_next) {
      prefetch_gemm_tile<Config>(
          A,
          B,
          &A_smem[next_buffer][0][0],
          &B_smem[next_buffer][0][0],
          tid,
          block_row,
          block_col,
          next_k,
          M,
          N,
          K);
    }

    #pragma unroll
    for (int k = 0; k < Config::BK; ++k) {
      float b[Config::TN];
      #pragma unroll
      for (int j = 0; j < Config::TN; j += Config::VEC) {
        const float4 values = *reinterpret_cast<const float4*>(
            &B_smem[current_buffer][k][thread_col + j]);
        b[j] = values.x;
        b[j + 1] = values.y;
        b[j + 2] = values.z;
        b[j + 3] = values.w;
      }

      #pragma unroll
      for (int i = 0; i < Config::TM; ++i) {
        const int row =
            warp_row +
            i * Config::LANES_M +
            lane_y;

        // 必须使用与写入阶段相同的swizzle。
        const int swizzled_k =
            k ^ (((row & 3) * Config::VEC) & (Config::BK - 1));

        const float a =
            A_smem[current_buffer][row][swizzled_k];

        #pragma unroll
        for (int j = 0; j < Config::TN; ++j) {
          acc[i][j] = fmaf(a, b[j], acc[i][j]);
        }
      }
    }

    if (has_next) {
      cp_async_wait_all();
      __syncthreads();

      current_buffer = next_buffer;
    }
  }

  const int row_base =
      block_row + warp_y * Config::WM;

  const int output_col =
      block_col +
      warp_x * Config::WN +
      lane_x * Config::TN;

  if (output_col < N) {
    #pragma unroll
    for (int i = 0; i < Config::TM; ++i) {
      const int output_row =
          row_base +
          i * Config::LANES_M +
          lane_y;

      if (output_row >= M) {
        break;
      }

      #pragma unroll
      for (int j = 0; j < Config::TN; j += Config::VEC) {
        const float4 result = make_float4(
            acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
        *reinterpret_cast<float4*>(
            &C[static_cast<size_t>(output_row) * N + output_col + j]) = result;
      }
    }
  }
}

torch::Tensor gemm_warp_tile(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  torch::Tensor c = torch::zeros({m, n}, a.options());
  using Config = ActiveGemmConfig;
  dim3 block(Config::THREADS);
  dim3 grid(
    ceil_div(n, Config::BN),
    ceil_div(m, Config::BM));
  if (n % Config::TN == 0 && k % Config::VEC == 0) {
    gemm_warp_tile_double_buffer_kernel<Config><<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
  }
  else {
  }
  CUDA_CHECK(cudaGetLastError());
  return c;
}
