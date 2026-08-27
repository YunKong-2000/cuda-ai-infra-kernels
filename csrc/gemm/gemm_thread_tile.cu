#include "gemm/gemm.h"
const int BM = 64;
const int BN = 64;
const int BK = 16;
const int BK_HALF = 8;
const int TM = 4;
const int TN = 4;
const int VEC = 4;

__global__ void gemm_thread_tile_scalar_kernel
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  __shared__ float A_tile[BM][BK];
  __shared__ float B_tile[BK][BN];
  float A_vec[TM];
  float B_vec[TN];
  float acc[TM][TN];
  for (int i = 0; i < TM; i ++){
    for (int j = 0; j < TN; j ++) {
      acc[i][j] = 0;
    }
  }
  const int thread_num = blockDim.y * blockDim.x;
  int idx = threadIdx.y * blockDim.x + threadIdx.x;
  int iy;
  int ix;
  constexpr int A_counts = BM * BK;
  constexpr int B_counts = BK * BN;
  const int A_base = blockIdx.y * BM;
  const int B_base = blockIdx.x * BN;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    //load A and B from global to shared memory.
    // A
    for (int i = idx; i < A_counts; i += thread_num) {
      iy = i / BK;
      ix = i % BK;
      A_tile[iy][ix] = 
      (A_base + iy < M && k_iter + ix < K) ?
      A[(A_base + iy) * K + k_iter + ix] : 0;
    }
    // B
    for (int i = idx; i < B_counts; i += thread_num) {
      iy = i / BN;
      ix = i % BN;
      B_tile[iy][ix] = 
      (k_iter + iy < K && B_base + ix < N) ?
      B[(k_iter + iy) * N + B_base + ix] : 0;
    }
    __syncthreads();

    for (int k = 0; k < BK ; k++ ){
      // load A and B from shared to register
      for (int i = 0; i < TM ; i++) {
        A_vec[i] = A_tile[ty * TM + i][k];
      }
      for (int i = 0; i < TN; i++) {
        B_vec[i] = B_tile[k][tx * TN + i];
      }
      for (int i = 0; i < TM; i ++) {
        for (int j = 0; j < TN; j ++) {
          acc[i][j] += A_vec[i] * B_vec[j];
        }
      }
    }
    __syncthreads();
  }
  
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j ++) {
      if (A_base + ty * TM + i < M && B_base + tx * TN + j < N) {
        C[(A_base + ty * TM + i) * N + B_base + tx * TN + j] = acc[i][j];
      }
    }
  }
}

__global__ void gemm_thread_tile_float4_kernel
(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  __shared__ float A_tile[BM][BK];
  __shared__ float __align__(16)  B_tile[BK][BN];
  float A_vec[TM];
  float B_vec[TN];
  float acc[TM][TN];
  for (int i = 0; i < TM; i ++){
    for (int j = 0; j < TN; j ++) {
      acc[i][j] = 0;
    }
  }
  const int thread_num = blockDim.y * blockDim.x;
  int idx = threadIdx.y * blockDim.x + threadIdx.x;
  int iy;
  int ix;
  constexpr int A_VEC_counts = BM * BK / VEC;
  constexpr int B_VEC_counts = BK * BN / VEC;
  const int A_base = blockIdx.y * BM;
  const int B_base = blockIdx.x * BN;
  for (int k_iter = 0; k_iter < K; k_iter += BK) {
    //load A and B from global to shared memory.
    // A
    for (int i = idx; i < A_VEC_counts; i += thread_num ) {
      iy = i / (BK / VEC);
      ix = i % (BK / VEC) * VEC;
      if (A_base + iy < N && k_iter + ix + 3 < K) {
        float4 value = *reinterpret_cast<const float4*>(&A[(A_base + iy) * K + k_iter + ix]);
        *reinterpret_cast<float4*>(&A_tile[iy][ix]) = value;
      }
      else {
        for (int k = 0; k < VEC; k++) {
          A_tile[iy][ix + k] = 
            (A_base + iy < M && k_iter + ix + k < K) ?
            A[(A_base + iy) * K + k_iter + ix + k] : 0;
        }
      }
    }
    // B
    for (int i = idx; i < B_VEC_counts; i += thread_num) {
      iy = i / (BN / VEC);
      ix = i % (BN / VEC) * 4;
      if (k_iter + iy < K && B_base + ix + 3 < N) {
        float4 value = *reinterpret_cast<const float4*>(&B[(k_iter + iy) * N + B_base + ix]);
        *reinterpret_cast<float4*>(&B_tile[iy][ix]) = value;
      }
      else {
        for (int k = 0; k < VEC; k++) {
          B_tile[iy][ix + k] = 
          (k_iter + iy < K && B_base + ix + k < N) ?
          B[(k_iter + iy) * N + B_base + ix + k] : 0;
        }
      }
      
    }
    __syncthreads();

    for (int k = 0; k < BK ; k++ ){
      // load A and B from shared to register
      for (int i = 0; i < TM ; i++) {
        A_vec[i] = A_tile[ty * TM + i][k];
      }
      for (int i = 0; i < TN; i++) {
        B_vec[i] = B_tile[k][tx * TN + i];
      }
      for (int i = 0; i < TM; i ++) {
        for (int j = 0; j < TN; j ++) {
          acc[i][j] += A_vec[i] * B_vec[j];
        }
      }
    }
    __syncthreads();
  }
  
  for (int i = 0; i < TM; i++) {
    if (A_base + ty * TM + i < M && B_base + tx * TN + 3 < N) {
      float4 value = *reinterpret_cast<float4*>(&acc[i][0]);
      *reinterpret_cast<float4*>(&C[(A_base + ty * TM + i) * N + B_base + tx * TN]) = value;
    }
    else {
      for (int k = 0; k < VEC; k++) {
        if (A_base + ty * TM + i < M && B_base + tx * TN + k < N) {
          C[(A_base + ty * TM + i) * N + B_base + tx * TN + k] = acc[i][k];
        }
      }
    }
  }
}

__device__ __forceinline__
float4 load_float4_safe(
    const float* src,
    int row,
    int col,
    int rows,
    int cols)
{
  float4 value = make_float4(0.f, 0.f, 0.f, 0.f);

  if (row >= rows || col >= cols) {
    return value;
  }

  const float* ptr = src + row * cols + col;

  if (col + 4 <= cols &&
      (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0) {
    return *reinterpret_cast<const float4*>(ptr);
  }

  const int remain = cols - col;

  if (remain >= 1) value.x = ptr[0];
  if (remain >= 2) value.y = ptr[1];
  if (remain >= 3) value.z = ptr[2];

  return value;
}

__device__ __forceinline__
float4 load_stage_task(
    const float* A,
    const float* B,
    int task,
    int k_base,
    int A_base,
    int B_base,
    int M,
    int N,
    int K)
{
  constexpr int A_TASKS = BM * BK_HALF / VEC;

  if (task < A_TASKS) {
    const int row_in_tile = task / (BK_HALF / VEC);
    const int col_in_tile =
        (task % (BK_HALF / VEC)) * VEC;

    return load_float4_safe(
        A,
        A_base + row_in_tile,
        k_base + col_in_tile,
        M,
        K);
  }

  const int b_task = task - A_TASKS;
  const int row_in_tile = b_task / (BN / VEC);
  const int col_in_tile =
      (b_task % (BN / VEC)) * VEC;

  return load_float4_safe(
      B,
      k_base + row_in_tile,
      B_base + col_in_tile,
      K,
      N);
}

__device__ __forceinline__
void store_stage_task(
    float4 value,
    int task,
    float* A_tile,
    float* B_tile)
{
  constexpr int A_TASKS = BM * BK_HALF / VEC;

  if (task < A_TASKS) {
    const int row = task / (BK_HALF / VEC);
    const int col =
        (task % (BK_HALF / VEC)) * VEC;

    *reinterpret_cast<float4*>(
        A_tile + row * BK_HALF + col) = value;
  } else {
    const int b_task = task - A_TASKS;
    const int row = b_task / (BN / VEC);
    const int col =
        (b_task % (BN / VEC)) * VEC;

    *reinterpret_cast<float4*>(
        B_tile + row * BN + col) = value;
  }
}

__device__ __forceinline__
void compute_stage(
    float acc[TM][TN],
    const float* A_tile,
    const float* B_tile,
    int ty,
    int tx)
{
  #pragma unroll
  for (int k = 0; k < BK_HALF; ++k) {
    float A_vec[TM];
    float B_vec[TN];

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      A_vec[i] =
          A_tile[(ty * TM + i) * BK_HALF + k];
    }

    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      B_vec[j] =
          B_tile[k * BN + tx * TN + j];
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        acc[i][j] =
            fmaf(A_vec[i], B_vec[j], acc[i][j]);
      }
    }
  }
}

__global__ __launch_bounds__(256, 4)
void gemm_thread_tile_pipeline_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
  static_assert(VEC == 4);
  static_assert(TN == 4);
  static_assert(BK_HALF % VEC == 0);
  static_assert(BN % VEC == 0);

  constexpr int A_TASKS = BM * BK_HALF / VEC;
  constexpr int B_TASKS = BK_HALF * BN / VEC;
  constexpr int TOTAL_TASKS = A_TASKS + B_TASKS;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int task = ty * blockDim.x + tx;

  const int A_base = blockIdx.y * BM;
  const int B_base = blockIdx.x * BN;

  __shared__ __align__(16)
      float A_tile[2][BM * BK_HALF];

  __shared__ __align__(16)
      float B_tile[2][BK_HALF * BN];

  float acc[TM][TN];

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = 0.f;
    }
  }

  const int num_stages =
      (K + BK_HALF - 1) / BK_HALF;

  if (num_stages > 0) {
    // Prologue: load stage 0.
    if (task < TOTAL_TASKS) {
      const float4 value = load_stage_task(
          A, B, task, 0,
          A_base, B_base,
          M, N, K);

      store_stage_task(
          value,
          task,
          A_tile[0],
          B_tile[0]);
    }

    __syncthreads();

    // Stable pipeline.
    for (int next_stage = 1;
         next_stage < num_stages;
         ++next_stage) {
      const int read_buffer = (next_stage - 1) & 1;
      const int write_buffer = next_stage & 1;

      /*
       * This is the only value intentionally kept alive across
       * compute_stage().
       */
      float4 prefetched = make_float4(
          0.f, 0.f, 0.f, 0.f);

      if (task < TOTAL_TASKS) {
        prefetched = load_stage_task(
            A,
            B,
            task,
            next_stage * BK_HALF,
            A_base,
            B_base,
            M,
            N,
            K);
      }

      compute_stage(
          acc,
          A_tile[read_buffer],
          B_tile[read_buffer],
          ty,
          tx);

      if (task < TOTAL_TASKS) {
        store_stage_task(
            prefetched,
            task,
            A_tile[write_buffer],
            B_tile[write_buffer]);
      }

      __syncthreads();
    }

    // Drain the final stage.
    const int last_buffer = (num_stages - 1) & 1;

    compute_stage(
        acc,
        A_tile[last_buffer],
        B_tile[last_buffer],
        ty,
        tx);
  }

  // Store C.
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = A_base + ty * TM + i;
    const int col = B_base + tx * TN;

    if (row >= M || col >= N) {
      continue;
    }

    float* dst = C + row * N + col;

    if (col + VEC <= N &&
        (reinterpret_cast<uintptr_t>(dst) & 0xF) == 0) {
      const float4 value = make_float4(
          acc[i][0],
          acc[i][1],
          acc[i][2],
          acc[i][3]);

      *reinterpret_cast<float4*>(dst) = value;
    } else {
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        if (col + j < N) {
          dst[j] = acc[i][j];
        }
      }
    }
  }
}

torch::Tensor gemm_thread_tile(torch::Tensor a, torch::Tensor b) {
  check_gemm_inputs(a, b);
  const int m = static_cast<int>(a.size(0));
  const int k = static_cast<int>(a.size(1));
  const int n = static_cast<int>(b.size(1));

  torch::Tensor c = torch::zeros({m, n}, a.options());
  dim3 block(16, 16); 
  dim3 grid(ceil_div(n, BN), ceil_div(m, BM));
  if (n % VEC == 0 && k % VEC == 0) {
    static_assert(VEC == 4);
    static_assert(TN == VEC);
    static_assert(BK_HALF % VEC == 0);
    static_assert(BN % VEC == 0);
    static_assert(BM == 16 * TM);
    static_assert(BN == 16 * TN);
    gemm_thread_tile_pipeline_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
  }
  else {
    gemm_thread_tile_scalar_kernel<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
  return c;
}
