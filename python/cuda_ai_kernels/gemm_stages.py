"""Ordered FP32 optimization ladder; tile sizes are (BM, BN, BK)."""

# Separate template instantiations and Python entries for every tested variant.
GEMM_STAGES = (
    (1, "naive", (16, 16, 1)),
    (2, "block_tile", (16, 16, 8)),
    (3, "block_tile_float4_16x16", (16, 16, 8)),
    (3, "block_tile_float4_16x32", (16, 32, 8)),
    (3, "block_tile_float4_32x32", (32, 32, 8)),
    (4, "thread_tile_4x4", (128, 128, 8)),
    (4, "thread_tile_4x8", (128, 128, 8)),
    (4, "thread_tile_8x8", (128, 128, 8)),
    (5, "warp_tile", (128, 128, 8)),
    (6, "prefetch_smem", (128, 128, 8)),
    (7, "prefetch_reg", (128, 128, 8)),
    (8, "split_k", (128, 128, 16)),
    (9, "cublas", (1, 1, 1)),
)
GEMM_STAGE_IMPLS = tuple(name for _, name, _ in GEMM_STAGES)
COLUMN_MAJOR_IMPLS = frozenset(GEMM_STAGE_IMPLS) | {"thread_tile"}
