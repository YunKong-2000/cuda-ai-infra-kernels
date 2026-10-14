import pytest
import torch

from cuda_ai_kernels import gemm
from cuda_ai_kernels.gemm_stages import GEMM_STAGES
from cuda_ai_kernels.testing import assert_close

@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_cublas_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="cublas")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_cublas_{shape}", actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 64, 64), (1024, 1024, 1024), (33, 65, 17)])
def test_gemm_tensor_core_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    # torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact comparison
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="tensor_core")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_tensor_core_{shape}", actual, expected, atol=3e-2, rtol=3e-2)

@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_cutlass_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True  # Disable TF32 for exact comparison
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="cutlass")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_cutlass_{shape}", actual, expected, atol=3e-2, rtol=3e-2)


REGULAR_IMPLS = [name for stage, name, _ in GEMM_STAGES if stage <= 5] + ["thread_tile"]
VECTOR_IMPLS = [name for stage, name, _ in GEMM_STAGES if 3 <= stage <= 5]


@pytest.mark.cuda
@pytest.mark.parametrize("impl", ["thread_tile_8x8", "thread_tile_8x8_warp4x8"])
def test_contiguous_thread_tile_output_coordinates(impl):
    # Distinct row/column values expose swapped or duplicated thread/warp tiles,
    # including the transition between adjacent CTAs in both dimensions.
    m, n, k = 256, 384, 24
    rows = torch.arange(1, m + 1, device="cuda", dtype=torch.float32)
    cols = torch.arange(1, n + 1, device="cuda", dtype=torch.float32)
    a = rows.expand(k, m).contiguous().t()
    b = cols.expand(k, n).contiguous()
    expected = k * rows[:, None] * cols[None, :]
    torch.testing.assert_close(gemm(a, b, impl=impl), expected, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("impl", REGULAR_IMPLS)
@pytest.mark.parametrize("shape", [(128, 128, 8), (128, 256, 24), (256, 128, 64)])
def test_regular_gemm_matches_fp64(impl, shape):
    m, n, k = shape
    torch.manual_seed(7)
    a = torch.randn(k, m, device="cuda").t()
    b = torch.randn(k, n, device="cuda")
    actual = gemm(a, b, impl=impl)
    expected = (a.double() @ b.double()).float()
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.parametrize("stage,impl,tile", [s for s in GEMM_STAGES if s[0] <= 5])
def test_regular_gemm_minimum_tile(stage, impl, tile):
    m, n, k = tile
    a = torch.arange(m * k, device="cuda", dtype=torch.float32).view(k, m).t() % 7
    # Explicitly retain column-major strides even if an elementwise op changes them.
    a = a.t().contiguous().t()
    b = torch.arange(k * n, device="cuda", dtype=torch.float32).view(k, n) % 5
    torch.testing.assert_close(gemm(a, b, impl=impl), a @ b, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("impl", REGULAR_IMPLS)
@pytest.mark.parametrize("bad", ["row_major_a", "column_major_b", "partial_m", "partial_n", "empty_k"])
def test_regular_gemm_rejects_bad_inputs(impl, bad):
    m = 127 if bad == "partial_m" else 128
    n = 127 if bad == "partial_n" else 128
    k = 0 if bad == "empty_k" else 16
    a = torch.empty(k, m, device="cuda").t()
    b = torch.empty(k, n, device="cuda")
    if bad == "row_major_a":
        a = a.contiguous()
    if bad == "column_major_b":
        b = b.t().contiguous().t()
    with pytest.raises(RuntimeError):
        gemm(a, b, impl=impl)


@pytest.mark.cuda
@pytest.mark.parametrize("impl", [i for i in REGULAR_IMPLS if i != "naive"])
def test_regular_gemm_rejects_partial_k(impl):
    a = torch.empty(7, 128, device="cuda").t()
    b = torch.empty(7, 128, device="cuda")
    with pytest.raises(RuntimeError, match="divisible"):
        gemm(a, b, impl=impl)


@pytest.mark.cuda
@pytest.mark.parametrize("impl", VECTOR_IMPLS)
@pytest.mark.parametrize("operand", ["a", "b"])
def test_regular_gemm_rejects_misaligned_storage(impl, operand):
    a = torch.empty(16, 128, device="cuda").t()
    b = torch.empty(16, 128, device="cuda")
    offset = torch.empty(16 * 128 + 1, device="cuda")[1:].view(16, 128)
    if operand == "a":
        a = offset.t()
    else:
        b = offset
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        gemm(a, b, impl=impl)


@pytest.mark.cuda
@pytest.mark.parametrize("impl", REGULAR_IMPLS)
def test_regular_gemm_current_stream(impl):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # A delayed producer and a same-stream consumer expose default-stream launches.
        torch.cuda._sleep(2_000_000)
        a = torch.full((24, 128), 2.0, device="cuda").t()
        b = torch.full((24, 256), 3.0, device="cuda")
        actual = gemm(a, b, impl=impl).clone()
    stream.synchronize()
    torch.testing.assert_close(actual, torch.full_like(actual, 144), atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("impl", REGULAR_IMPLS)
def test_regular_gemm_graph_replay(impl):
    a = torch.ones(16, 128, device="cuda").t()
    b = torch.ones(16, 256, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gemm(a, b, impl=impl)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = gemm(a, b, impl=impl)
    for value in (2.0, 3.0):
        a.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, torch.full_like(actual, value * 16), atol=0, rtol=0)
