"""CUDA tests for logical column-major A[M,K] and row-major B/C."""

import pytest
import torch

from cuda_ai_kernels import gemm


pytestmark = pytest.mark.cuda
PREFETCH_IMPLS = ("prefetch_smem", "prefetch_reg")


def inputs(m=128, n=128, k=24, device="cuda"):
    return (
        torch.randn(k, m, device=device, dtype=torch.float32).t(),
        torch.randn(k, n, device=device, dtype=torch.float32),
    )


@pytest.mark.parametrize("m,n,k", [
    (128, 128, 8), (128, 128, 16), (128, 128, 24), (128, 128, 32),
    (256, 384, 40), (384, 256, 72), (256, 128, 1024), (1024, 1024, 1024),
])
@pytest.mark.parametrize("impl", PREFETCH_IMPLS)
def test_matches_fp64_and_cublas(m, n, k, impl):
    torch.manual_seed(7)
    a, b = inputs(m, n, k)
    expected = (a.double() @ b.double()).float()
    actual = gemm(a, b, impl=impl)
    baseline = gemm(a, b, impl="cublas")
    assert actual.shape == (m, n) and actual.is_contiguous()
    assert actual.dtype == torch.float32 and actual.device == a.device
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(baseline, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(actual, baseline, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("impl", PREFETCH_IMPLS)
def test_each_k_tile_is_used_once(impl):
    # Distinct exact tile contributions expose duplication and dropped stages.
    a, b = inputs(k=40)
    for tile in range(5):
        a[:, tile * 8:(tile + 1) * 8] = tile + 1
    b.fill_(1)
    expected = torch.full((128, 128), 120.0, device="cuda")
    torch.testing.assert_close(gemm(a, b, impl=impl), expected, atol=0, rtol=0)


@pytest.mark.parametrize("impl", PREFETCH_IMPLS)
@pytest.mark.parametrize("k", [8, 16, 24, 40])
def test_each_k_row_is_used_once(impl, k):
    # Distinct row values catch repeated/missing register-prefetch rows both
    # within a shared-memory tile and across shared-memory stage transitions.
    a, b = inputs(k=k)
    values = torch.arange(1, k + 1, device="cuda", dtype=torch.float32)
    a.copy_(values.unsqueeze(0).expand_as(a))
    b.copy_(values.unsqueeze(1).expand_as(b))
    expected = torch.full((128, 128), k * (k + 1) * (2 * k + 1) / 6, device="cuda")
    torch.testing.assert_close(gemm(a, b, impl=impl), expected, atol=0, rtol=0)


def test_cublas_remains_fp32_when_tf32_is_enabled():
    a, b = inputs()
    a.fill_(1.0001)
    b.fill_(1)
    expected = (a.double() @ b.double()).float()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        actual = gemm(a, b, impl="cublas")
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-6)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("m,n,k", [
    (127, 128, 24), (128, 129, 24), (128, 128, 23),
    (128, 128, 4), (0, 128, 24), (128, 0, 24), (128, 128, 0),
])
@pytest.mark.parametrize("impl", PREFETCH_IMPLS)
def test_rejects_unsupported_sizes(m, n, k, impl):
    a, b = inputs(m, n, k)
    with pytest.raises(RuntimeError, match="positive|divisible"):
        gemm(a, b, impl=impl)


@pytest.mark.parametrize("case,message", [
    ("row_a", "column-major"), ("column_b", "row-major"),
    ("dtype", "float32"), ("rank", "2D"), ("shape", "shape mismatch"),
    ("offset_a", "16-byte aligned"), ("offset_b", "16-byte aligned"),
])
@pytest.mark.parametrize("impl", PREFETCH_IMPLS)
def test_rejects_invalid_inputs(case, message, impl):
    a, b = inputs()
    if case == "row_a":
        a = a.contiguous()
    elif case == "column_b":
        b = b.t().contiguous().t()
    elif case == "dtype":
        a = a.half()
    elif case == "rank":
        a = a.unsqueeze(0)
    elif case == "shape":
        b = b[:-1]
    elif case == "offset_a":
        a = torch.empty(a.numel() + 1, device="cuda")[1:].view(24, 128).t()
    elif case == "offset_b":
        b = torch.empty(b.numel() + 1, device="cuda")[1:].view(24, 128)
    with pytest.raises(RuntimeError, match=message):
        gemm(a, b, impl=impl)


@pytest.mark.parametrize("impl", [*PREFETCH_IMPLS, "cublas"])
def test_aligned_nonzero_storage_offsets(impl):
    a = torch.randn(128 * 24 + 4, device="cuda")[4:].view(24, 128).t()
    b = torch.randn(24 * 128 + 4, device="cuda")[4:].view(24, 128)
    expected = (a.double() @ b.double()).float()
    torch.testing.assert_close(gemm(a, b, impl=impl), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("impl", [*PREFETCH_IMPLS, "cublas"])
def test_current_stream(impl):
    # Queue producers and consumers behind a delay on a non-default stream.
    # Launching GEMM on the default stream would race the input writes.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a, b = inputs(k=40)
        torch.cuda._sleep(5_000_000)
        a.fill_(2)
        b.fill_(3)
        actual = gemm(a, b, impl=impl)
        captured = actual.clone()
    stream.synchronize()
    torch.testing.assert_close(captured, torch.full_like(captured, 240), atol=0, rtol=0)


@pytest.mark.parametrize("impl", [*PREFETCH_IMPLS, "cublas"])
def test_input_device_guard(impl):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    a, b = inputs(device="cuda:1")
    expected = (a.double() @ b.double()).float()
    with torch.cuda.device(0):
        actual = gemm(a, b, impl=impl)
        assert torch.cuda.current_device() == 0
    assert actual.device == a.device
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
