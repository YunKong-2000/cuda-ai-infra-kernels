"""Regression tests for two K slices reduced into a zeroed output."""

import pytest
import torch

from cuda_ai_kernels import gemm


pytestmark = pytest.mark.cuda


def inputs(m=128, n=128, k=32):
    return (
        torch.randn(k, m, device="cuda", dtype=torch.float32).t(),
        torch.randn(k, n, device="cuda", dtype=torch.float32),
    )


@pytest.mark.parametrize("m,n,k", [
    (128, 128, 16), (128, 256, 32), (256, 640, 48),
    (384, 256, 64), (256, 128, 1024),
])
def test_matches_fp64(m, n, k):
    # Cover one tile per slice, both final stages, and multiple output tiles.
    torch.manual_seed(7)
    a, b = inputs(m, n, k)
    expected = (a.double() @ b.double()).float()
    actual = gemm(a, b, impl="split_k")
    assert actual.shape == (m, n) and actual.is_contiguous()
    assert actual.dtype == torch.float32 and actual.device == a.device
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("n", [128, 256, 640])
@pytest.mark.parametrize("k", [16, 32, 48, 64])
def test_each_k_row_is_used_once(n, k):
    a, b = inputs(n=n, k=k)
    values = torch.arange(1, k + 1, device="cuda", dtype=torch.float32)
    a.copy_(values.unsqueeze(0).expand_as(a))
    b.fill_(1)
    expected = torch.full((128, n), k * (k + 1) / 2, device="cuda")
    torch.testing.assert_close(
        gemm(a, b, impl="split_k"), expected, atol=0, rtol=0)


@pytest.mark.parametrize("m,n,k", [
    (128, 128, 8), (128, 128, 24), (128, 128, 40),
    (127, 128, 32), (128, 129, 32),
    (0, 128, 32), (128, 0, 32), (128, 128, 0),
])
def test_rejects_incomplete_tiles(m, n, k):
    a, b = inputs(m, n, k)
    with pytest.raises(RuntimeError, match="positive|divisible"):
        gemm(a, b, impl="split_k")


@pytest.mark.parametrize("case,message", [
    ("row_a", "column-major"), ("column_b", "row-major"),
    ("offset_a", "16-byte aligned"), ("offset_b", "16-byte aligned"),
])
def test_rejects_invalid_layout(case, message):
    a, b = inputs()
    if case == "row_a":
        a = a.contiguous()
    elif case == "column_b":
        b = b.t().contiguous().t()
    elif case == "offset_a":
        a = torch.empty(a.numel() + 1, device="cuda")[1:].view(32, 128).t()
    elif case == "offset_b":
        b = torch.empty(b.numel() + 1, device="cuda")[1:].view(32, 128)
    with pytest.raises(RuntimeError, match=message):
        gemm(a, b, impl="split_k")


def test_output_is_reset_between_calls():
    a, b = inputs(n=256)
    b.fill_(1)
    for value in (2, 0, 3, 0):
        a.fill_(value)
        actual = gemm(a, b, impl="split_k")
        torch.testing.assert_close(
            actual, torch.full_like(actual, value * 32), atol=0, rtol=0)
        # Return dirty output storage to the allocator before the next call.
        actual.fill_(12345)
        del actual


def test_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a, b = inputs(n=256, k=48)
        torch.cuda._sleep(5_000_000)
        a.fill_(2)
        b.fill_(3)
        actual = gemm(a, b, impl="split_k").clone()
    stream.synchronize()
    torch.testing.assert_close(
        actual, torch.full_like(actual, 288), atol=0, rtol=0)
