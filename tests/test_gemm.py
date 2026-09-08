import pytest
import torch

from cuda_ai_kernels import gemm
from cuda_ai_kernels.testing import assert_close
@pytest.mark.parametrize("impl", ["tiled", "thread_tile", "warp_tile", "tensor_core", "cublas"])

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
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_naive_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact comparison
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="naive")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_naive_{shape}", actual, expected, atol=1e-3, rtol=1e-3)

@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_block_tile_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact comparison
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="block_tile")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_block_tile_{shape}", actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_thread_tile_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact comparison
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="thread_tile")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_thread_tile_{shape}", actual, expected, atol=1e-3, rtol=1e-3)

@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_warp_tile_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact comparison
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="warp_tile")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_warp_tile_{shape}", actual, expected, atol=1e-3, rtol=1e-3)

@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_tensor_core_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    # torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact comparison
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = gemm(a, b, impl="tensor_core")
    expected = torch.matmul(a, b)

    assert_close(f"gemm_tensor_core_{shape}", actual, expected, atol=1e-3, rtol=1e-3)


def test_gemm_optimized_placeholders(impl):
    pass
