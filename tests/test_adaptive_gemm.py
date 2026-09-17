import pytest
import torch

from cuda_ai_kernels import adaptive_gemm
from cuda_ai_kernels.testing import assert_close


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(128, 128, 64), (33, 65, 17)])
def test_adaptive_gemm_linear_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = adaptive_gemm(a, b)
    expected = torch.matmul(a, b)

    assert_close(
        f"adaptive_gemm_linear_{shape}",
        actual,
        expected,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.cuda
def test_adaptive_gemm_alpha_beta_matches_torch():
    m, n, k = 65, 67, 33
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    alpha = 0.75
    beta = 0.25

    actual = adaptive_gemm(a, b, c=c, alpha=alpha, beta=beta)
    expected = alpha * torch.matmul(a, b) + beta * c

    assert_close(
        "adaptive_gemm_alpha_beta",
        actual,
        expected,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.cuda
def test_adaptive_gemm_rejects_unsupported_dtype():
    a = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="float32"):
        adaptive_gemm(a, b)
