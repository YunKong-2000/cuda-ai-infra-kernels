import pytest
import torch

from cuda_ai_kernels import adaptive_gemm
from cuda_ai_kernels.testing import assert_close


@pytest.fixture(autouse=True)
def restore_tf32_setting():
    previous = torch.backends.cuda.matmul.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


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
@pytest.mark.parametrize("shape", [(128, 128, 64), (33, 65, 17)])
def test_adaptive_gemm_cublas_matches_torch(shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = adaptive_gemm(a, b, kernel="cublas")
    expected = torch.matmul(a, b)

    assert_close(
        f"adaptive_gemm_cublas_{shape}",
        actual,
        expected,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.cuda
def test_adaptive_gemm_cublas_alpha_beta_matches_torch():
    m, n, k = 65, 67, 33
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32)
    alpha = 0.75
    beta = 0.25

    actual = adaptive_gemm(
        a,
        b,
        c=c,
        alpha=alpha,
        beta=beta,
        kernel="cublas",
    )
    expected = alpha * torch.matmul(a, b) + beta * c

    assert_close(
        "adaptive_gemm_cublas_alpha_beta",
        actual,
        expected,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kernel",
    [
        "fast",
        "fallback",
        "fast_mwarp",
        "fallback_mwarp",
        "fast_stage3",
        "fast_stage2",
    ],
)
@pytest.mark.parametrize("shape", [(128, 128, 64), (33, 68, 20)])
def test_adaptive_gemm_forced_kernel_matches_torch(kernel, shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = adaptive_gemm(a, b, kernel=kernel)
    expected = torch.matmul(a, b)

    assert_close(
        f"adaptive_gemm_forced_{kernel}",
        actual,
        expected,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("kernel", ["fast", "fast_mwarp", "fast_stage3", "fast_stage2"])
@pytest.mark.parametrize("shape", [(33, 65, 20), (33, 68, 17)])
def test_adaptive_gemm_forced_kernel_rejects_misaligned_shape(kernel, shape):
    m, n, k = shape
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    # Explicit selection must fail instead of silently running a different kernel.
    with pytest.raises(RuntimeError, match="adaptive GEMM dispatch failed"):
        adaptive_gemm(a, b, kernel=kernel)


@pytest.mark.cuda
@pytest.mark.parametrize("kernel", ["fallback", "fallback_mwarp"])
@pytest.mark.parametrize("shape", [(33, 65, 20), (33, 68, 17), (33, 65, 17)])
def test_adaptive_gemm_forced_fallback_matches_torch_on_misaligned_shape(kernel, shape):
    m, n, k = shape
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)

    actual = adaptive_gemm(a, b, kernel=kernel)
    expected = torch.matmul(a, b)

    assert_close(
        f"adaptive_gemm_forced_{kernel}_{shape}",
        actual,
        expected,
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.cuda
def test_adaptive_gemm_rejects_unknown_kernel():
    a = torch.randn((16, 16), device="cuda", dtype=torch.float32)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="unsupported adaptive_gemm kernel"):
        adaptive_gemm(a, b, kernel="unknown")


@pytest.mark.cuda
def test_adaptive_gemm_rejects_unsupported_dtype():
    a = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="float32"):
        adaptive_gemm(a, b)


RELU_CASES = [
    ("auto", (128, 128, 64)),
    ("auto", (33, 65, 17)),
    ("auto", (1, 1, 1)),
    ("fast_relu", (128, 128, 64)),
    ("fast_relu", (33, 68, 20)),
    ("fallback_relu", (128, 128, 64)),
    ("fallback_relu", (33, 65, 17)),
    ("fallback_relu", (1, 1, 1)),
]


@pytest.mark.cuda
@pytest.mark.parametrize("kernel,shape", RELU_CASES)
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.75, 0.25), (-0.5, 1.0), (0.0, -1.0)])
def test_adaptive_gemm_relu_matches_torch(kernel, shape, alpha, beta):
    m, n, k = shape
    torch.manual_seed(0)
    # Use full FP32 for the reference; CUTLASS still uses its TF32 MMA path.
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)
    c = torch.randn((m, n), device="cuda", dtype=torch.float32) if beta else None
    c_before = c.clone() if c is not None else None

    actual = adaptive_gemm(a, b, c=c, alpha=alpha, beta=beta, epilogue="relu", kernel=kernel)
    expected = alpha * (a @ b)
    if c is not None:
        expected = expected + beta * c
    expected = torch.relu(expected)

    assert actual.shape == (m, n)
    assert actual.dtype == a.dtype and actual.device == a.device
    assert torch.all(actual >= 0).item()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    if c is not None:
        torch.testing.assert_close(c, c_before, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("kernel", ["auto", "fast_relu", "fallback_relu", "fast_Relu", "fallback_Relu"])
def test_adaptive_gemm_relu_applies_after_alpha_beta(kernel):
    # Exact, representable inputs distinguish ReLU(alpha * AB + beta * C)
    # from a missing activation or activation applied before scaling/source add.
    a = torch.eye(4, device="cuda", dtype=torch.float32)
    b = torch.tensor([[-2.0, -1.0, 1.0, 2.0]], device="cuda").repeat(4, 1)
    c = torch.tensor([[-2.0, -2.0, 2.0, 2.0]], device="cuda").repeat(4, 1)
    actual = adaptive_gemm(a, b, c=c, alpha=-1.0, beta=1.0, epilogue="relu", kernel=kernel)
    expected = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device="cuda").repeat(4, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("kernel", ["auto", "fast_relu", "fallback_relu"])
def test_adaptive_gemm_relu_beta_zero_ignores_nan_source(kernel):
    a = torch.eye(4, device="cuda", dtype=torch.float32)
    b = torch.tensor([[-2.0, -1.0, 1.0, 2.0]], device="cuda").repeat(4, 1)
    c = torch.full((4, 4), float("nan"), device="cuda")
    actual = adaptive_gemm(a, b, c=c, beta=0.0, epilogue="relu", kernel=kernel)
    torch.testing.assert_close(actual, b.relu(), atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(33, 65, 20), (33, 68, 17)])
def test_adaptive_gemm_fast_relu_rejects_misaligned_shape(shape):
    m, n, k = shape
    a = torch.randn((m, k), device="cuda")
    b = torch.randn((k, n), device="cuda")
    with pytest.raises(RuntimeError, match="adaptive GEMM dispatch failed"):
        adaptive_gemm(a, b, epilogue="relu", kernel="fast_relu")


@pytest.mark.cuda
@pytest.mark.parametrize("kernel,epilogue", [
    ("fast", "relu"), ("fallback", "relu"),
    ("fast_relu", "linear"), ("fallback_relu", "linear"),
])
def test_adaptive_gemm_rejects_kernel_epilogue_mismatch(kernel, epilogue):
    a = torch.eye(4, device="cuda")
    with pytest.raises(RuntimeError, match="adaptive GEMM dispatch failed"):
        adaptive_gemm(a, a, epilogue=epilogue, kernel=kernel)


@pytest.mark.cuda
def test_adaptive_gemm_cublas_rejects_fused_relu():
    a = torch.eye(4, device="cuda")
    with pytest.raises(RuntimeError, match="supports only the linear epilogue"):
        adaptive_gemm(a, a, epilogue="relu", kernel="cublas")


@pytest.mark.cuda
def test_adaptive_gemm_relu_requires_c_when_beta_nonzero():
    a = torch.eye(4, device="cuda")
    with pytest.raises(RuntimeError, match="C is required"):
        adaptive_gemm(a, a, beta=1.0, epilogue="relu")
