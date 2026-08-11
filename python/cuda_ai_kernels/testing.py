import torch


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this project")


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    max_abs = (actual - expected).abs().max().item()
    denom = expected.abs().clamp_min(1e-12)
    max_rel = ((actual - expected).abs() / denom).max().item()
    print(f"{name}: max_abs_error={max_abs:.6e}, max_rel_error={max_rel:.6e}")


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * inv_rms * weight

