import torch

from cuda_ai_kernels import _C


_GEMM_IMPLS = {
    "naive": _C.gemm_naive,
    "tiled": _C.gemm_tiled,
    "thread_tile": _C.gemm_thread_tile,
    "warp_tile": _C.gemm_warp_tile,
}


def gemm(a: torch.Tensor, b: torch.Tensor, impl: str = "naive") -> torch.Tensor:
    if impl not in _GEMM_IMPLS:
        raise ValueError(f"unknown GEMM impl: {impl}")
    return _GEMM_IMPLS[impl](a.contiguous(), b.contiguous())


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _C.rmsnorm_forward(x.contiguous(), weight.contiguous(), eps)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return _C.softmax_forward(x.contiguous())


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _C.rope_forward(x.contiguous(), cos.contiguous(), sin.contiguous())


def attention_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    return _C.attention_decode_forward(q.contiguous(), k_cache.contiguous(), v_cache.contiguous())

