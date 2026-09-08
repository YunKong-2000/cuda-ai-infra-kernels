import torch

from cuda_ai_kernels import _C


_GEMM_IMPLS = {
    "cublas": _C.gemm_cublas,
    "naive": _C.gemm_naive,
    "block_tile": _C.gemm_block_tile,
    "thread_tile": _C.gemm_thread_tile,
    "warp_tile": _C.gemm_warp_tile,
    "tensor_core": _C.gemm_tensor_core
}

_RMSNORM_IMPLS = {
    "naive": _C.rmsnorm_naive,
    "cache_x": _C.rmsnorm_cache_x
}

_SOFTMAX_IMPLS = {
    "naive": _C.softmax_naive
}


def gemm(a: torch.Tensor, b: torch.Tensor, impl: str = "naive") -> torch.Tensor:
    if impl not in _GEMM_IMPLS:
        raise ValueError(f"unknown GEMM impl: {impl}")
    return _GEMM_IMPLS[impl](a.contiguous(), b.contiguous())


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, impl: str = "naive") -> torch.Tensor:
    if impl not in _RMSNORM_IMPLS:
        raise ValueError(f"unknown RMSNorm impl: {impl}")
    return _RMSNORM_IMPLS[impl](x.contiguous(), weight.contiguous(), eps)


def softmax(x: torch.Tensor, impl: str = "naive") -> torch.Tensor:
    if impl not in _SOFTMAX_IMPLS:
        raise ValueError(f"unknown softmax impl: {impl}")
    return _SOFTMAX_IMPLS[impl](x.contiguous())


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return _C.rope_forward(x.contiguous(), cos.contiguous(), sin.contiguous())


def attention_decode(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    return _C.attention_decode_forward(q.contiguous(), k_cache.contiguous(), v_cache.contiguous())
