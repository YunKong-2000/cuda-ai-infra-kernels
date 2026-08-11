__all__ = [
    "attention_decode",
    "gemm",
    "rmsnorm",
    "rope",
    "softmax",
]


def gemm(*args, **kwargs):
    from .ops import gemm as op

    return op(*args, **kwargs)


def rmsnorm(*args, **kwargs):
    from .ops import rmsnorm as op

    return op(*args, **kwargs)


def softmax(*args, **kwargs):
    from .ops import softmax as op

    return op(*args, **kwargs)


def rope(*args, **kwargs):
    from .ops import rope as op

    return op(*args, **kwargs)


def attention_decode(*args, **kwargs):
    from .ops import attention_decode as op

    return op(*args, **kwargs)

