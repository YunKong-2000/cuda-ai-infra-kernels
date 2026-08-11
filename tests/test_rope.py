import pytest

from cuda_ai_kernels import rope  # noqa: F401


@pytest.mark.skip(reason="RoPE kernel is planned for v0.2")
def test_rope_placeholder():
    pass

