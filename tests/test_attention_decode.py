import pytest

from cuda_ai_kernels import attention_decode  # noqa: F401


@pytest.mark.skip(reason="Attention decode kernel is planned for v0.3")
def test_attention_decode_placeholder():
    pass

