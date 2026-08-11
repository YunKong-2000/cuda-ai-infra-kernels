import pytest


@pytest.mark.skip(reason="RMSNorm kernel is a TODO for user implementation")
@pytest.mark.parametrize("shape", [(16, 768), (32, 1024), (8, 4096)])
def test_rmsnorm_matches_reference(shape):
    pass
