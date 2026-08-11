import pytest


@pytest.mark.skip(reason="Softmax kernel is a TODO for user implementation")
@pytest.mark.parametrize("shape", [(16, 128), (32, 512), (8, 2048)])
def test_softmax_matches_torch(shape):
    pass
