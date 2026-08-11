import pytest


@pytest.mark.skip(reason="GEMM kernels are TODOs for user implementation")
@pytest.mark.parametrize("impl", ["naive", "tiled", "thread_tile", "warp_tile"])
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64), (33, 65, 17)])
def test_gemm_matches_torch(impl, shape):
    pass
