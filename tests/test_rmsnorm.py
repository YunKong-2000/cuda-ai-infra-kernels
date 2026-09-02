import pytest
import torch

from cuda_ai_kernels import rmsnorm_forward
from cuda_ai_kernels.testing import assert_close

@pytest.mark.skip(reason="RMSNorm kernel is a TODO for user implementation")
@pytest.mark.parametrize("shape", [(16, 768), (32, 1024), (8, 4096)])


def test_rmsnorm_matches_reference(shape):
    # Create random input tensor and weight
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    weight = torch.randn(shape[1], device="cuda", dtype=torch.float32)

    # Compute RMSNorm using the custom kernel
    out_custom = rmsnorm_forward(x, weight)

    # Compute RMSNorm using PyTorch's built-in operations for reference
    mean_square = torch.mean(x * x, dim=1, keepdim=True)
    rms = torch.sqrt(mean_square + 1e-8)
    out_reference = (x / rms) * weight

    # Check if the outputs are close
    assert torch.allclose(out_custom, out_reference, atol=1e-5), "RMSNorm outputs do not match reference implementation"
