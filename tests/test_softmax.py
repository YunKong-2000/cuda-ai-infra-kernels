import pytest
import torch

from cuda_ai_kernels import softmax
from cuda_ai_kernels.testing import assert_close
@pytest.mark.parametrize("impl", ["naive"], ids=["naive"])
@pytest.mark.parametrize("shape", [(16, 128), (32, 512), (8, 2048)])
def test_softmax_matches_torch(shape, impl):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for exact comparison
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    # Compute Softmax using the custom kernel
    out_custom = softmax(x, impl)

    # Compute Softmax using PyTorch's built-in operations for reference
    out_reference = torch.softmax(x, dim=1)

    # Check if the outputs are close
    assert torch.allclose(out_custom, out_reference, atol=1e-5), "Softmax outputs do not match reference implementation"

