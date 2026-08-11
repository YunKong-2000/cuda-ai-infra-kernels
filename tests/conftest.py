import pytest
import torch


def pytest_runtest_setup(item):
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

