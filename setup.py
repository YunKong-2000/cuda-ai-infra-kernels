from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent


def source(path: str) -> str:
    return str(ROOT / path)


setup(
    name="cuda-ai-infra-kernels",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="cuda_ai_kernels._C",
            sources=[
                source("csrc/bindings.cpp"),
                source("csrc/gemm/gemm_naive.cu"),
                source("csrc/gemm/gemm_tiled.cu"),
                source("csrc/gemm/gemm_thread_tile.cu"),
                source("csrc/gemm/gemm_warp_tile.cu"),
                source("csrc/rmsnorm/rmsnorm.cu"),
                source("csrc/softmax/softmax.cu"),
                source("csrc/rope/rope.cu"),
                source("csrc/attention/attention_decode.cu"),
            ],
            include_dirs=[source("csrc")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

