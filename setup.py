from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent.resolve()


setup(
    name="cuda-ai-infra-kernels",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="cuda_ai_kernels._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/gemm/gemm_naive.cu",
                "csrc/gemm/gemm_block_tile.cu",
                "csrc/gemm/gemm_thread_tile.cu",
                "csrc/gemm/gemm_warp_tile.cu",
                "csrc/rmsnorm/rmsnorm.cu",
                "csrc/softmax/softmax.cu",
                "csrc/rope/rope.cu",
                "csrc/attention/attention_decode.cu",
            ],
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "--generate-line-info"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
