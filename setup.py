from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent.resolve()
CUTLASS_ROOT = ROOT / "third_party" / "cutlass"



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
                "csrc/gemm/gemm_cublas.cu",
                "csrc/gemm/gemm_block_tile.cu",
                "csrc/gemm/gemm_block_tile_float4.cu",
                "csrc/gemm/gemm_thread_tile.cu",
                "csrc/gemm/gemm_warp_tile.cu",
                "csrc/gemm/gemm_prefetchSmem.cu",
                "csrc/gemm/gemm_prefetchRegister.cu",
                "csrc/gemm/gemm_splictK.cu",
                "csrc/gemm/gemm_tensor_core.cu",
                "csrc/gemm/gemm_cutlass.cu",
                "csrc/rmsnorm/rmsnorm_naive.cu",
                "csrc/rmsnorm/rmsnorm_cache_x.cu",
                "csrc/softmax/softmax.cu",
                "csrc/rope/rope.cu",
                "csrc/attention/attention_decode.cu",
                "csrc/adaptive_gemm/adaptive_gemm.cu",
                "csrc/adaptive_gemm/cublas_reference.cu",
                "csrc/adaptive_gemm/gemm_problem.cpp",
                "csrc/adaptive_gemm/kernel_dispatch.cu",
            ],
            include_dirs=[
                str(ROOT / "csrc"),
                str(CUTLASS_ROOT / "include"),
            ],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3"],
                # Let PyTorch honor TORCH_CUDA_ARCH_LIST or detect the GPU.
                # The prefetch kernels require SM80+; do not hard-code sm_80
                # cubins, which cannot run on a different major architecture.
                "nvcc": ["-O3", "--use_fast_math", "--generate-line-info"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
