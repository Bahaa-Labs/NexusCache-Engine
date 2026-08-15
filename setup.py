import os
import sys

from setuptools import find_packages, setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ext_modules = []
cmdclass = {}

# Check if CUDA extension build should be skipped (e.g. inside Docker build stage)
skip_cuda_build = os.getenv("NO_CUDA_EXT", "0") == "1" or "--no-cuda" in sys.argv

if "--no-cuda" in sys.argv:
    sys.argv.remove("--no-cuda")

if not skip_cuda_build:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        sources = [
            "csrc/bindings/bindings.cpp",
            "csrc/src/block_manager.cpp",
            "csrc/src/page_table.cpp",
            "csrc/src/memory_utils.cpp",
            "csrc/src/pinned_memory.cpp",
            "csrc/kernels/paged_attention_kernel.cu",
            "csrc/kernels/memory_kernels.cu",
        ]

        include_dirs = [
            os.path.join(BASE_DIR, "csrc", "include"),
        ]

        extra_compile_args = {
            "cxx": [
                "-O3",
                "-std=c++17",
                "-DPYBIND11_DETAILED_ERROR_MESSAGES",
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "-Wno-unknown-pragmas",
                "-Wno-attributes",
            ],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-Xptxas",
                "-O3",
                "-Xcompiler",
                "-fPIC",
            ],
        }

        extra_link_args = ["-lculibos"]

        ext_modules = [
            CUDAExtension(
                name="nexuscache._C",
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ]
        cmdclass = {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}

    except Exception as e:
        print(
            f"[WARN] Failed to configure C++/CUDA extensions: {e}. Building pure Python package."
        )

setup(
    name="nexuscache",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
