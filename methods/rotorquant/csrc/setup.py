from setuptools import setup, find_packages
import os
import sys

ext_modules = []
cmdclass = {}

build_cuda = "--cuda" in sys.argv or os.environ.get("TURBOQUANT_BUILD_CUDA", "0") == "1"
if "--cuda" in sys.argv:
    sys.argv.remove("--cuda")

if build_cuda or "build_ext" in sys.argv:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    def nvcc_flags():
        nvcc_threads = os.getenv("NVCC_THREADS", "8")
        return [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            f"--threads={nvcc_threads}",
        ]

    csrc_dir = os.path.dirname(__file__)

    ext_modules = [
        CUDAExtension(
            name="rotorquant.rotor_fused",
            sources=[os.path.join(csrc_dir, "rotor_fused_kernel.cu")],
            extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": nvcc_flags()},
        ),
    ]
    cmdclass = {"build_ext": BuildExtension}
    print("RotorQuant CUDA extensions will be built.")

setup(
    name="rotorquant",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
