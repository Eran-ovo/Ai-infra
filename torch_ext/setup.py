from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 一个模块 ai_infra_ops 暴露多类算子（工业界标准：一个共享库，多个 forward）
setup(
    name="ai_infra_ops",
    ext_modules=[
        CUDAExtension(
            name="ai_infra_ops",
            sources=[
                "csrc/bindings.cpp",           # PyBind11 绑定：把所有 forward 暴露给 Python
                "csrc/rmsnorm_cuda.cu",
                "csrc/softmax_cuda.cu",
                "csrc/layernorm_cuda.cu",
                "csrc/gemm_cuda.cu",
                "csrc/gemm_mma_cuda.cu",       # fp16 Tensor Core GEMM（手写 mma.sync）
                "csrc/gemm_mma_v10_cuda.cu",   # GEMM v10（BK=32 + cp.async + ldmatrix）
                "csrc/gemm_dispatch.cpp",      # 稳定入口：按 shape 路由 v10/v8/v4
                "csrc/flashattention_cuda.cu",
                "csrc/flashattention_mma_cuda.cu",  # FlashAttention v4（fp16 Tensor Core）
                "csrc/flashattention_v5_cuda.cu",   # FlashAttention v5（P fragment 寄存器直连）
                "csrc/flashattention_v6_cuda.cu",   # FlashAttention v6（D=128 实验版）
                "csrc/flashattention_dispatch.cpp", # 稳定入口：按 head dimension 路由 v5/v6
            ],
            libraries=["cublas"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
