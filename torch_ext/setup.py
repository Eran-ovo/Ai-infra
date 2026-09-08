from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 一个模块 ai_infra_ops 暴露全部 5 个算子（工业界标准：一个共享库，多个 forward）
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
                "csrc/flashattention_cuda.cu",
            ],
            extra_cuda_cflags=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
