// PyBind11 绑定：把 C++ forward 函数统一暴露给 Python
#include <torch/extension.h>
#include "ops.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Ai-infra 手写 CUDA 算子的 PyTorch Extension（GEMM/Softmax/LayerNorm/RMSNorm/FlashAttention）";
    m.def("rmsnorm",        &rmsnorm_forward,        "RMSNorm forward (CUDA)");
    m.def("softmax",        &softmax_forward,        "Softmax forward (CUDA)");
    m.def("layernorm",      &layernorm_forward,      "LayerNorm forward (CUDA，无 affine)");
    m.def("gemm",           &gemm_forward,           "GEMM forward (CUDA, tiled)");
    m.def("gemm_mma",       &gemm_mma_forward,       "GEMM forward (CUDA, fp16 Tensor Core mma.sync)");
    m.def("flashattention", &flashattention_forward, "FlashAttention forward (CUDA, online softmax)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
    m.def("flashattention_fp16", &flashattention_fp16_forward,
          "FlashAttention v4 forward (CUDA, fp16 Tensor Core, shared-memory P)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
    m.def("flashattention_v5", &flashattention_v5_forward,
          "FlashAttention v5 forward (CUDA, fp16 Tensor Core, register P fragment)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false);
}
