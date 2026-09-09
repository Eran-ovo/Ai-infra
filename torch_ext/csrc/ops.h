#pragma once
#include <torch/extension.h>

// 所有算子的 Python 侧入口函数声明，bindings.cpp 和各自 .cu 共同引用
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor g, double eps);
torch::Tensor softmax_forward(torch::Tensor x);
torch::Tensor layernorm_forward(torch::Tensor x, double eps);
torch::Tensor gemm_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_mma_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor flashattention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
torch::Tensor flashattention_fp16_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);
