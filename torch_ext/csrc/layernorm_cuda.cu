// LayerNorm 的 CUDA kernel + PyTorch 包装
// 与 kernels/layernorm_v1.cu 同源，差异只在输入输出换成 torch::Tensor
// 注：v1 为纯归一化（无 affine），公式 y=(x-mean)/sqrt(var+eps)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

#define BLOCK 256

// 一行一个 block，两次归约：求均值 -> 求方差
__global__ void layernorm_fused(const float* __restrict__ x, float* __restrict__ y, int N, float eps) {
    int row = blockIdx.x;
    const float* rx = x + (size_t)row * N;
    float* ry = y + (size_t)row * N;
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;

    // 1. 归约求 sum -> mean
    float v = 0;
    for (int idx = tid; idx < N; idx += BLOCK)
        v += rx[idx];
    s[tid] = v;
    __syncthreads();
    for (int s2 = BLOCK/2; s2 >= 32; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    if (tid < 32) {
        float val = s[tid];
        for (int s2 = 16; s2 > 0; s2 >>= 1)
            val += __shfl_down_sync(0xffffffff, val, s2);
        if (tid == 0) s[0] = val;
    }
    __syncthreads();
    float mean = s[0] / N;

    // 2. 归约求方差：暂存差值，同时累加平方和
    float sum_sq = 0;
    for (int idx = tid; idx < N; idx += BLOCK) {
        float d = rx[idx] - mean;
        ry[idx] = d;        // 暂存差值到输出，最后一步直接乘 inv
        sum_sq += d * d;
    }
    s[tid] = sum_sq;
    __syncthreads();
    for (int s2 = BLOCK/2; s2 >= 32; s2 >>= 1) {
        if (tid < s2) s[tid] += s[tid + s2];
        __syncthreads();
    }
    if (tid < 32) {
        float val = s[tid];
        for (int s2 = 16; s2 > 0; s2 >>= 1)
            val += __shfl_down_sync(0xffffffff, val, s2);
        if (tid == 0) s[0] = val;
    }
    __syncthreads();
    float var = s[0] / N;
    float inv = rsqrtf(var + eps);

    // 3. 归一化写回（差值已在 ry 中暂存）
    for (int idx = tid; idx < N; idx += BLOCK)
        ry[idx] *= inv;
}

// PyTorch 侧入口：x [B, N] float32 contiguous（无 gamma/beta）
torch::Tensor layernorm_forward(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be [B, N]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "eps must be finite and non-negative");

    const int64_t B64 = x.size(0);
    const int64_t N64 = x.size(1);
    TORCH_CHECK(B64 <= INT_MAX, "B is too large for the CUDA kernel");
    TORCH_CHECK(N64 > 0 && N64 <= INT_MAX, "N must be in [1, INT_MAX]");

    c10::cuda::CUDAGuard device_guard(x.device());

    auto y = torch::empty_like(x);
    if (B64 == 0)
        return y;

    const int B = static_cast<int>(B64);
    const int N = static_cast<int>(N64);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    layernorm_fused<<<B, BLOCK, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
