// RMSNorm 的 CUDA kernel + PyTorch C++ 包装
// 与 kernels/rmsnorm_v1.cu 同源，差异只在输入输出换成 torch::Tensor
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK 256

// y = x / sqrt(mean(x^2) + eps) * g
__global__ void rmsnorm_fused(const float* __restrict__ x,
                              const float* __restrict__ g,
                              float* __restrict__ y, int N, float eps){
    int row = blockIdx.x;
    const float* rx = x + (size_t)row * N;
    float* ry = y + (size_t)row * N;
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;

    float sum_sq = 0;
    for (int idx = tid; idx < N; idx += BLOCK)
        sum_sq += rx[idx] * rx[idx];
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

    float inv = rsqrtf(s[0] / N + eps);
    for (int idx = tid; idx < N; idx += BLOCK)
        ry[idx] = rx[idx] * inv * g[idx];
}

// PyTorch 侧入口：x [B, N] float32 contiguous，g [N]
torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor g, double eps) {
    TORCH_CHECK(x.is_cuda() && g.is_cuda(), "x/g must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous() && g.is_contiguous(), "x/g must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be [B, N]");

    const int B = x.size(0);
    const int N = x.size(1);
    TORCH_CHECK(g.size(0) == N, "g must be [N]");

    auto y = torch::empty_like(x);
    rmsnorm_fused<<<B, BLOCK>>>(x.data_ptr<float>(),
                                g.data_ptr<float>(),
                                y.data_ptr<float>(),
                                N, (float)eps);
    return y;
}
