// RMSNorm 的 CUDA kernel + PyTorch C++ 包装
// 与 kernels/rmsnorm_v1.cu 同源，差异只在输入输出换成 torch::Tensor
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

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
    TORCH_CHECK(x.dtype() == torch::kFloat32 && g.dtype() == torch::kFloat32,
                "x/g must be float32");
    TORCH_CHECK(x.is_contiguous() && g.is_contiguous(), "x/g must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be [B, N]");
    TORCH_CHECK(g.dim() == 1, "g must be [N]");
    TORCH_CHECK(x.device() == g.device(), "x/g must be on the same CUDA device");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "eps must be finite and non-negative");

    const int64_t B64 = x.size(0);
    const int64_t N64 = x.size(1);
    TORCH_CHECK(B64 <= INT_MAX, "B is too large for the CUDA kernel");
    TORCH_CHECK(N64 > 0 && N64 <= INT_MAX, "N must be in [1, INT_MAX]");
    TORCH_CHECK(g.size(0) == N64, "g must have shape [N]");

    c10::cuda::CUDAGuard device_guard(x.device());

    auto y = torch::empty_like(x);
    if (B64 == 0)
        return y;

    const int B = static_cast<int>(B64);
    const int N = static_cast<int>(N64);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    rmsnorm_fused<<<B, BLOCK, 0, stream>>>(x.data_ptr<float>(),
                                           g.data_ptr<float>(),
                                           y.data_ptr<float>(),
                                           N, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
