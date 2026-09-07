// Softmax 的 CUDA kernel + PyTorch 包装
// 与 kernels/softmax_v1.cu 同源，差异只在输入输出换成 torch::Tensor
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK 256

// softmax(x) = exp(x-max) / sum(exp(x-max))，一行一个 block，两次归约 max -> sum
__global__ void softmax_fused(const float* __restrict__ x, float* __restrict__ y, int N) {
    int row = blockIdx.x;
    const float* row_x = x + row * N;
    float* row_y = y + row * N;
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;

    // 1. 线程粗化求 max
    float v = -1e20f;
    for (int idx = tid; idx < N; idx += BLOCK)
        v = fmaxf(v, row_x[idx]);
    sdata[tid] = v;
    __syncthreads();
    for (int s = BLOCK/2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid < 32) {
        float val = sdata[tid];
        for (int s = 16; s > 0; s >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, s));
        if (tid == 0) sdata[0] = val;
    }
    __syncthreads();
    float row_max = sdata[0];

    // 2. 归约求 sum(exp(x-max))
    float e = 0;
    for (int idx = tid; idx < N; idx += BLOCK)
        e += expf(row_x[idx] - row_max);
    sdata[tid] = e;
    __syncthreads();
    for (int s = BLOCK/2; s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float val = sdata[tid];
        for (int s = 16; s > 0; s >>= 1)
            val += __shfl_down_sync(0xffffffff, val, s);
        if (tid == 0) sdata[0] = val;
    }
    __syncthreads();
    float row_sum = sdata[0];

    // 3. 写回
    for (int idx = tid; idx < N; idx += BLOCK)
        row_y[idx] = expf(row_x[idx] - row_max) / row_sum;
}

// PyTorch 侧入口：x [B, N] float32 contiguous
torch::Tensor softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be [B, N]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int B = x.size(0);
    const int N = x.size(1);

    auto y = torch::empty_like(x);
    softmax_fused<<<B, BLOCK>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}
