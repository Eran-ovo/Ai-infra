// GEMM 的 CUDA kernel + PyTorch 包装
// 与 kernels/gemm_v2.cu 同源（tiled + coalesced + __ldg），差异只在输入输出换成 torch::Tensor
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 32

// v2: shared memory tiling + __ldg 只读缓存
// A [M,K] row-major, B [K,N] row-major, C [M,N] row-major
__global__ void gemm_v2(const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1)/TILE; ++t) {
        if (row < M && t*TILE+threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row*K + t*TILE + threadIdx.x]);
        else As[threadIdx.y][threadIdx.x] = 0;
        if (col < N && t*TILE+threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(t*TILE+threadIdx.y)*N + col]);
        else Bs[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}

// PyTorch 侧入口：a [M,K] b [K,N] float32 contiguous -> c [M,N]
torch::Tensor gemm_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "a/b must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a/b must be contiguous");

    const int M = a.size(0);
    const int K = a.size(1);
    TORCH_CHECK(b.size(0) == K, "inner dim mismatch: a is [M,K], b must be [K,N]");
    const int N = b.size(1);

    auto c = torch::empty({M, N}, a.options());
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1)/TILE, (M + TILE - 1)/TILE);
    gemm_v2<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    return c;
}
