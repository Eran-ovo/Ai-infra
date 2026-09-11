// GEMM 的 CUDA kernel + PyTorch 包装
// 与 kernels/gemm_v2.cu 同源（tiled + coalesced + __ldg），差异只在输入输出换成 torch::Tensor
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>

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
    TORCH_CHECK(a.device() == b.device(), "a/b must be on the same CUDA device");

    const int64_t M64 = a.size(0);
    const int64_t K64 = a.size(1);
    const int64_t N64 = b.size(1);
    TORCH_CHECK(b.size(0) == K64, "inner dim mismatch: a is [M,K], b must be [K,N]");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
                "M/N/K are too large for the CUDA kernel");
    TORCH_CHECK((M64 + TILE - 1) / TILE <= 65535,
                "M exceeds the CUDA grid.y limit for this kernel");

    c10::cuda::CUDAGuard device_guard(a.device());

    auto c = torch::empty({M64, N64}, a.options());
    if (M64 == 0 || N64 == 0)
        return c;
    if (K64 == 0) {
        c.zero_();
        return c;
    }

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);
    dim3 block(TILE, TILE);
    dim3 grid(static_cast<unsigned>((N64 + TILE - 1) / TILE),
              static_cast<unsigned>((M64 + TILE - 1) / TILE));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gemm_v2<<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
