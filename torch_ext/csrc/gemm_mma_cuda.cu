// FP16 Tensor Core GEMM（手写 mma.sync.m16n8k16）的 PyTorch 包装
// 与 kernels/gemm_v4_mma.cu 同源，差异只在输入输出换成 torch::Tensor
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 块级 tile：每个 block 算 64x64 的 C 块，K 方向每轮 BK=16
#define BM 64
#define BN 64
#define BK 16
// 256 线程 = 8 warps，排列成 4(沿M) x 2(沿N)，每 warp 算 16x32 = 4 个 m16n8k16
#define WM 4
#define WN 2
#define WMMA_M 16
#define WMMA_N 8

__device__ __forceinline__ unsigned as_u32(__half2 x) {
    return *reinterpret_cast<unsigned*>(&x);
}

// 单次 m16n8k16 mma：C(4x fp32) += A(4x half2) * B(2x half2)
__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    __half2 a0, __half2 a1, __half2 a2, __half2 a3,
    __half2 b0, __half2 b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(as_u32(a0)), "r"(as_u32(a1)), "r"(as_u32(a2)), "r"(as_u32(a3)),
          "r"(as_u32(b0)), "r"(as_u32(b1)));
}

__global__ void gemm_mma(const __half* __restrict__ A,
                         const __half* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K)
{
    const int m_base  = blockIdx.y * BM;
    const int n_base  = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / WN;   // 0..3
    const int warp_n  = warp_id % WN;   // 0..1

    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BK][BN];

    float acc[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[t][i] = 0.0f;

    const int lane = tid % 32;
    const int g    = lane / 4;   // groupID 0..7
    const int gid  = lane % 4;   // threadID_in_group 0..3

    const int N_TILES    = (BN / WN) / WMMA_N;   // 4
    const int n_warp_off = warp_n * (BN / WN);   // warp_n * 32
    const int NT = WM * WN * 32;                 // 256

    for (int kt = 0; kt < K; kt += BK) {
        // 全 block 协作搬 A/B tile（越界补零）
        for (int i = tid; i < BM * BK; i += NT) {
            int r = i / BK, c = i % BK;
            int gr = m_base + r, gc = kt + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = tid; i < BK * BN; i += NT) {
            int r = i / BN, c = i % BN;
            int gr = kt + r, gc = n_base + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        // A fragment（m16n8k16.row.col 规范）：
        //   a0=(g, 2*gid)  a1=(g+8, 2*gid)  a2=(g, 2*gid+8)  a3=(g+8, 2*gid+8)
        int ar = warp_m * WMMA_M;
        __half2 a0 = __halves2half2(As[ar + g][gid * 2],         As[ar + g][gid * 2 + 1]);
        __half2 a1 = __halves2half2(As[ar + g + 8][gid * 2],     As[ar + g + 8][gid * 2 + 1]);
        __half2 a2 = __halves2half2(As[ar + g][gid * 2 + 8],     As[ar + g][gid * 2 + 9]);
        __half2 a3 = __halves2half2(As[ar + g + 8][gid * 2 + 8], As[ar + g + 8][gid * 2 + 9]);

        #pragma unroll
        for (int t = 0; t < N_TILES; ++t) {
            int nc = n_warp_off + t * WMMA_N;
            // B fragment col.major：k 由 gid 拆成 (2*gid, 2*gid+1, +8, +9)，n = g
            __half2 b0 = __halves2half2(Bs[gid * 2][nc + g],     Bs[gid * 2 + 1][nc + g]);
            __half2 b1 = __halves2half2(Bs[gid * 2 + 8][nc + g], Bs[gid * 2 + 9][nc + g]);

            mma_m16n8k16(acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                         a0, a1, a2, a3, b0, b1);
        }
        __syncthreads();
    }

    // 写回 C：本 warp 的 16x32 块 = 4 个 n8 tile
    #pragma unroll
    for (int t = 0; t < N_TILES; ++t) {
        int nc = n_base + n_warp_off + t * WMMA_N;
        int mr = m_base + warp_m * WMMA_M;
        // c0=(g, gid*2) c1=(g, gid*2+1) c2=(g+8, gid*2) c3=(g+8, gid*2+1)
        int r0 = mr + g, c0 = nc + gid * 2, r2 = mr + g + 8;
        if (r0 < M && c0 + 1 < N) {
            C[r0 * N + c0]     = acc[t][0];
            C[r0 * N + c0 + 1] = acc[t][1];
        }
        if (r2 < M && c0 + 1 < N) {
            C[r2 * N + c0]     = acc[t][2];
            C[r2 * N + c0 + 1] = acc[t][3];
        }
    }
}

// PyTorch 侧入口：a [M,K] fp16, b [K,N] fp16, contiguous -> c [M,N] fp32
// 与 cuBLAS fp16 gemm（CUBLAS_COMPUTE_32F + fp32 输出）语义一致：fp16 输入 / fp32 累加输出
torch::Tensor gemm_mma_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kHalf && b.dtype() == torch::kHalf, "a/b must be fp16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a/b must be contiguous");

    const int M = a.size(0);
    const int K = a.size(1);
    TORCH_CHECK(b.size(0) == K, "inner dim mismatch: a is [M,K], b must be [K,N]");
    const int N = b.size(1);

    auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const __half* pa = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* pb = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    gemm_mma<<<grid, WM * WN * 32>>>(pa, pb, c.data_ptr<float>(), M, N, K);
    return c;
}
