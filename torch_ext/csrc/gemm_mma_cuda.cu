// FP16 Tensor Core GEMM（手写 mma.sync.m16n8k16）的 PyTorch 包装
// 与 kernels/gemm_v4_mma.cu 同源，差异只在输入输出换成 torch::Tensor
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstdint>

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

// ldmatrix 已经把 fp16 fragment 装入 32-bit 寄存器，直接送入 mma，
// 避免先转成 __half2 再由 as_u32 取位模式。
__device__ __forceinline__ void mma_m16n8k16_regs(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned to_shared_u32(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// 一个 warp 合作加载四个 8x8 fp16 矩阵，每个 lane 得到 4 个 32-bit 寄存器。
// lane 0..7/8..15/16..23/24..31 分别提供四个矩阵的 8 个行首地址。
__device__ __forceinline__ void ldmatrix_x4(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
    const void* row_addr)
{
    const unsigned addr = to_shared_u32(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// B 在 shared memory 中是行主序 KxN；.trans 将两个上下排列的 8x8
// 子矩阵按列主序装入 b0/b1，匹配 mma.row.col 的 B fragment。
__device__ __forceinline__ void ldmatrix_x2_trans(
    unsigned& r0, unsigned& r1, const void* row_addr)
{
    const unsigned addr = to_shared_u32(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
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
        if (r0 < M && c0 < N)
            C[r0 * N + c0] = acc[t][0];
        if (r0 < M && c0 + 1 < N)
            C[r0 * N + c0 + 1] = acc[t][1];

        if (r2 < M && c0 < N)
            C[r2 * N + c0] = acc[t][2];
        if (r2 < M && c0 + 1 < N)
            C[r2 * N + c0 + 1] = acc[t][3];
    }
}

// 16-byte 同步搬运基线。每个 block 的 A/B tile 各有 2048 bytes：
// 前 128 个线程各搬 A 的 16 bytes，后 128 个线程各搬 B 的 16 bytes。
//一次搬16B，走寄存器，同步
__device__ __forceinline__ void copy_tile_vec_sync(
    const __half* A, const __half* B,
    __half* As, __half* Bs,
    int as_stride, int bs_stride,
    int m_base, int n_base, int kt, int K, int N, int tid)
{
    if (tid < 128) {
        const int elem = tid * 8;
        const int row = elem / BK;
        const int col = elem % BK;
        *reinterpret_cast<uint4*>(As + row * as_stride + col) =
            *reinterpret_cast<const uint4*>(A + (m_base + row) * K + kt + col);
    } else {
        const int load_tid = tid - 128;
        const int elem = load_tid * 8;
        const int row = elem / BN;
        const int col = elem % BN;
        *reinterpret_cast<uint4*>(Bs + row * bs_stride + col) =
            *reinterpret_cast<const uint4*>(B + (kt + row) * N + n_base + col);
    }
}

//一次搬16B，不走寄存器，异步
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned dst_shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_shared), "l"(src) : "memory");
#else
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

//表示把当前发起的一组异步复制提交
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" :: : "memory");
#endif
}

//表示等待已经提交的异步复制完成
__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
#endif
}

__device__ __forceinline__ void copy_tile_async(
    const __half* A, const __half* B,
    __half* As, __half* Bs,
    int as_stride, int bs_stride,
    int m_base, int n_base, int kt, int K, int N, int tid)
{
    if (tid < 128) {
        const int elem = tid * 8;
        const int row = elem / BK;
        const int col = elem % BK;
        cp_async_16(As + row * as_stride + col,
                    A + (m_base + row) * K + kt + col);
    } else {
        const int load_tid = tid - 128;
        const int elem = load_tid * 8;
        const int row = elem / BN;
        const int col = elem % BN;
        cp_async_16(Bs + row * bs_stride + col,
                    B + (kt + row) * N + n_base + col);
    }
}

// USE_ASYNC=false：对齐尺寸特化 + 16-byte 向量搬运 + 单缓冲。
// USE_ASYNC=true ：保持相同 fast-path 映射，只增加 cp.async + 双缓冲，隔离流水收益。
// USE_LDMATRIX=true：保持同步搬运，只替换 shared→fragment 路径，隔离 ldmatrix 收益。
// USE_SMEM_PADDING=true：A/B 每行增加 8 个 half，打散 ldmatrix 的 bank 映射。
// 该 kernel 只处理 M%64=N%64=0、K%16=0 的 fast path。
template <bool USE_ASYNC, bool USE_LDMATRIX, bool USE_SMEM_PADDING>
__global__ void gemm_mma_pipeline(const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    constexpr int STAGES = USE_ASYNC ? 2 : 1;
    // 32 个 shared-memory bank，每个 bank 4B。ldmatrix 每行读取 8 个 half
    //（16B，覆盖 4 个 bank），所以 padding 后让相邻行的起始 bank 至少错开 4：
    //   A: 24 half/row = 48B = 12 banks
    //   B: 72 half/row = 144B = 36 banks ≡ 4 (mod 32)
    // 两个 stride 仍是 16B 的倍数，不会破坏 uint4/ldmatrix 的对齐要求。
    constexpr int AS_STRIDE = BK + (USE_SMEM_PADDING ? 8 : 0);
    constexpr int BS_STRIDE = BN + (USE_SMEM_PADDING ? 8 : 0);
    static_assert(!USE_SMEM_PADDING || USE_LDMATRIX,
                  "padding is only defined for the ldmatrix path");
    //要求 As/Bs 数组的起始地址按 16 字节对齐。
    // padding 版的 A/B 行跨度分别为 48B/144B，仍保持 16B 对齐。
    __shared__ __align__(16) __half As[STAGES][BM][AS_STRIDE];
    __shared__ __align__(16) __half Bs[STAGES][BK][BS_STRIDE];

    const int m_base  = blockIdx.y * BM;
    const int n_base  = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / WN;
    const int warp_n  = warp_id % WN;
    const int lane    = tid % 32;
    const int g       = lane / 4;
    const int gid     = lane % 4;
    const int N_TILES = (BN / WN) / WMMA_N;
    const int n_warp_off = warp_n * (BN / WN);

    float acc[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[t][i] = 0.0f;

    int read_stage = 0;
    // 先加载第一个 tile
    if constexpr (USE_ASYNC) {
        copy_tile_async(A, B, &As[0][0][0], &Bs[0][0][0],
                        AS_STRIDE, BS_STRIDE,
                        m_base, n_base, 0, K, N, tid);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    for (int kt = 0; kt < K; kt += BK) {
        if constexpr (USE_ASYNC) {
            const int next_kt = kt + BK;
            if (next_kt < K) {
                const int write_stage = read_stage ^ 1;
                // 预取下一个 tile 到另一个 shared-memory buffer
                copy_tile_async(A, B,
                                &As[write_stage][0][0], &Bs[write_stage][0][0],
                                AS_STRIDE, BS_STRIDE,
                                m_base, n_base, next_kt, K, N, tid);
                cp_async_commit();
            }
        } else {
            copy_tile_vec_sync(A, B, &As[0][0][0], &Bs[0][0][0],
                               AS_STRIDE, BS_STRIDE,
                               m_base, n_base, kt, K, N, tid);
            __syncthreads();
        }

        // 计算当前 stage 的 tile
        const int ar = warp_m * WMMA_M;
        if constexpr (USE_LDMATRIX) {
            // A 的 16x16 fragment 拆成四个 8x8：
            //   matrix 0 = rows 0..7,  cols 0..7   -> a0
            //   matrix 1 = rows 8..15, cols 0..7   -> a1
            //   matrix 2 = rows 0..7,  cols 8..15  -> a2
            //   matrix 3 = rows 8..15, cols 8..15  -> a3
            // 每 8 个 lane 为对应矩阵提供 8 个行首地址。
            const int a_matrix = lane / 8;
            const int a_row = ar + (a_matrix % 2) * 8 + lane % 8;
            const int a_col = (a_matrix / 2) * 8;
            unsigned a_frag[4];
            ldmatrix_x4(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                        &As[read_stage][a_row][a_col]);

            #pragma unroll
            for (int t = 0; t < N_TILES; ++t) {
                const int nc = n_warp_off + t * WMMA_N;
                // .x2 只需要 lane 0..15 提供 16 个行地址。高 16 个 lane
                // 复制低 16 个 lane 的合法地址，兼容要求所有 lane 地址有效的架构。
                const int b_addr_lane = lane & 15;//等价于lane % 16
                const int b_row = (b_addr_lane / 8) * 8 + b_addr_lane % 8;
                unsigned b_frag[2];
                ldmatrix_x2_trans(b_frag[0], b_frag[1],
                                  &Bs[read_stage][b_row][nc]);
                mma_m16n8k16_regs(
                    acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                    a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                    b_frag[0], b_frag[1]);
            }
        } else {
            __half2 a0 = __halves2half2(As[read_stage][ar + g][gid * 2],
                                        As[read_stage][ar + g][gid * 2 + 1]);
            __half2 a1 = __halves2half2(As[read_stage][ar + g + 8][gid * 2],
                                        As[read_stage][ar + g + 8][gid * 2 + 1]);
            __half2 a2 = __halves2half2(As[read_stage][ar + g][gid * 2 + 8],
                                        As[read_stage][ar + g][gid * 2 + 9]);
            __half2 a3 = __halves2half2(As[read_stage][ar + g + 8][gid * 2 + 8],
                                        As[read_stage][ar + g + 8][gid * 2 + 9]);

            #pragma unroll
            for (int t = 0; t < N_TILES; ++t) {
                const int nc = n_warp_off + t * WMMA_N;
                __half2 b0 = __halves2half2(Bs[read_stage][gid * 2][nc + g],
                                            Bs[read_stage][gid * 2 + 1][nc + g]);
                __half2 b1 = __halves2half2(Bs[read_stage][gid * 2 + 8][nc + g],
                                            Bs[read_stage][gid * 2 + 9][nc + g]);
                mma_m16n8k16(acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                             a0, a1, a2, a3, b0, b1);
            }
        }

        if constexpr (USE_ASYNC) {
            // 等待下一个 tile 加载完成
            if (kt + BK < K) {
                cp_async_wait_all();
                __syncthreads();
                read_stage ^= 1;
            }
        } else {
            __syncthreads();
        }
    }

    #pragma unroll
    for (int t = 0; t < N_TILES; ++t) {
        const int nc = n_base + n_warp_off + t * WMMA_N;
        const int mr = m_base + warp_m * WMMA_M;
        const int r0 = mr + g;
        const int c0 = nc + gid * 2;
        const int r2 = mr + g + 8;
        C[r0 * N + c0] = acc[t][0];
        C[r0 * N + c0 + 1] = acc[t][1];
        C[r2 * N + c0] = acc[t][2];
        C[r2 * N + c0 + 1] = acc[t][3];
    }
}

// PyTorch 侧入口：a [M,K] fp16, b [K,N] fp16, contiguous -> c [M,N] fp32
// 与 cuBLAS fp16 gemm（CUBLAS_COMPUTE_32F + fp32 输出）语义一致：fp16 输入 / fp32 累加输出
torch::Tensor gemm_mma_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kHalf && b.dtype() == torch::kHalf, "a/b must be fp16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a/b must be contiguous");
    TORCH_CHECK(a.device() == b.device(), "a/b must be on the same CUDA device");

    const int64_t M64 = a.size(0);
    const int64_t K64 = a.size(1);
    const int64_t N64 = b.size(1);
    TORCH_CHECK(b.size(0) == K64, "inner dim mismatch: a is [M,K], b must be [K,N]");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
                "M/N/K are too large for the CUDA kernel");
    TORCH_CHECK((M64 + BM - 1) / BM <= 65535,
                "M exceeds the CUDA grid.y limit for this kernel");

    c10::cuda::CUDAGuard device_guard(a.device());

    auto c = torch::empty({M64, N64}, a.options().dtype(torch::kFloat32));
    if (M64 == 0 || N64 == 0)
        return c;
    if (K64 == 0) {
        c.zero_();
        return c;
    }

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);
    dim3 grid(static_cast<unsigned>((N64 + BN - 1) / BN),
              static_cast<unsigned>((M64 + BM - 1) / BM));
    const __half* pa = reinterpret_cast<const __half*>(a.data_ptr<at::Half>());
    const __half* pb = reinterpret_cast<const __half*>(b.data_ptr<at::Half>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gemm_mma<<<grid, WM * WN * 32, 0, stream>>>(pa, pb, c.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

template <bool USE_ASYNC, bool USE_LDMATRIX, bool USE_SMEM_PADDING>
torch::Tensor gemm_mma_pipeline_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kHalf && b.dtype() == torch::kHalf, "a/b must be fp16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a/b must be contiguous");
    TORCH_CHECK(a.device() == b.device(), "a/b must be on the same CUDA device");

    const int64_t M64 = a.size(0);
    const int64_t K64 = a.size(1);
    const int64_t N64 = b.size(1);
    TORCH_CHECK(b.size(0) == K64, "inner dim mismatch: a is [M,K], b must be [K,N]");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
                "M/N/K are too large for the CUDA kernel");
    TORCH_CHECK((M64 + BM - 1) / BM <= 65535,
                "M exceeds the CUDA grid.y limit for this kernel");

    // 向量化 fast path 要求每个 16-byte chunk 都完整且对齐。
    // 通用边界继续复用已验证的 v4 kernel，避免把 tail 与流水化混成一个实验变量。
    const bool pointers_aligned =
        reinterpret_cast<std::uintptr_t>(a.data_ptr<at::Half>()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(b.data_ptr<at::Half>()) % 16 == 0;
    if (M64 == 0 || N64 == 0 || K64 == 0 || !pointers_aligned ||
        M64 % BM != 0 || N64 % BN != 0 || K64 % BK != 0) {
        return gemm_mma_forward(a, b);
    }

    c10::cuda::CUDAGuard device_guard(a.device());
    auto c = torch::empty({M64, N64}, a.options().dtype(torch::kFloat32));
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);
    dim3 grid(static_cast<unsigned>(N64 / BN), static_cast<unsigned>(M64 / BM));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gemm_mma_pipeline<USE_ASYNC, USE_LDMATRIX, USE_SMEM_PADDING>
        <<<grid, WM * WN * 32, 0, stream>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        c.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

torch::Tensor gemm_mma_vec_forward(torch::Tensor a, torch::Tensor b) {
    return gemm_mma_pipeline_forward<false, false, false>(a, b);
}

torch::Tensor gemm_mma_async_forward(torch::Tensor a, torch::Tensor b) {
    return gemm_mma_pipeline_forward<true, false, false>(a, b);
}

torch::Tensor gemm_mma_ldmatrix_forward(torch::Tensor a, torch::Tensor b) {
    return gemm_mma_pipeline_forward<false, true, false>(a, b);
}

torch::Tensor gemm_mma_ldmatrix_padded_forward(torch::Tensor a, torch::Tensor b) {
    return gemm_mma_pipeline_forward<false, true, true>(a, b);
}

// 公平 benchmark 基线：与 gemm_mma_forward 完全相同的输入/累加/输出语义。
// cuBLAS 按列主序解释矩阵；交换 A/B 后计算 B^T @ A^T，内存中正好是行主序 A @ B。
torch::Tensor gemm_cublas_fp32_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kHalf && b.dtype() == torch::kHalf, "a/b must be fp16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a/b must be contiguous");
    TORCH_CHECK(a.device() == b.device(), "a/b must be on the same CUDA device");

    const int64_t M64 = a.size(0);
    const int64_t K64 = a.size(1);
    const int64_t N64 = b.size(1);
    TORCH_CHECK(b.size(0) == K64, "inner dim mismatch: a is [M,K], b must be [K,N]");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
                "M/N/K are too large for cuBLAS int dimensions");

    c10::cuda::CUDAGuard device_guard(a.device());
    auto c = torch::empty({M64, N64}, a.options().dtype(torch::kFloat32));
    if (M64 == 0 || N64 == 0)
        return c;
    if (K64 == 0) {
        c.zero_();
        return c;
    }

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data_ptr<at::Half>(), CUDA_R_16F, N,
        a.data_ptr<at::Half>(), CUDA_R_16F, K,
        &beta,
        c.data_ptr<float>(), CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
    return c;
}
