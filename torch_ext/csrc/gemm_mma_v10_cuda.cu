// GEMM v10：BK=32 的 Tensor Core 实验版
//
// v8 的核心路径是：
//   16-byte global->shared copy -> padded shared layout -> ldmatrix -> mma.sync
// v9 在此基础上加入 cp.async 双缓冲，但 BK=16 时一个 tile 的计算窗口太短，
// commit/wait/barrier 的开销没有完全被隐藏。
//
// v10 的唯一主要变量是把 K tile 从 16 增大到 32：
//   一个 stage 先搬入 64x32 的 A tile 和 32x64 的 B tile，
//   再在同一个 stage 内连续执行两个 k-slice：k=[0,16)、[16,32)。
// 这样每次预取后有两倍 MMA 工作量，给 cp.async 更长的重叠窗口。
//
// 该文件刻意独立于 gemm_mma_cuda.cu：v4-v9 的实验结果和代码不被 v10
// 的 tile / copy mapping 改动污染，便于做可复现的单变量对照实验。

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "ops.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <climits>
#include <cstdint>

namespace gemm_mma_v10 {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int K_SLICE = 16;
constexpr int WN = 2;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int THREADS = 256;
constexpr int STAGES = 2;

// 每个 shared-memory 行增加 8 个 half：
// A stride = 40 half = 80B = 20 banks；B stride = 72 half = 144B = 36 banks。
// 两者都是 16B 的倍数，既满足 uint4/cp.async 的自然对齐，也保留 v8 的
// ldmatrix 无冲突布局。真正写入的矩阵仍只有 BMxBK 或 BKxBN，padding 区域
// 不参与数学计算。
constexpr int AS_STRIDE = BK + 8;  // 40 half
constexpr int BS_STRIDE = BN + 8;  // 72 half

__device__ __forceinline__ unsigned as_shared_u32(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// ldmatrix 得到的每个寄存器正好是 MMA 所需的 packed half fragment，
// 因此这里直接把 unsigned register 送给 inline PTX，不再经过 __half2。
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

// 一个 warp 合作加载四个 8x8 fp16 子矩阵，分别组成 16x16 A fragment。
// lane 0..7、8..15、16..23、24..31 提供四组 8 个行首地址。
__device__ __forceinline__ void ldmatrix_x4(
    unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
    const void* row_addr)
{
    const unsigned addr = as_shared_u32(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

// B 使用 .trans，把 shared 中行主序的 8x8 子块装成 mma.row.col
// 需要的列主序 fragment。高 16 个 lane 会复用低 16 个 lane 的地址。
__device__ __forceinline__ void ldmatrix_x2_trans(
    unsigned& r0, unsigned& r1, const void* row_addr)
{
    const unsigned addr = as_shared_u32(row_addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned dst_shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_shared), "l"(src) : "memory");
#else
    // v10 的 dispatch 只在 sm_86 fast path 上使用；这个 fallback 让源码在
    // 较老架构上仍然具有合理的编译语义，但不会被当前 wrapper 调用。
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" :: : "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
#endif
}

// v10 的 copy mapping 与 v8/v9 不同：每个线程同时搬 A 和 B 各 16B。
//
// A tile = 64x32 half = 4096B，正好由 256 个线程各搬一次 16B 完成。
// B tile = 32x64 half = 4096B，同样由全部 256 个线程各搬一次 16B 完成。
// 每个线程发起两条 cp.async，因此单个 stage 共 512 条 16B copy。
__device__ __forceinline__ void copy_tile_async_v10(
    const __half* A, const __half* B,
    __half* As, __half* Bs,
    int m_base, int n_base, int kt, int K, int N, int tid)
{
    // A: tid*8 个 half 对应一段 16B；row/col 都是合法且 16B 对齐的起点。
    const int a_elem = tid * 8;
    const int a_row = a_elem / BK;
    const int a_col = a_elem % BK;
    cp_async_16(
        As + a_row * AS_STRIDE + a_col,
        A + (m_base + a_row) * K + kt + a_col);

    // B 采用同样的 16B 粒度，但 B tile 的行宽是 BN=64 half。
    const int b_elem = tid * 8;
    const int b_row = b_elem / BN;
    const int b_col = b_elem % BN;
    cp_async_16(
        Bs + b_row * BS_STRIDE + b_col,
        B + (kt + b_row) * N + n_base + b_col);
}

__device__ __forceinline__ void load_and_mma_k_slice(
    const __half* As, const __half* Bs,
    int as_stride, int bs_stride, int k_offset,
    int warp_m, int warp_n, int lane,
    float (&acc)[4][4])
{
    const int a_matrix = lane / 8;
    const int a_row = warp_m * WMMA_M + (a_matrix & 1) * 8 + lane % 8;
    const int a_col = k_offset + (a_matrix / 2) * 8;

    unsigned a_frag[4];
    ldmatrix_x4(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3],
        &As[a_row * as_stride + a_col]);

    // 一个 warp 计算 16x32 的输出区域，因此有 4 个连续的 n8 MMA tile。
    constexpr int N_TILES = (BN / WN) / WMMA_N;  // 32/8 = 4
    const int n_warp_off = warp_n * (BN / WN);

    #pragma unroll
    for (int t = 0; t < N_TILES; ++t) {
        const int nc = n_warp_off + t * WMMA_N;

        // .x2 需要 16 个有效行地址；lane 16..31 复制 lane 0..15。
        const int b_addr_lane = lane & 15;
        const int b_row = k_offset + b_addr_lane;
        unsigned b_frag[2];
        ldmatrix_x2_trans(
            b_frag[0], b_frag[1], &Bs[b_row * bs_stride + nc]);

        mma_m16n8k16_regs(
            acc[t][0], acc[t][1], acc[t][2], acc[t][3],
            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
            b_frag[0], b_frag[1]);
    }
}

__global__ void gemm_mma_v10_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / WN;
    const int warp_n = warp_id % WN;
    const int m_base = blockIdx.y * BM;
    const int n_base = blockIdx.x * BN;

    // 两个 stage 的 A/B layout 都带 padding。总静态 shared memory：
    // 2 * (64*40 + 32*72) * sizeof(half) = 19,456 bytes。
    __shared__ __align__(16) __half As[STAGES][BM][AS_STRIDE];
    __shared__ __align__(16) __half Bs[STAGES][BK][BS_STRIDE];

    float acc[4][4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[t][i] = 0.0f;
    }

    int read_stage = 0;

    // 首个 tile 没有可以被重叠的计算，必须显式等待后才能进入 ldmatrix。
    copy_tile_async_v10(A, B, &As[0][0][0], &Bs[0][0][0],
                        m_base, n_base, 0, K, N, tid);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int kt = 0; kt < K; kt += BK) {
        const int next_kt = kt + BK;
        if (next_kt < K) {
            const int write_stage = read_stage ^ 1;
            // 当前 read_stage 正被 ldmatrix 使用，下一 tile 写入另一个 stage。
            copy_tile_async_v10(
                A, B, &As[write_stage][0][0], &Bs[write_stage][0][0],
                m_base, n_base, next_kt, K, N, tid);
            cp_async_commit();
        }

        const __half* current_a = &As[read_stage][0][0];
        const __half* current_b = &Bs[read_stage][0][0];

        // 一个 BK=32 tile 拆成两个 K=16 fragment。
        // 这两个 slice 共享同一组 FP32 accumulator，数学上等价于
        // 原来的两个 BK=16 tile，但只需在 tile 边界做一次预取/等待。
        load_and_mma_k_slice(
            current_a, current_b, AS_STRIDE, BS_STRIDE, 0,
            warp_m, warp_n, lane, acc);
        load_and_mma_k_slice(
            current_a, current_b, AS_STRIDE, BS_STRIDE, K_SLICE,
            warp_m, warp_n, lane, acc);

        if (next_kt < K) {
            // 等待下一 stage 的 A/B 都完成后，整个 block 再一致切换 stage。
            cp_async_wait_all();
            __syncthreads();
            read_stage ^= 1;
        }
    }

    // 与 v4-v9 使用完全相同的 C fragment 写回布局。
    constexpr int N_TILES = (BN / WN) / WMMA_N;
    const int n_warp_off = warp_n * (BN / WN);
    const int mr = m_base + warp_m * WMMA_M;

    #pragma unroll
    for (int t = 0; t < N_TILES; ++t) {
        const int nc = n_base + n_warp_off + t * WMMA_N;
        const int r0 = mr + lane / 4;
        const int c0 = nc + (lane % 4) * 2;
        const int r2 = r0 + 8;
        C[r0 * N + c0] = acc[t][0];
        C[r0 * N + c0 + 1] = acc[t][1];
        C[r2 * N + c0] = acc[t][2];
        C[r2 * N + c0 + 1] = acc[t][3];
    }
}

torch::Tensor gemm_mma_v10_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kHalf && b.dtype() == torch::kHalf,
                "a/b must be fp16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a/b must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "a/b must be contiguous");
    TORCH_CHECK(a.device() == b.device(),
                "a/b must be on the same CUDA device");

    const int64_t M64 = a.size(0);
    const int64_t K64 = a.size(1);
    const int64_t N64 = b.size(1);
    TORCH_CHECK(b.size(0) == K64,
                "inner dim mismatch: a is [M,K], b must be [K,N]");
    TORCH_CHECK(M64 <= INT_MAX && N64 <= INT_MAX && K64 <= INT_MAX,
                "M/N/K are too large for the CUDA kernel");
    TORCH_CHECK((M64 + BM - 1) / BM <= 65535,
                "M exceeds the CUDA grid.y limit for this kernel");

    // v10 只处理完整的 64x64x32 tile 和 16-byte 对齐输入。
    // 其余情况回退到已验证的 v4 通用 kernel，避免把 tail predicate
    // 与 BK=32 的流水实验混在一起。
    const bool pointers_aligned =
        reinterpret_cast<std::uintptr_t>(a.data_ptr<at::Half>()) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(b.data_ptr<at::Half>()) % 16 == 0;
    if (M64 == 0 || N64 == 0 || K64 == 0 || !pointers_aligned ||
        M64 % BM != 0 || N64 % BN != 0 || K64 % BK != 0) {
        // v10 的核心实验只处理完整 tile；边界和非对齐输入回退到 v4，
        // 这样 Python API 对任意合法矩阵仍保持与其他 GEMM 版本一致的行为。
        return gemm_mma_forward(a, b);
    }

    c10::cuda::CUDAGuard device_guard(a.device());
    auto c = torch::empty({M64, N64}, a.options().dtype(torch::kFloat32));
    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);
    dim3 grid(static_cast<unsigned>(N64 / BN),
              static_cast<unsigned>(M64 / BM));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    gemm_mma_v10_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        c.data_ptr<float>(), M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

}  // namespace gemm_mma_v10

// PyBind11/ops.h 使用全局 forward 符号；device kernel 仍放在 namespace 内，
// 这里提供一个很薄的 host-side wrapper，避免跨 translation unit 的符号名
// 与 namespace 内实现不一致。
torch::Tensor gemm_mma_v10_forward(torch::Tensor a, torch::Tensor b) {
    return gemm_mma_v10::gemm_mma_v10_forward(a, b);
}
