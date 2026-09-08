// ---------------------------------------------------------------------------
// v4: 手写 Tensor Core GEMM（fp16 输入 / fp32 累加）
//
// 用 mma.sync.aligned.m16n8k16.row.col 指令，绕开 wmma API 与 CUTLASS，
// 亲手管理 fragment 布局 + shared memory 搬运。这是 v2（fp32 FMA）→ 工业级
// GEMM 之间的关键一跳：同样的访存模型，把"每周期 32 lane 各 1 次 FMA"换成
// "每个 warp 每周期 16x8x16 矩阵乘"。
//
// 数据流（每 warp）：
//   global fp16 A/B --load--> smem tile --fragment--> mma.sync --> fp32 acc
//   --write--> global C
// 本版为"正确性优先"：手动 fragment 加载、无 cp.async 双缓冲、无 ldmatrix，
// 先验证 mma 通路与布局，再迭代优化。
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// 块级 tile：每个 block 算 BM x BN 的 C 块，K 方向每轮 BK
#define BM 64
#define BN 64
#define BK 16

// 256 线程 = 8 warps，排列成 4(沿M) x 2(沿N)
// warp_m = warp_id/2（0..3），warp_n = warp_id%2（0..1）
// 每个 warp 计算 16 x 32 的 tile = 4 个 m16n8k16
#define WARPS 8
#define WM 4   // M 方向 warp 数
#define WN 2   // N 方向 warp 数
#define WMMA_M 16
#define WMMA_N 8
#define WMMA_K 16

// half2 -> 32bit，便于内联 PTX 传参
// __forceinline__ 让编译器直接展开，避免函数调用开销
// __half2: 两个 half (2x16bit)组成的向量
// reinterpret_cast<unsigned*>(&x) 将 __half2 的地址 reinterpret_cast 为 unsigned*，然后解引用得到 unsigned 类型的值
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
        // m16n8k16: 矩阵形状
        // row.col: A 按行，B 按列
        // f32.f16.f16.f32: C 是 fp32，A/B 是 fp16 
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)// 输入/输出操作数（累加器）f表示浮点数寄存器
        : "r"(as_u32(a0)), "r"(as_u32(a1)), "r"(as_u32(a2)), "r"(as_u32(a3)),
          "r"(as_u32(b0)), "r"(as_u32(b1)));// 输入操作数（A/B 矩阵）r表示通用寄存器
}

__global__ void gemm_mma(const __half* __restrict__ A,
                         const __half* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K)
{
    // 本 block 负责的 C 块起点
    const int m_base = blockIdx.y * BM;
    const int n_base = blockIdx.x * BN;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / WN;   // 0..3
    const int warp_n  = warp_id % WN;   // 0..1

    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BK][BN];

    // 本 warp 的 accum：4 个 mma 输出 tile（沿 N 方向 4 个 n8）x 4 个 fp32
    float acc[4][4];
    #pragma unroll//作用：展开循环，减少循环开销
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[t][i] = 0.0f;

    const int lane   = tid % 32;
    const int g      = lane / 4;   // groupID 0..7
    const int gid    = lane % 4;   // threadID_in_group 0..3

    // 本 warp 沿 N 覆盖 32 列 = 4 个 m16n8k16 tile（BN/WN=32，WMMA_N=8）
    const int N_TILES = (BN / WN) / WMMA_N;   // = 4
    const int n_warp_off = warp_n * (BN / WN); // = warp_n * 32

    for (int kt = 0; kt < K; kt += BK) {
        // --- 1. 全 block 协作搬 A/B tile 进 smem -------------------------
        // As: BM x BK = 64x16 = 1024 halves，256 线程 x 4 个（步长=线程数 256）
        const int NT = WARPS * 32;                        // 256
        for (int i = tid; i < BM * BK; i += NT) {
            int r = i / BK, c = i % BK;
            int gr = m_base + r, gc = kt + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        // Bs: BK x BN = 16x64 = 1024 halves
        for (int i = tid; i < BK * BN; i += NT) {
            int r = i / BN, c = i % BN;
            int gr = kt + r, gc = n_base + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        // --- 2. 每 warp 从 smem 加载 fragment + mma ----------------------
        // 本 warp 的 A 片段布局（m16n8k16 规范，row=g / col=2*gid 打底）：
        //   a0=(g, 2*gid)  a1=(g+8, 2*gid)  a2=(g, 2*gid+8)  a3=(g+8, 2*gid+8)
        int ar = warp_m * WMMA_M;
        __half2 a0 = __halves2half2(As[ar + g][gid * 2],         As[ar + g][gid * 2 + 1]);
        __half2 a1 = __halves2half2(As[ar + g + 8][gid * 2],     As[ar + g + 8][gid * 2 + 1]);
        __half2 a2 = __halves2half2(As[ar + g][gid * 2 + 8],     As[ar + g][gid * 2 + 9]);
        __half2 a3 = __halves2half2(As[ar + g + 8][gid * 2 + 8], As[ar + g + 8][gid * 2 + 9]);

        #pragma unroll
        for (int t = 0; t < N_TILES; ++t) {
            int nc = n_warp_off + t * WMMA_N;   // 本 mma tile 的 N 列基址
            // B fragment: k = gid*2 / gid*2+1 / +8 / +9，n = g
            __half2 b0 = __halves2half2(Bs[gid * 2][nc + g],     Bs[gid * 2 + 1][nc + g]);
            __half2 b1 = __halves2half2(Bs[gid * 2 + 8][nc + g], Bs[gid * 2 + 9][nc + g]);

            mma_m16n8k16(acc[t][0], acc[t][1], acc[t][2], acc[t][3],
                         a0, a1, a2, a3, b0, b1);
        }
        __syncthreads();
    }

    // --- 3. 写回 C：本 warp 的 16x32 块变成 4 个 n8 tile -----------------
    #pragma unroll
    for (int t = 0; t < N_TILES; ++t) {
        int nc = n_base + n_warp_off + t * WMMA_N;
        int mr = m_base + warp_m * WMMA_M;
        // c0 = (g, gid*2), c1 = (g, gid*2+1), c2 = (g+8, gid*2), c3 = (g+8, gid*2+1)
        int r0 = mr + g, c0 = nc + gid * 2;
        int r2 = mr + g + 8;
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

// CPU fp32 参考（OpenMP 并行，避免 4096^3 单线程跑几分钟）
static void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static double gflops_of(int M, int N, int K, double ms) {
    return 2.0 * M * N * K / (ms / 1e3) / 1e9;
}

int main() {
    for (int dim : {1024, 2048, 4096}) {
        const int M = dim, N = dim, K = dim;
        const size_t bA = (size_t)M * K * 2;   // fp16 = 2 字节
        const size_t bB = (size_t)K * N * 2;
        const size_t bC = (size_t)M * N * 4;   // fp32 输出
        const size_t fA = (size_t)M * K * 4;   // float 版（用于 CPU 参考 + 转换）

        float* hAf = (float*)malloc(fA);
        float* hBf = (float*)malloc(fA ? (size_t)K * N * 4 : 0);
        __half* hA = (__half*)malloc(bA);
        __half* hB = (__half*)malloc(bB);
        float* hC = (float*)malloc(bC);
        float* hRef = (float*)malloc(bC);

        srand(42);
        for (int i = 0; i < M * K; ++i) hAf[i] = (rand() % 6) / 5.0f;   // [0,1] 避免溢出
        for (int i = 0; i < K * N; ++i) hBf[i] = (rand() % 6) / 5.0f;
        // 转 fp16 再转回 float：让 CPU 参考与 GPU 吃同一份 fp16 量化后的输入，
        // 否则 0.2/0.4 这类 fp16 无法精确表示的值会让两边基准不同，误报 FAIL
        for (int i = 0; i < M * K; ++i) { hA[i] = __float2half(hAf[i]); hAf[i] = __half2float(hA[i]); }
        for (int i = 0; i < K * N; ++i) { hB[i] = __float2half(hBf[i]); hBf[i] = __half2float(hB[i]); }
        gemm_cpu(hAf, hBf, hRef, M, N, K);

        __half *dA, *dB;
        float* dC;
        cudaMalloc(&dA, bA); cudaMalloc(&dB, bB); cudaMalloc(&dC, bC);
        cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice);

        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_mma<<<grid, WARPS * 32>>>(dA, dB, dC, M, N, K);
        cudaDeviceSynchronize();

        // 计时（交错多次取均值）
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) gemm_mma<<<grid, WARPS * 32>>>(dA, dB, dC, M, N, K);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms_mma = 0; cudaEventElapsedTime(&ms_mma, s, e); ms_mma /= 10;

        // cuBLAS fp16 对比（fp16 in / fp32 compute，切到 fp32 out）
        cublasHandle_t h; cublasCreate(&h);
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     dB, CUDA_R_16F, N, dA, CUDA_R_16F, K, &beta,
                     dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i)
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         dB, CUDA_R_16F, N, dA, CUDA_R_16F, K, &beta,
                         dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms_cublas = 0; cudaEventElapsedTime(&ms_cublas, s, e); ms_cublas /= 10;
        cublasDestroy(h);

        // 对拍（mma 结果重跑一次干净写入）
        gemm_mma<<<grid, WARPS * 32>>>(dA, dB, dC, M, N, K);
        cudaMemcpy(hC, dC, bC, cudaMemcpyDeviceToHost);
        float err = 0.0f, maxc = 1e-6f;
        for (int i = 0; i < M * N; ++i) {
            err = fmaxf(err, fabsf(hC[i] - hRef[i]));
            maxc = fmaxf(maxc, fabsf(hRef[i]));
        }
        float rel = err / maxc;

        printf("v4 mma fp16 M=N=K=%d | mma %.3f ms (%.1f GFLOPS) | cuBLAS fp16 %.3f ms (%.1f GFLOPS) | %s maxErr=%.4f relErr=%.2e\n",
               dim, ms_mma, gflops_of(M, N, K, ms_mma),
               ms_cublas, gflops_of(M, N, K, ms_cublas),
               rel < 1e-3 ? "PASS!" : "FAIL!", err, rel);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hAf); free(hBf); free(hA); free(hB); free(hC); free(hRef);
    }
    return 0;
}
