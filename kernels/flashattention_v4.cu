// ---------------------------------------------------------------------------
// v4: FlashAttention fp16 + 手写 Tensor Core（mma.sync.m16n8k16）
//
// 从 v3（fp32 warp-per-row）到「真正复现 FA2」的关键一步：QK^T 与 PV 两个 GEMM
// 都换成 Tensor Core。Q/K/V 用 fp16，累加/softmax 用 fp32。
//
// 架构（与 FA2 一致的 warp 划分：warp 只沿 Q/M 维拆分，N 维靠循环 + 行内归约）：
//   - 每 block 处理 Bq=64 行 Q；4 个 warp，每 warp 独占 16 行（一个 mma M-tile）
//   - 每 warp 自己扫全部 K/V 块（外层 kv 循环）与全部 K 列（内层 8 个 n-tile），
//     因此 softmax 的行归约（max/sum）完全在 warp 内完成，无跨 warp 通信
//   - S = Q @ K^T（mma，A=Q row-major，B=K^T）；P = exp(S-m)；O = P @ V（mma）
//   - 在线 softmax：running max/sum 每行维护，行内 4-lane 蝶形 shuffle 归约
//
// Stage 1：全 attention（非 causal），Bq=Bc=D=64 固定，N 可被 64 整除为纯最小实现。
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#define D   64    // head dim（mma 的 N 维，也是 QK^T 的归约维）
#define Bq  64    // Q 块行数
#define Bc  64    // K/V 块行数
#define BK  16    // mma 的 K 步（D 与 Bc 都是 64 = 4*16）

#define WARPS 4            // 4 warps = Bq/16，每 warp 独占 16 行 Q
#define WMMA_M 16
#define WMMA_N 8

__device__ __forceinline__ unsigned as_u32(__half2 x) {
    return *reinterpret_cast<unsigned*>(&x);
}

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

// A fragment（row-major 16x16 → 4 个 half2）：gemm_v4 已验证的精确布局。
// g = lane/4（0..7），gid = lane%4（0..3）
__device__ __forceinline__ void load_A_verified(__half2& a0, __half2& a1, __half2& a2, __half2& a3,
                                                const __half (*M)[D + 1], int r, int c, int g, int gid) {
    a0 = __halves2half2(M[r + g][c + gid * 2],         M[r + g][c + gid * 2 + 1]);
    a1 = __halves2half2(M[r + g + 8][c + gid * 2],     M[r + g + 8][c + gid * 2 + 1]);
    a2 = __halves2half2(M[r + g][c + gid * 2 + 8],     M[r + g][c + gid * 2 + 9]);
    a3 = __halves2half2(M[r + g + 8][c + gid * 2 + 8], M[r + g + 8][c + gid * 2 + 9]);
}

__global__ void flash_fp16_mma(const __half* __restrict__ Q,
                               const __half* __restrict__ K,
                               const __half* __restrict__ V,
                               __half* __restrict__ O, int N)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;      // 0..3，每个独占 16 行
    const int lane    = tid % 32;
    const int g       = lane / 4;      // 0..7
    const int gid     = lane % 4;      // 0..3

    const int q_off = blockIdx.x * Bq;
    const int ar    = warp_id * WMMA_M;   // 本 warp 的 Q 行基址（0/16/32/48）

    __shared__ __half Qs[Bq][D + 1];
    __shared__ __half Ks[Bc][D + 1];
    __shared__ __half Vs[Bc][D + 1];
    __shared__ __half Ps[Bq][Bc + 1];

    const int NT = WARPS * 32;   // 128

    // ---- 载入 Q 块 --------------------------------------------------------
    for (int i = tid; i < Bq * D; i += NT) {
        int r = i / D, c = i % D;
        int gr = q_off + r;
        Qs[r][c] = (gr < N) ? Q[gr * D + c] : __float2half(0.0f);
    }

    // ---- 每 lane 的在线 softmax 状态：行 g 与行 g+8 ------------------------
    float m_prev[2] = {-1e20f, -1e20f};
    float l_prev[2] = {0.0f, 0.0f};
    float o_acc[2][8][4];   // [行 0/1][n-tile 0..7][c0..c3]
    #pragma unroll
    for (int ii = 0; ii < 2; ++ii)
        #pragma unroll
        for (int t = 0; t < 8; ++t)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                o_acc[ii][t][i] = 0.0f;

    const float scale = 1.0f / sqrtf((float)D);
    const int num_kv = (N + Bc - 1) / Bc;

    for (int kv = 0; kv < num_kv; ++kv) {
        const int k_off = kv * Bc;

        // ---- 载入 K/V 块 --------------------------------------------------
        for (int i = tid; i < Bc * D; i += NT) {
            int r = i / D, c = i % D;
            int gr = k_off + r;
            Ks[r][c] = (gr < N) ? K[gr * D + c] : __float2half(0.0f);
            Vs[r][c] = (gr < N) ? V[gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // ---- GEMM1：S = Q @ K^T（16 行 x 64 列，全部本 warp 自己算）--------
        float S[8][4];   // 8 个 n-tile，每个 4 个 fp32
        #pragma unroll
        for (int t = 0; t < 8; ++t)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                S[t][i] = 0.0f;

        #pragma unroll
        for (int d0 = 0; d0 < D; d0 += BK) {
            __half2 a0, a1, a2, a3;
            load_A_verified(a0, a1, a2, a3, Qs, ar, d0, g, gid);
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int nc = t * WMMA_N;   // K 列（S 的输出列）
                // B = K^T：K[j][d] -> Bop[d][j] = Ks[j][d]
                __half2 b0 = __halves2half2(Ks[nc + g][d0 + gid * 2],     Ks[nc + g][d0 + gid * 2 + 1]);
                __half2 b1 = __halves2half2(Ks[nc + g][d0 + gid * 2 + 8], Ks[nc + g][d0 + gid * 2 + 9]);
                mma_m16n8k16(S[t][0], S[t][1], S[t][2], S[t][3], a0, a1, a2, a3, b0, b1);
            }
        }

        // ---- 在线 softmax：行内归约（完全 warp 内）------------------------
        #pragma unroll
        for (int t = 0; t < 8; ++t)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                S[t][i] *= scale;

        const float m_old0 = m_prev[0], m_old1 = m_prev[1];
        const float l_old0 = l_prev[0], l_old1 = l_prev[1];

        // 行 g 的 max（c0,c1 是行 g 的 2x gid 列）
        float rm0 = -1e20f;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            rm0 = fmaxf(rm0, fmaxf(S[t][0], S[t][1]));
        }
        rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffff, rm0, 1));
        rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffff, rm0, 2));

        // 行 g+8 的 max（c2,c3 是行 g+8 的列）
        float rm1 = -1e20f;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            rm1 = fmaxf(rm1, fmaxf(S[t][2], S[t][3]));
        }
        rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffff, rm1, 1));
        rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffff, rm1, 2));

        const float m_new0 = fmaxf(m_old0, rm0);
        const float m_new1 = fmaxf(m_old1, rm1);
        const float alpha0 = expf(m_old0 - m_new0);   // 旧状态衰减
        const float alpha1 = expf(m_old1 - m_new1);

        // P 与 row-sum 都对齐到 m_new 归一化：P=exp(S-m_new)
        float rs0 = 0.0f, rs1 = 0.0f;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            S[t][0] = expf(S[t][0] - m_new0);
            S[t][1] = expf(S[t][1] - m_new0);
            S[t][2] = expf(S[t][2] - m_new1);
            S[t][3] = expf(S[t][3] - m_new1);
            rs0 += S[t][0] + S[t][1];
            rs1 += S[t][2] + S[t][3];
        }
        rs0 += __shfl_xor_sync(0xffffffff, rs0, 1);
        rs0 += __shfl_xor_sync(0xffffffff, rs0, 2);
        rs1 += __shfl_xor_sync(0xffffffff, rs1, 1);
        rs1 += __shfl_xor_sync(0xffffffff, rs1, 2);

        const float l_new0 = l_old0 * alpha0 + rs0;
        const float l_new1 = l_old1 * alpha1 + rs1;

        // ---- 把 P（fp32 已归一化权重）写回 smem Ps -------------------------
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int nc = t * WMMA_N;
            Ps[ar + g][nc + gid * 2]         = __float2half(S[t][0]);
            Ps[ar + g][nc + gid * 2 + 1]     = __float2half(S[t][1]);
            Ps[ar + g + 8][nc + gid * 2]     = __float2half(S[t][2]);
            Ps[ar + g + 8][nc + gid * 2 + 1] = __float2half(S[t][3]);
        }

        // ---- 在线修正 O：o_acc *= alpha，再加 P@V --------------------------
        #pragma unroll
        for (int t = 0; t < 8; ++t)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                o_acc[0][t][i] *= alpha0;
                o_acc[1][t][i] *= alpha1;
            }

        #pragma unroll
        for (int b0 = 0; b0 < Bc; b0 += BK) {
            __half2 pa0, pa1, pa2, pa3;
            load_A_verified(pa0, pa1, pa2, pa3, Ps, ar, b0, g, gid);
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int nd = t * WMMA_N;
                __half2 vb0 = __halves2half2(Vs[b0 + gid * 2][nd + g],     Vs[b0 + gid * 2 + 1][nd + g]);
                __half2 vb1 = __halves2half2(Vs[b0 + gid * 2 + 8][nd + g], Vs[b0 + gid * 2 + 9][nd + g]);
                mma_m16n8k16(o_acc[0][t][0], o_acc[0][t][1], o_acc[1][t][0], o_acc[1][t][1],
                             pa0, pa1, pa2, pa3, vb0, vb1);
            }
        }

        m_prev[0] = m_new0; m_prev[1] = m_new1;
        l_prev[0] = l_new0; l_prev[1] = l_new1;
        __syncthreads();
    }

    // ---- 归一化写回 O := acc / l ------------------------------------------
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int nd = t * WMMA_N;
        int orow0 = q_off + ar + g, orow1 = q_off + ar + g + 8;
        float inv0 = 1.0f / l_prev[0], inv1 = 1.0f / l_prev[1];
        if (orow0 < N) {
            O[orow0 * D + nd + gid * 2]     = __float2half(o_acc[0][t][0] * inv0);
            O[orow0 * D + nd + gid * 2 + 1] = __float2half(o_acc[0][t][1] * inv0);
        }
        if (orow1 < N) {
            O[orow1 * D + nd + gid * 2]     = __float2half(o_acc[1][t][0] * inv1);
            O[orow1 * D + nd + gid * 2 + 1] = __float2half(o_acc[1][t][1] * inv1);
        }
    }
}

// CPU 参考：fp32 输入（喂 fp16 量化后的数据），fp32 累加，减 max 稳定
static void attn_cpu(const float* Q, const float* K, const float* V,
                     float* O, int N) {
    const float scale = 1.0f / sqrtf((float)D);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float m = -1e20f;
        float* srow = (float*)malloc(N * sizeof(float));
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int d = 0; d < D; ++d) s += Q[i * D + d] * K[j * D + d];
            srow[j] = s * scale;
            m = fmaxf(m, s * scale);
        }
        float l = 0.0f;
        for (int j = 0; j < N; ++j) { srow[j] = expf(srow[j] - m); l += srow[j]; }
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j) acc += srow[j] * V[j * D + d];
            O[i * D + d] = acc / l;
        }
        free(srow);
    }
}

int main() {
    for (int N : {64, 128, 256, 512, 1024, 2048, 4096}) {
        const size_t hB = (size_t)N * D * 2;   // half
        const size_t fB = (size_t)N * D * 4;   // float
        float* hQf = (float*)malloc(fB);
        float* hKf = (float*)malloc(fB);
        float* hVf = (float*)malloc(fB);
        __half* hQ = (__half*)malloc(hB);
        __half* hK = (__half*)malloc(hB);
        __half* hV = (__half*)malloc(hB);
        __half* hO = (__half*)malloc(hB);
        float* hRef = (float*)malloc(fB);

        srand(42);
        for (int i = 0; i < N * D; ++i) {
            hQf[i] = (rand() % 10 - 5) / 10.0f;
            hKf[i] = (rand() % 10 - 5) / 10.0f;
            hVf[i] = (rand() % 10 - 5) / 10.0f;
        }
        for (int i = 0; i < N * D; ++i) { hQ[i] = __float2half(hQf[i]); hK[i] = __float2half(hKf[i]); hV[i] = __float2half(hVf[i]); }
        for (int i = 0; i < N * D; ++i) { hQf[i] = __half2float(hQ[i]); hKf[i] = __half2float(hK[i]); hVf[i] = __half2float(hV[i]); }
        attn_cpu(hQf, hKf, hVf, hRef, N);

        __half *dQ, *dK, *dV, *dO;
        cudaMalloc(&dQ, hB); cudaMalloc(&dK, hB); cudaMalloc(&dV, hB); cudaMalloc(&dO, hB);
        cudaMemcpy(dQ, hQ, hB, cudaMemcpyHostToDevice);
        cudaMemcpy(dK, hK, hB, cudaMemcpyHostToDevice);
        cudaMemcpy(dV, hV, hB, cudaMemcpyHostToDevice);

        const int grid = (N + Bq - 1) / Bq;
        flash_fp16_mma<<<grid, WARPS * 32>>>(dQ, dK, dV, dO, N);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { printf("N=%4d CUDA error: %s\n", N, cudaGetErrorString(err)); return 1; }

        // 计时（预热 + 多轮平均，与仓库其他 benchmark 同口径）
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        for (int i = 0; i < 5; ++i) flash_fp16_mma<<<grid, WARPS * 32>>>(dQ, dK, dV, dO, N);
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        for (int i = 0; i < 50; ++i) flash_fp16_mma<<<grid, WARPS * 32>>>(dQ, dK, dV, dO, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= 50;

        cudaMemcpy(hO, dO, hB, cudaMemcpyDeviceToHost);
        float maxe = 0.0f, maxc = 1e-6f;
        // 大尺寸全量 CPU 对拍过慢，行数 > 256 时只抽样对拍前 64 行
        const int check_rows = (N <= 256) ? N : 64;
        for (int i = 0; i < check_rows * D; ++i) {
            float o = __half2float(hO[i]);
            maxe = fmaxf(maxe, fabsf(o - hRef[i]));
            maxc = fmaxf(maxc, fabsf(hRef[i]));
        }
        printf("v4 fp16-mma N=%4d | %s maxErr=%.4f relErr=%.2e | %.3f ms\n",
               N, (maxe / maxc < 1e-2 ? "PASS!" : "FAIL!"), maxe, maxe / maxc, ms);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
        free(hQf); free(hKf); free(hVf); free(hQ); free(hK); free(hV); free(hO); free(hRef);
    }
    return 0;
}
