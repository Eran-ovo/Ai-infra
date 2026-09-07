#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------------
// 尺寸：Q/K/V 都是 N x D。每个 Block 负责 Br 行 Q，每次搬 Bc 行 K/V 进 SRAM。
// 中间 S = Br x Bc 只在寄存器/Shared 里算完即扔，永不落地 HBM。
// ---------------------------------------------------------------------------
#define Br 256   // 每 Block 处理 256 行 Q（一线程一行）
#define Bc 32   // 每轮搬 32 行 K/V
#define D  64   // 每行向量维度


// ---------------------------------------------------------------------------
// flash_fwd：分块 + 在线 Softmax 的 Attention 前向
//
//   数学目标（对每一行 q）：
//     O = softmax(q·K^T/√D) · V
//
//   分块做法：不存整行 N 个分数，边看 KV 块边维护三个运行量：
//     m   = 到目前为止见过的最大分数

//     l   = Σ exp(S - m)        （分母）
//     acc = Σ exp(S - m) · V    （分子，未归一化）
//   每来一块，旧统计乘 alpha = exp(m_old - m_new) 修正后再加新块贡献。
// ---------------------------------------------------------------------------
template<bool IS_CAUSAL>
__global__ void flash_fwd(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          float* __restrict__ O, int N)
{
    // --- 1. 定位：本 Block 管第 q_blk 块 Q，本线程管其中第 tid 行 ---------
    const int q_blk = blockIdx.x;
    const int tid   = threadIdx.x;
    if (tid >= Br) return;                 // 防御：blockDim 不应超过 Br

    const int q_row = q_blk * Br + tid;    // 本线程负责的全局行号
    if (q_row >= N) return;                // 尾块越界防护

    // --- 2. SRAM：本 Block 所有线程共享的 K/V 块 ---------------------------
    __shared__ float Ktile[Bc][D];         // 32x64 = 8KB
    __shared__ float Vtile[Bc][D];         // 32x64 = 8KB

    // --- 3. 本行 Q 载入寄存器，16 个 KV 块循环里复用，不再回 HBM -----------
    float q_reg[D];
    for (int d = 0; d < D; ++d) q_reg[d] = Q[q_row * D + d];

    // --- 4. 在线 Softmax 运行量 -------------------------------------------
    float m_prev = -1e20f;                 // 见过的最大分数
    float l_prev = 0.0f;                   // Σ exp(S - m)
    float acc[D] = {0.0f};                 // Σ exp(S - m) · V

    float S[Bc];                           // 当前 KV 块的 Bc 个分数

    const int num_kv_blks = (N + Bc - 1) / Bc;
    // causal 优化关键：循环上界必须按 BLOCK 粒度算（全 block 一致），
    // 否则 __syncthreads 会被 kv_end 最大的线程拖着走，块数一点省不下来。
    // 本 block 最后一行 q_row_max = q_blk*Br + Br-1，它需要的最右 kv 块是
    // (q_blk*Br + Br-1)/Bc。Br==Bc 时简化为 q_blk+1。
    // 块内逐元素 mask（4b 处）再处理对角线块的精细可见性。
    const int q_row_max = q_blk * Br + Br - 1;
    const int kv_end = IS_CAUSAL ? (q_row_max / Bc + 1) : num_kv_blks;
    for (int kv_blk = 0; kv_blk < kv_end; ++kv_blk) {

        // 4a. 协作搬 K/V 块：Br 个线程每人搬 D/Br 列 × Bc 行，合并访存
        for (int d = tid; d < D; d += Br) {
            for (int c = 0; c < Bc; ++c) {
                const int r = kv_blk * Bc + c;
                if (r < N) {
                    Ktile[c][d] = K[r * D + d];
                    Vtile[c][d] = V[r * D + d];
                } else {
                    Ktile[c][d] = 0.0f;    // 尾块补零
                    Vtile[c][d] = 0.0f;
                }
            }
        }
        __syncthreads();                   // 等全 Block 搬完再算

        // 4b. S[c] = q · Kc^T / √D ，并记录块内最大值
        float row_max = -1e20f;
        for (int c = 0; c < Bc; ++c) {
            const int kv_col = kv_blk * Bc + c;
            if(IS_CAUSAL && kv_col > q_row) {
                S[c] = -1e20f; // causal mask
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) 
                dot += q_reg[d] * Ktile[c][d];
            dot /= sqrtf((float)D);          // 缩放
            S[c] = dot;
            row_max = fmaxf(row_max, dot);
        }//此时 Q[Br*b.x+tid][D] 与 K^T[0-Bc*kv_blk+Bc-1][D] 的点积结果 S[c] 已经算完，且 row_max 是本块的最大分数

        // 4c. 更新 max，并算本块按新 max 归一的 exp 与 sum
        const float m_new = fmaxf(m_prev, row_max);
        float row_sum = 0.0f;
        for (int c = 0; c < Bc; ++c) {
            S[c] = expf(S[c] - m_new);     // 覆盖：分数 → 权重（未归一化）
            row_sum += S[c];
        }

        // 4d. 旧统计修正系数 alpha = exp(m_old - m_new)
        //     max 变大了，旧的 l/acc 都是按旧 max 归一的，要乘 alpha 缩小
        const float alpha = expf(m_prev - m_new);
        const float l_new = l_prev * alpha + row_sum;//修正之前的和并加上这一块的sum

        // 4e. 分子累加：acc=Σ exp(S - m) · V
        for (int d = 0; d < D; ++d) {
            float pv = 0.0f;
            for (int c = 0; c < Bc; ++c) 
                pv += S[c] * Vtile[c][d];
            acc[d] = acc[d] * alpha + pv;
        }
        //acc里存放了当前行的部分的（K/V从0-当前块）分子累加结果，l_new存放了当前行的部分的（K/V从0-当前块）分母累加结果

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();// 下一轮要复用 Ktile，先等算完
    }
    //此时 acc 里存放了当前行的完整的分子累加结果，l_prev 存放了当前行的完整的分母累加结果

    // --- 5. 归一化写回：O = acc / l ----------------------------------------
    for (int d = 0; d < D; ++d) 
        O[q_row * D + d] = acc[d] / l_prev;
}


// ---------------------------------------------------------------------------
// CPU Naive 参考：每行做三遍循环（max → sum → 输出），用于对拍
// IS_CAUSAL=true 时第 i 行只看 j<=i 的 KV（下三角），与 GPU 端 mask 一致
// ---------------------------------------------------------------------------
template<bool IS_CAUSAL>
static void attention_cpu(const float* Q, const float* K, const float* V,
                          float* O, int N)
{
    const float scale = sqrtf((float)D);
    for (int i = 0; i < N; ++i) {
        const int j_end = IS_CAUSAL ? (i + 1) : N;   // causal: 只看 0..i
        // 1) 行最大值
        float row_max = -1e20f;
        for (int j = 0; j < j_end; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[i * D + d] * K[j * D + d];
            row_max = fmaxf(row_max, dot / scale);
        }
        // 2) 分母
        float sum = 0.0f;
        for (int j = 0; j < j_end; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[i * D + d] * K[j * D + d];
            sum += expf(dot / scale - row_max);
        }
        // 3) 输出
        for (int d = 0; d < D; ++d) {
            float out = 0.0f;
            for (int j = 0; j < j_end; ++j) {
                float dot = 0.0f;
                for (int k = 0; k < D; ++k) dot += Q[i * D + k] * K[j * D + k];
                out += expf(dot / scale - row_max) / sum * V[j * D + d];
            }
            O[i * D + d] = out;
        }
    }
}


int main() {
    const int N = 8192;
    const size_t bytes = (size_t)N * D * sizeof(float);

    float *hQ   = (float*)malloc(bytes);
    float *hK   = (float*)malloc(bytes);
    float *hV   = (float*)malloc(bytes);
    float *hO   = (float*)malloc(bytes);
    float *hRef = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * D; ++i) {
        hQ[i] = (rand() % 20 - 10) / 10.0f;
        hK[i] = (rand() % 20 - 10) / 10.0f;
        hV[i] = (rand() % 20 - 10) / 10.0f;
    }

    float *hRefC = (float*)malloc(bytes);   // causal 参考
    attention_cpu<false>(hQ, hK, hV, hRef, N);
    attention_cpu<true>(hQ, hK, hV, hRefC, N);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes); cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    const int grid = (N + Br - 1) / Br;

    // 预热两种模式
    flash_fwd<false><<<grid, Br>>>(dQ, dK, dV, dO, N);
    flash_fwd<true><<<grid, Br>>>(dQ, dK, dV, dO, N);
    cudaDeviceSynchronize();

    // 交错计时（同一次运行内交替测，避免 GPU 频率波动干扰对比）
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    float ms_full = 0, ms_causal = 0, t;
    for (int rep = 0; rep < 3; ++rep) {
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) flash_fwd<false><<<grid, Br>>>(dQ, dK, dV, dO, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e); ms_full += t / 10;

        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) flash_fwd<true><<<grid, Br>>>(dQ, dK, dV, dO, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e); ms_causal += t / 10;
    }
    ms_full /= 3; ms_causal /= 3;
    std::cout << "FlashAttention-v2 N=" << N << " D=" << D
              << " | full: " << ms_full << " ms"
              << " | causal: " << ms_causal << " ms"
              << " | speedup: " << ms_full / ms_causal << "x" << std::endl;

    // 对拍 causal 模式
    flash_fwd<true><<<grid, Br>>>(dQ, dK, dV, dO, N);
    cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
    float err = 0.0f;
    for (int i = 0; i < N * D; ++i) err = fmaxf(err, fabsf(hO[i] - hRefC[i]));
    std::cout << (err < 1e-3 ? "PASS!" : "FAIL!") << " causal maxErr=" << err
              << " Sample O[0]=" << hO[0] << " Ref=" << hRefC[0] << std::endl;

    // 对拍 full 模式
    flash_fwd<false><<<grid, Br>>>(dQ, dK, dV, dO, N);
    cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
    err = 0.0f;
    for (int i = 0; i < N * D; ++i) err = fmaxf(err, fabsf(hO[i] - hRef[i]));
    std::cout << (err < 1e-3 ? "PASS!" : "FAIL!") << " full   maxErr=" << err
              << " Sample O[0]=" << hO[0] << " Ref=" << hRef[0] << std::endl;

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hO); free(hRef); free(hRefC);
    return 0;
}