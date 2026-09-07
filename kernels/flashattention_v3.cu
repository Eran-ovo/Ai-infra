#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------------
// FlashAttention v3：一 warp 一行 Q（FA2 的核心重划分思想）
//
// 对比 v2（一线程一行）：
//   v2: block=32线程=1 warp，每线程管一整行(D=64) -> q_reg[64]+acc[64] 寄存器爆炸
//       occupancy 2%，内存延迟无法隐藏
//   v3: block=256线程=8 warp，每 warp 管一行，warp 内 32 lane 分摊 D=64
//       每线程只需 q_reg[2]+acc[2]，寄存器骤降 -> occupancy 起飞
//
// 每 warp 内部数据分布（D=64, warp=32, num_q=2）：
//   lane i 负责第 [i*2, i*2+1] 两维（连续 2 列，合并访存友好）
//   点积 q·Kc：每 lane 算 2 维部分积 -> __shfl_xor 归约 -> 全 warp 得到同一 dot
//   S[Bc]：不再每线程一份！lane i 只负责 c=i 的分数（Bc=32=warp，正好 1 个）
// ---------------------------------------------------------------------------
#define WARP 32
#define Br   256              // 每 Block 线程数 = 8 warp = 8 行 Q
#define Bc   32               // 每轮搬 32 行 K/V（恰好 = WARP，每 lane 一个分数）
#define D    64               // 每行向量维度
#define ROWS_PER_BLK (Br / WARP)   // 8 行 Q / block
#define DQ   (D / WARP)            // 每 lane 负责的维数 = 2

template<bool IS_CAUSAL>
__global__ void flash_fwd(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          float* __restrict__ O, int N)
{
    const int q_blk   = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane    = tid % WARP;    // warp 内编号 0-31
    const int warp_id = tid / WARP;    // 第几个 warp 0-7（= block 内第几行）

    // 本 warp 负责的 Q 行号：block 内第 warp_id 行
    const int q_row = q_blk * ROWS_PER_BLK + warp_id;
    // 【死锁修复】不能用 if(q_row>=N) return：block 内 8 个 warp 的 q_row 不同，
    // 部分 warp 提前退出后，留下的 warp 会在 __syncthreads() 上永久等待。
    // 改为标记 valid，越界 warp 全程陪跑（搬运/同步照做），只是不写回。
    const bool valid = (q_row < N);

    // --- SRAM：全 block 共享的 K/V 块 + Q 行 --------------------------------
    // 【bank conflict 修复】Ktile/Vtile 按 [c][d] 行主序存储，4b/4e 中 warp 内
    // 32 个 lane 用不同 c、相同 d 访问 -> 地址间隔 D 个 float，D=64 是 32 的倍数
    // -> 全撞同一 bank（32 路冲突，ncu 实测每条 load ~7 次冲突）。
    // 经典解法：每行 +1 float padding，步长变 D+1=65（与 32 互质），冲突消除。
    // Qtile 不用 pad：4b 中全 warp 读同一地址 Qtile[warp_id][d]，走广播无冲突。
    __shared__ float Ktile[Bc][D+1];   // 32x65
    __shared__ float Vtile[Bc][D+1];   // 32x65
    __shared__ float Qtile[ROWS_PER_BLK][D];  // 8x64 = 2KB，本 block 的 8 行 Q

    // Q 行一次性协作搬入 smem（block 内统一循环，无发散；越界行补零陪跑）
    for (int i = lane*DQ; i < lane*DQ + DQ; i ++) {
        const int r = warp_id, d = i;
        const int qr = q_blk * ROWS_PER_BLK + r;
        Qtile[r][d] = (qr < N) ? Q[qr * D + d] : 0.0f;
    }
    __syncthreads();

    // --- 在线 Softmax 运行量（warp 级，全 warp 值一致）----------------------
    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float acc[DQ] = {0.0f};            // 本 lane 的 DQ 维分子累加

    const int num_kv_blks = (N + Bc - 1) / Bc;
    // causal 块级截断：本 block 最大行号 q_blk*ROWS_PER_BLK + ROWS_PER_BLK-1
    // 它需要的最右 kv 块 = q_row_max / Bc + 1（全 block 一致，不拖 syncthreads）
    const int q_row_max = q_blk * ROWS_PER_BLK + ROWS_PER_BLK - 1;
    //当Bv_blk*Bc > q_row_max时，说明后续的kv块对本block的q行没有贡献，可以提前结束循环
    //所以kv_end = IS_CAUSAL ? (q_row_max / Bc + 1) : num_kv_blks;
    const int kv_end = IS_CAUSAL ? (q_row_max / Bc + 1) : num_kv_blks;

    for (int kv_blk = 0; kv_blk < kv_end; ++kv_blk) {

        // 4a. 全 block 协作搬 K/V：256 线程，每线程搬 D*Bc/256 = 8 个元素
        //     tid 连续 -> 同行内 d 连续，合并访存
        for (int idx = tid; idx < Bc * D; idx += Br) {
            int c = idx / D, d = idx % D;
            int r = kv_blk * Bc + c;
            Ktile[c][d] = (r < N) ? K[r * D + d] : 0.0f;
            Vtile[c][d] = (r < N) ? V[r * D + d] : 0.0f;
        }
        __syncthreads();

        // 4b. 每 lane 只负责一个分数 c = lane（Bc == WARP 的巧妙之处）
        //     每 lane 独立算完整点积 dot(Q[本warp行], Ktile[lane])，无 shuffle。
        //     （旧代码用 xor 归约把 32 个 lane 的"不同 K 行×不同 q 维"混加，
        //      每个 lane 得到的是 32 行 K 的混合值而非自己那行的 dot；且该
        //      shuffle 在 causal mask 下发散使用会直接死锁。）
        float S = -1e20f;
        const int kv_col = kv_blk * Bc + lane;//warp内一个线程算一个分数，kv_col是该线程对应的K行号
        // 越界 warp（valid=false）不参与计算，S 保持 -1e20 -> p=0，不影响归约
        bool visible = valid && (!IS_CAUSAL || kv_col <= q_row) && (kv_col < N);
        if (visible) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d)
                dot += Qtile[warp_id][d] * Ktile[lane][d];
            S = dot / sqrtf((float)D);
        }

        // 4c. warp 归约求 row_max（对 S 取 max；不可见 lane 的 -1e20 不影响）
        float row_max = S;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

        const float m_new  = fmaxf(m_prev, row_max);
        const float alpha  = expf(m_prev - m_new);   // 旧统计修正
        const float p      = expf(S - m_new);        // 本 lane 那个分数的权重
        // warp 归约求 row_sum（p 之和；不可见 lane p=exp(-1e20-m)=0，不影响）
        float row_sum = p;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);
        const float l_new = l_prev * alpha + row_sum;

        // 4e. 分子累加：本 lane 的 DQ 维，pv = Σ_c weight[c] * V[c][本lane的维]
        //     需要所有 32 个分数的权重 p -> 从各 lane 广播收集
        #pragma unroll
        for (int i = 0; i < DQ; ++i) {
            float pv = 0.0f;
            const int d = lane * DQ + i;             // 本 lane 负责的维度
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                // 取 lane c 上存的 p 值（权重）广播给全 warp
                float w = __shfl_sync(0xffffffff, p, c);
                pv += w * Vtile[c][d];
            }
            acc[i] = acc[i] * alpha + pv;
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();               // 下一轮复用 Ktile/Vtile，先等算完
    }

    // --- 5. 归一化写回：本 lane 的 DQ 维（越界 warp 不写）-------------------
    if (valid) {
        #pragma unroll
        for (int i = 0; i < DQ; ++i)
            O[q_row * D + lane * DQ + i] = acc[i] / l_prev;
    }
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
        const int j_end = IS_CAUSAL ? (i + 1) : N;
        float row_max = -1e20f;
        for (int j = 0; j < j_end; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[i * D + d] * K[j * D + d];
            row_max = fmaxf(row_max, dot / scale);
        }
        float sum = 0.0f;
        for (int j = 0; j < j_end; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Q[i * D + d] * K[j * D + d];
            sum += expf(dot / scale - row_max);
        }
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
    float *hRefC = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * D; ++i) {
        hQ[i] = (rand() % 20 - 10) / 10.0f;
        hK[i] = (rand() % 20 - 10) / 10.0f;
        hV[i] = (rand() % 20 - 10) / 10.0f;
    }

    attention_cpu<false>(hQ, hK, hV, hRef, N);
    attention_cpu<true>(hQ, hK, hV, hRefC, N);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes); cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    // 每 block 管 8 行 Q，grid = N/8
    const int grid = (N + ROWS_PER_BLK - 1) / ROWS_PER_BLK;

    flash_fwd<false><<<grid, Br>>>(dQ, dK, dV, dO, N);
    flash_fwd<true><<<grid, Br>>>(dQ, dK, dV, dO, N);
    cudaDeviceSynchronize();

    // 交错计时
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
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
    std::cout << "FlashAttention-v3 N=" << N << " D=" << D
              << " | full: " << ms_full << " ms"
              << " | causal: " << ms_causal << " ms"
              << " | speedup: " << ms_full / ms_causal << "x" << std::endl;

    // 对拍 causal
    flash_fwd<true><<<grid, Br>>>(dQ, dK, dV, dO, N);
    cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
    float err = 0.0f;
    for (int i = 0; i < N * D; ++i) err = fmaxf(err, fabsf(hO[i] - hRefC[i]));
    std::cout << (err < 1e-3 ? "PASS!" : "FAIL!") << " causal maxErr=" << err
              << " Sample O[0]=" << hO[0] << " Ref=" << hRefC[0] << std::endl;

    // 对拍 full
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
