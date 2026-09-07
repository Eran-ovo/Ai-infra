// FlashAttention v3 的 CUDA kernel + PyTorch 包装
// 与 kernels/flashattention_v3.cu 同源（一 warp 一行 + smem padding 消 bank conflict）
// 差异只在：输入输出换成 torch::Tensor，causal 由运行时 bool 派发到模板实例
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP 32
#define Br   256              // 每 block 线程数 = 8 warp = 8 行 Q
#define Bc   32               // 每轮搬 32 行 K/V
#define D    64               // 每行向量维度（编译期固定，与 v3 一致）
#define ROWS_PER_BLK (Br / WARP)
#define DQ   (D / WARP)

template<bool IS_CAUSAL>
__global__ void flash_fwd(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          const float* __restrict__ V,
                          float* __restrict__ O, int N)
{
    const int q_blk   = blockIdx.x;
    const int tid     = threadIdx.x;
    const int lane    = tid % WARP;
    const int warp_id = tid / WARP;

    const int q_row = q_blk * ROWS_PER_BLK + warp_id;
    // 【死锁修复】越界 warp 全程陪跑，不提前 return
    const bool valid = (q_row < N);

    // 【bank conflict 修复】Ktile/Vtile 每行 +1 padding，步长 65 与 32 互质
    __shared__ float Ktile[Bc][D+1];
    __shared__ float Vtile[Bc][D+1];
    __shared__ float Qtile[ROWS_PER_BLK][D];

    for (int i = lane*DQ; i < lane*DQ + DQ; i++) {
        const int qr = q_blk * ROWS_PER_BLK + warp_id;
        Qtile[warp_id][i] = (qr < N) ? Q[qr * D + i] : 0.0f;
    }
    __syncthreads();

    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float acc[DQ] = {0.0f};

    const int num_kv_blks = (N + Bc - 1) / Bc;
    const int q_row_max = q_blk * ROWS_PER_BLK + ROWS_PER_BLK - 1;
    const int kv_end = IS_CAUSAL ? (q_row_max / Bc + 1) : num_kv_blks;

    for (int kv_blk = 0; kv_blk < kv_end; ++kv_blk) {
        for (int idx = tid; idx < Bc * D; idx += Br) {
            int c = idx / D, d = idx % D;
            int r = kv_blk * Bc + c;
            Ktile[c][d] = (r < N) ? K[r * D + d] : 0.0f;
            Vtile[c][d] = (r < N) ? V[r * D + d] : 0.0f;
        }
        __syncthreads();

        float S = -1e20f;
        const int kv_col = kv_blk * Bc + lane;
        bool visible = valid && (!IS_CAUSAL || kv_col <= q_row) && (kv_col < N);
        if (visible) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d)
                dot += Qtile[warp_id][d] * Ktile[lane][d];
            S = dot / sqrtf((float)D);
        }

        float row_max = S;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

        const float m_new  = fmaxf(m_prev, row_max);
        const float alpha  = expf(m_prev - m_new);
        const float p      = expf(S - m_new);

        float row_sum = p;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);
        const float l_new = l_prev * alpha + row_sum;

        #pragma unroll
        for (int i = 0; i < DQ; ++i) {
            float pv = 0.0f;
            const int d = lane * DQ + i;
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                float w = __shfl_sync(0xffffffff, p, c);
                pv += w * Vtile[c][d];
            }
            acc[i] = acc[i] * alpha + pv;
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();
    }

    if (valid) {
        #pragma unroll
        for (int i = 0; i < DQ; ++i)
            O[q_row * D + lane*DQ + i] = acc[i] / l_prev;
    }
}

// PyTorch 侧入口：q/k/v [N, D] float32 contiguous -> o [N, D]
// causal 由运行时 bool 派发到模板实参（编译期特化，无运行时分支开销）
torch::Tensor flashattention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32 && v.dtype() == torch::kFloat32,
                "q/k/v must be float32");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "q/k/v must be [N, D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");

    const int N = q.size(0);
    TORCH_CHECK(q.size(1) == D, "feature dim must be 64 (v3 kernel 编译期固定 D=64)");
    TORCH_CHECK(k.size(0) == N && v.size(0) == N && k.size(1) == D && v.size(1) == D,
                "q/k/v must 同形状 [N, 64]");

    auto o = torch::empty_like(q);
    const int grid = (N + ROWS_PER_BLK - 1) / ROWS_PER_BLK;
    if (causal)
        flash_fwd<true><<<grid, Br>>>(q.data_ptr<float>(), k.data_ptr<float>(),
                                      v.data_ptr<float>(), o.data_ptr<float>(), N);
    else
        flash_fwd<false><<<grid, Br>>>(q.data_ptr<float>(), k.data_ptr<float>(),
                                       v.data_ptr<float>(), o.data_ptr<float>(), N);
    return o;
}
