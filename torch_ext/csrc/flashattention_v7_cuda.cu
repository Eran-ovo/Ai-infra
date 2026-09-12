// FlashAttention v7：只验证一个假设——缓存 Q fragment 并复用 Q/K smem。
// softmax、P fragment、PV MMA 和 launch 形状均与 v5 保持一致。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

#define FLASHATTENTION_V5_NO_MAIN
#define FLASHATTENTION_V5_CACHE_Q_REGS
#define FLASHATTENTION_V5_KERNEL flash_fp16_mma_v7
#include "../../kernels/flashattention_v5.cu"
#undef FLASHATTENTION_V5_KERNEL
#undef FLASHATTENTION_V5_CACHE_Q_REGS
#undef FLASHATTENTION_V5_NO_MAIN

torch::Tensor flashattention_v7_forward(torch::Tensor q,
                                        torch::Tensor k,
                                        torch::Tensor v,
                                        bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kHalf &&
                    k.dtype() == torch::kHalf &&
                    v.dtype() == torch::kHalf,
                "q/k/v must be float16");
    TORCH_CHECK(q.dim() == 2 || q.dim() == 4,
                "q/k/v must be [N, 64] or [B, H, N, 64]");
    TORCH_CHECK(k.dim() == q.dim() && v.dim() == q.dim(),
                "q/k/v must have the same rank");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q/k/v must have the same shape");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q/k/v must be on the same CUDA device");

    c10::cuda::CUDAGuard device_guard(q.device());

    const bool batched = q.dim() == 4;
    const int64_t N64 = batched ? q.size(2) : q.size(0);
    TORCH_CHECK(N64 > 0 && N64 <= INT_MAX, "N must be in [1, INT_MAX]");
    TORCH_CHECK(q.size(-1) == D, "head dim must be 64");

    int64_t batch_heads = 1;
    if (batched) {
        const int64_t batch = q.size(0);
        const int64_t heads = q.size(1);
        TORCH_CHECK(batch > 0 && heads > 0, "B and H must be positive");
        TORCH_CHECK(heads <= 65535 && batch <= 65535 / heads,
                    "B*H must not exceed CUDA grid.y limit 65535");
        batch_heads = batch * heads;
    }

    const int N = static_cast<int>(N64);
    auto o = torch::empty_like(q);
    const dim3 grid((N + Bq - 1) / Bq, static_cast<unsigned>(batch_heads));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (causal) {
        flash_fp16_mma_v7<true><<<grid, WARPS * 32, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()), N);
    } else {
        flash_fp16_mma_v7<false><<<grid, WARPS * 32, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()), N);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
