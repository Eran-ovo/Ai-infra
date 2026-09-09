// PyTorch 包装：复用 kernels/flashattention_v4.cu 中已经验证过的 Tensor Core kernel。
// v3 保留在 flashattention_cuda.cu，作为 fp32 baseline；本文件新增 fp16 v4 API。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

#define FLASHATTENTION_V4_NO_MAIN
#include "../../kernels/flashattention_v4.cu"
#undef FLASHATTENTION_V4_NO_MAIN

// PyTorch 侧入口：q/k/v [N, 64] fp16 contiguous -> o [N, 64] fp16。
// full/causal 由 Python bool 在包装层派发到编译期模板实例。
torch::Tensor flashattention_fp16_forward(torch::Tensor q,
                                          torch::Tensor k,
                                          torch::Tensor v,
                                          bool causal) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q/k/v must be CUDA tensors");
    TORCH_CHECK(q.dtype() == torch::kHalf &&
                    k.dtype() == torch::kHalf &&
                    v.dtype() == torch::kHalf,
                "q/k/v must be float16");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2,
                "q/k/v must be [N, D]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q/k/v must be on the same CUDA device");

    c10::cuda::CUDAGuard device_guard(q.device());

    const int64_t N64 = q.size(0);
    TORCH_CHECK(N64 > 0 && N64 <= INT_MAX, "N must be in [1, INT_MAX]");
    TORCH_CHECK(q.size(1) == D, "feature dim must be 64");
    TORCH_CHECK(k.size(0) == N64 && v.size(0) == N64 &&
                    k.size(1) == D && v.size(1) == D,
                "q/k/v must have shape [N, 64]");

    const int N = static_cast<int>(N64);
    auto o = torch::empty_like(q);
    const int grid = (N + Bq - 1) / Bq;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (causal) {
        flash_fp16_mma<true><<<grid, WARPS * 32, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()), N);
    } else {
        flash_fp16_mma<false><<<grid, WARPS * 32, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(o.data_ptr<at::Half>()), N);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}
