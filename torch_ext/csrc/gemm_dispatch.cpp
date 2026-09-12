// GEMM 生产入口：把实验版本收敛成一个稳定、可解释的 shape-aware dispatch。
//
// 路由层次：
//   1. 小 M、完整 BK=32 tile：v10（减少 K-loop 的 wait/barrier 次数）；
//   2. 其他完整 64x64x16 tile：v8（较小 shared memory，跨尺寸更稳定）；
//   3. tail / 非对齐 storage：v8 wrapper 内部继续回退 v4 通用 kernel。
//
// 为什么不在第一次调用时现场 benchmark？
// CUDA Event autotune 会强制 stream 同步、增加冷启动延迟，还需要缓存 shape
// 和设备信息。当前学习项目先使用离线 benchmark 得出的保守规则；未来可以把
// policy 表生成为静态配置，或在 Python 层提供显式 autotune/cache。

#include <torch/extension.h>
#include "ops.h"

#include <cstdint>

namespace {

constexpr int64_t BM = 64;
constexpr int64_t BN = 64;
constexpr int64_t V10_BK = 32;
constexpr int64_t V10_MAX_M = 256;

bool can_inspect_as_fp16_gemm(const torch::Tensor& a, const torch::Tensor& b) {
    return a.is_cuda() && b.is_cuda() &&
           a.dtype() == torch::kHalf && b.dtype() == torch::kHalf &&
           a.dim() == 2 && b.dim() == 2 &&
           a.is_contiguous() && b.is_contiguous() &&
           a.device() == b.device() && b.size(0) == a.size(1);
}

bool pointers_are_16b_aligned(const torch::Tensor& a, const torch::Tensor& b) {
    return reinterpret_cast<std::uintptr_t>(a.data_ptr<at::Half>()) % 16 == 0 &&
           reinterpret_cast<std::uintptr_t>(b.data_ptr<at::Half>()) % 16 == 0;
}

bool use_v10_for_shape(const torch::Tensor& a, const torch::Tensor& b) {
    if (!can_inspect_as_fp16_gemm(a, b))
        return false;

    const int64_t M = a.size(0);
    const int64_t K = a.size(1);
    const int64_t N = b.size(1);

    // 这是有意收窄的经验规则，不是硬件定律：
    // - M<=256 的 Transformer token-batch 形状中，BK=32 多轮中位数通常不差；
    // - M>=512 的结果受频率和 shape 影响较大，默认回到更稳的 v8；
    // - 完整 tile/真实指针对齐保证 v10 不会立即再次走 fallback。
    return M > 0 && N > 0 && K > 0 && M <= V10_MAX_M &&
           M % BM == 0 && N % BN == 0 && K % V10_BK == 0 &&
           pointers_are_16b_aligned(a, b);
}

}  // namespace

torch::Tensor gemm_mma_auto_forward(torch::Tensor a, torch::Tensor b) {
    if (use_v10_for_shape(a, b))
        return gemm_mma_v10_forward(a, b);

    // v8 自己负责完整的 TORCH_CHECK，并在 M/N/K tail、K=0 或 storage
    // 未对齐时回退 v4。dispatch 层不复制这些边界逻辑，避免两处规则漂移。
    return gemm_mma_ldmatrix_padded_forward(a, b);
}
