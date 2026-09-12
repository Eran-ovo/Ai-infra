#include <torch/extension.h>

#include "ops.h"

// FlashAttention 的 head dimension 会直接决定：
//   1. QK^T 在 K 维需要执行多少次 mma.sync；
//   2. shared memory 中 Q/K/V tile 的行跨度；
//   3. PV 阶段要维护多少个输出 fragment；
//   4. 每个线程的寄存器压力。
// 因此 D=64 和 D=128 目前保留为两个编译期专用 kernel，而不是在 CUDA
// kernel 内部用 if (D == ...) 分支。这个很薄的 host 函数负责隐藏版本选择。
torch::Tensor flashattention_auto_forward(torch::Tensor q,
                                          torch::Tensor k,
                                          torch::Tensor v,
                                          bool causal) {
    // 这里只检查调度真正需要的信息。dtype、device、contiguous、Q/K/V
    // shape 一致性等完整契约仍由被选中的 v5/v6 wrapper 统一检查，避免维护
    // 两份容易逐渐不一致的验证逻辑。
    TORCH_CHECK(q.dim() == 2 || q.dim() == 4,
                "q/k/v must be [N, D] or [B, H, N, D]");

    const int64_t head_dim = q.size(-1);
    if (head_dim == 64) {
        // v7 的额外成本是 Q fragment 初始化、同步和较高寄存器压力；序列
        // 很短时，Q 只会复用少量 KV tile，v5 更适合作为稳定小问题路径。
        // 两轮 N/BH sweep 在 N>=1024 上都观察到稳定收益，因此只把这个
        // 经过验证的区间交给 v7，避免用一次 benchmark 过拟合所有 shape。
        constexpr int64_t V7_MIN_N = 1024;
        const int64_t n = (q.dim() == 4) ? q.size(2) : q.size(0);
        if (n >= V7_MIN_N) {
            return flashattention_v7_forward(q, k, v, causal);
        }
        return flashattention_v5_forward(q, k, v, causal);
    }
    if (head_dim == 128) {
        return flashattention_v6_forward(q, k, v, causal);
    }

    // 暂不静默回退到 PyTorch：扩展内部返回 PyTorch 算子会让性能归因变得
    // 模糊。显式报错能让调用方清楚当前手写 kernel 的能力边界。
    TORCH_CHECK(false,
                "flashattention_auto currently supports head_dim 64 or 128, got ",
                head_dim);
}
