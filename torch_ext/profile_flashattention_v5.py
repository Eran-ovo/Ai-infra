"""Launch one warmed-up FlashAttention v5/v7 kernel for Nsight Compute.

Example:
    ncu --profile-from-start off \
        --section LaunchStats --section Occupancy --section SpeedOfLight \
        --section MemoryWorkloadAnalysis --section WarpStateStats \
        python profile_flashattention_v5.py --version 7

The normal benchmark loops many times to reduce timing noise.  That is useful
for latency, but wasteful for Nsight Compute because one logical kernel may be
replayed several times to collect all hardware counters.  This script therefore
warms up before profiling and exposes only one target launch to the profiler.
"""

import argparse

import torch

import ai_infra_ops


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--n", type=int, default=1024)
parser.add_argument("--causal", action="store_true")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--version", type=int, choices=(5, 7), default=5)
args = parser.parse_args()

torch.manual_seed(42)
q = torch.randn(
    args.batch,
    args.heads,
    args.n,
    64,
    device="cuda",
    dtype=torch.float16,
)
k = torch.randn_like(q)
v = torch.randn_like(q)

# v5/v7 具有完全相同的输入和数学语义，用同一脚本可以减少 profile 配置
# 不一致造成的伪差异。
op = {
    5: ai_infra_ops.flashattention_v5,
    7: ai_infra_ops.flashattention_v7,
}[args.version]

# 预热发生在 cudaProfilerStart() 之前。这样既让 CUDA context、PyTorch
# extension 和 GPU 时钟进入稳定状态，又不会让 ncu 捕获十几个重复 kernel。
for _ in range(args.warmup):
    op(q, k, v, args.causal)
torch.cuda.synchronize()

# 配合 ncu 的 --profile-from-start off：只有 Start/Stop 之间的区域会采集
# counters。直接调用实验版本而不是 auto，保证报告中的唯一目标就是 CUDA kernel，
# 不把 host dispatcher 混入这次“内核瓶颈定位”实验。
torch.cuda.cudart().cudaProfilerStart()
out = op(q, k, v, args.causal)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

# 保持输出 tensor 活跃，并给直接运行脚本时提供一个简短完成标志。
print(
    f"profile target finished: v{args.version}, shape={tuple(q.shape)}, "
    f"causal={args.causal}, output_mean={out.float().mean().item():.6f}"
)
