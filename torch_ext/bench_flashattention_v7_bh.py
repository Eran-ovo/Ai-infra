"""Measure how B*H parallelism affects the v5/v7 FlashAttention trade-off.

N and D stay fixed.  Only the number of independent sequences/heads changes,
which changes grid.y and the total number of Q blocks available to the SMs.
"""

import argparse
from statistics import median

import torch

import ai_infra_ops


def measure_cuda(fn, iters):
    # CUDA launch 对 CPU 是异步的，所以使用同一 stream 上的 Event 计时。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def compare_pair(v5_fn, v7_fn, warmup, iters, rounds):
    for fn in (v5_fn, v7_fn):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[], []]
    for round_id in range(rounds):
        # 偶数轮保证 v5/v7 都等次数先执行和后执行，降低频率/温度偏差。
        order = (0, 1) if round_id % 2 == 0 else (1, 0)
        functions = (v5_fn, v7_fn)
        for index in order:
            samples[index].append(measure_cuda(functions[index], iters))
    return median(samples[0]), median(samples[1])


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--rounds", type=int, default=10)
args = parser.parse_args()

if args.rounds <= 0 or args.rounds % 2 != 0:
    parser.error("--rounds must be a positive even number")

# 这些 case 从单头逐步增加到足以填满 GPU 的 B*H，N/D 保持不变。
cases = ((1, 1), (1, 4), (2, 8), (4, 16))
torch.manual_seed(42)
print("B,H,N,mode,total_q_blocks,v5_ms,v7_ms,v7_over_v5,v7_speedup")

for batch, heads in cases:
    q = torch.randn(batch, heads, args.n, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    total_q_blocks = batch * heads * ((args.n + 63) // 64)

    for causal in (False, True):
        v5_fn = lambda c=causal: ai_infra_ops.flashattention_v5(q, k, v, c)
        v7_fn = lambda c=causal: ai_infra_ops.flashattention_v7(q, k, v, c)
        t_v5, t_v7 = compare_pair(
            v5_fn, v7_fn, args.warmup, args.iters, args.rounds
        )
        print(
            f"{batch},{heads},{args.n},"
            f"{'causal' if causal else 'full'},{total_q_blocks},"
            f"{t_v5:.4f},{t_v7:.4f},{t_v7 / t_v5:.3f}x,{t_v5 / t_v7:.3f}x"
        )
