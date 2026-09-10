"""Benchmark the v5 [B,H,N,D] grid against per-head dispatch and PyTorch SDPA."""
import argparse
from statistics import median

import torch
import torch.nn.functional as F

import ai_infra_ops


def measure_cuda(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def bench_many(functions, warmup, iters, rounds):
    for fn in functions:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[] for _ in functions]
    for round_id in range(rounds):
        for index in ((round_id + i) % len(functions) for i in range(len(functions))):
            samples[index].append(measure_cuda(functions[index], iters))
    return [median(values) for values in samples]


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--n", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--rounds", type=int, default=5)
args = parser.parse_args()

torch.manual_seed(42)
device = "cuda"
q = torch.randn(args.batch, args.heads, args.n, 64, device=device, dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)


def per_head_dispatch(causal):
    # 对照“Python for 循环逐 head 调单头 API”的 launch 开销。
    result = None
    for batch in range(args.batch):
        for head in range(args.heads):
            result = ai_infra_ops.flashattention_v5(
                q[batch, head], k[batch, head], v[batch, head], causal
            )
    return result


print("B,H,N,mode,v5_bhd_ms,per_head_dispatch_ms,torch_sdpa_ms,dispatch_over_bhd")
for causal in (False, True):
    functions = [
        lambda c=causal: ai_infra_ops.flashattention_v5(q, k, v, c),
        lambda c=causal: per_head_dispatch(c),
        lambda c=causal: F.scaled_dot_product_attention(q, k, v, is_causal=c),
    ]
    t_bhd, t_dispatch, t_sdpa = bench_many(
        functions, args.warmup, args.iters, args.rounds
    )
    print(
        f"{args.batch},{args.heads},{args.n},"
        f"{'causal' if causal else 'full'},"
        f"{t_bhd:.4f},{t_dispatch:.4f},{t_sdpa:.4f},{t_dispatch / t_bhd:.2f}x"
    )
