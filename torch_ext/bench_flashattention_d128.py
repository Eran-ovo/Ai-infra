"""Benchmark the D=128 v6 kernel against PyTorch SDPA."""
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


def bench_pair(v6_fn, sdpa_fn, warmup, iters, rounds):
    for fn in (v6_fn, sdpa_fn):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[], []]
    for round_id in range(rounds):
        for index in ((round_id + i) % 2 for i in range(2)):
            fn = (v6_fn, sdpa_fn)[index]
            samples[index].append(measure_cuda(fn, iters))
    return median(samples[0]), median(samples[1])


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--n", type=int, nargs="+", default=[1024, 4096])
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--rounds", type=int, default=5)
args = parser.parse_args()

torch.manual_seed(42)
print("B,H,N,mode,v6_d128_ms,torch_sdpa_ms,v6_over_sdpa")
for n in args.n:
    q = torch.randn(args.batch, args.heads, n, 128, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for causal in (False, True):
        v6_fn = lambda c=causal: ai_infra_ops.flashattention_v6(q, k, v, c)
        sdpa_fn = lambda c=causal: F.scaled_dot_product_attention(q, k, v, is_causal=c)
        t_v6, t_sdpa = bench_pair(
            v6_fn, sdpa_fn, args.warmup, args.iters, args.rounds
        )
        print(
            f"{args.batch},{args.heads},{n},"
            f"{'causal' if causal else 'full'},"
            f"{t_v6:.4f},{t_sdpa:.4f},{t_v6 / t_sdpa:.2f}x"
        )
