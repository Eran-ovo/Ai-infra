"""Controlled FlashAttention v5/v7 sequence-length sweep.

Only one implementation detail differs:
  v5: every KV tile reloads the Q MMA fragments from shared memory;
  v7: Q fragments are cached in registers and Q/K share one smem tile.

Keeping B/H/D, dtype, layout and math mode fixed lets us study how the number
of KV tiles changes the benefit of Q reuse without mixing in other variables.
"""

import argparse
from statistics import median

import torch

import ai_infra_ops


def measure_cuda(fn, iters):
    """Return average GPU time per call, measured on the current stream."""
    # CUDA launches are asynchronous to the CPU. Events are inserted into the
    # CUDA stream, so this measures device execution rather than Python time.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def compare_pair(v5_fn, v7_fn, warmup, iters, rounds):
    """Measure v5/v7 with balanced alternating order and return medians."""
    for fn in (v5_fn, v7_fn):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[], []]
    for round_id in range(rounds):
        # With an even number of rounds, each implementation appears first and
        # second equally often. This reduces fixed-order temperature/boost bias.
        order = (0, 1) if round_id % 2 == 0 else (1, 0)
        functions = (v5_fn, v7_fn)
        for index in order:
            samples[index].append(measure_cuda(functions[index], iters))
    return median(samples[0]), median(samples[1])


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument(
    "--n",
    type=int,
    nargs="+",
    default=[128, 256, 512, 1024, 2048, 4096],
)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--rounds", type=int, default=10)
args = parser.parse_args()

if args.rounds <= 0 or args.rounds % 2 != 0:
    parser.error("--rounds must be a positive even number for balanced order")

torch.manual_seed(42)
print("B,H,N,mode,kv_tiles,v5_ms,v7_ms,v7_over_v5,v7_speedup")

for n in args.n:
    q = torch.randn(
        args.batch,
        args.heads,
        n,
        64,
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for causal in (False, True):
        v5_fn = lambda c=causal: ai_infra_ops.flashattention_v5(q, k, v, c)
        v7_fn = lambda c=causal: ai_infra_ops.flashattention_v7(q, k, v, c)
        t_v5, t_v7 = compare_pair(
            v5_fn, v7_fn, args.warmup, args.iters, args.rounds
        )

        # full 的每个 Q block 都访问全部 KV tiles；causal 的不同 Q block
        # 访问数量不同。这里输出最大 tile 数，便于理解缓存复用机会的上限。
        kv_tiles = (n + 63) // 64
        print(
            f"{args.batch},{args.heads},{n},"
            f"{'causal' if causal else 'full'},{kv_tiles},"
            f"{t_v5:.4f},{t_v7:.4f},{t_v7 / t_v5:.3f}x,{t_v5 / t_v7:.3f}x"
        )
