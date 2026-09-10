"""FlashAttention v3/v4/v5 performance comparison on the current GPU.

v3 is the fp32 baseline. v4 and v5 use the same fp16 I/O and Tensor Core math;
their only intended difference is how softmax P reaches the PV MMA:
v4 uses shared memory Ps, while v5 constructs the A fragment from registers.
"""
import argparse
from statistics import median

import torch
import ai_infra_ops


torch.manual_seed(42)
assert torch.cuda.is_available(), "CUDA device is required"


def measure_cuda(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def bench_many(functions, warmup=20, iters=50, rounds=5):
    """交错、轮换顺序计时，降低 GPU boost/温度和固定执行顺序带来的偏差。"""
    for fn in functions:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[] for _ in functions]
    for round_id in range(rounds):
        order = [(round_id + i) % len(functions) for i in range(len(functions))]
        for index in order:
            samples[index].append(measure_cuda(functions[index], iters))
    return [median(values) for values in samples]


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, nargs="+", default=[1024, 4096, 8192])
parser.add_argument("--mode", choices=("all", "full", "causal"), default="all")
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--rounds", type=int, default=5)
args = parser.parse_args()
modes = (False, True) if args.mode == "all" else (args.mode == "causal",)

print("N,mode,v3_fp32_ms,v4_smem_p_ms,v5_reg_p_ms,v3_over_v5,v4_over_v5")
for n in args.n:
    q32 = torch.randn(n, 64, device="cuda", dtype=torch.float32)
    k32 = torch.randn(n, 64, device="cuda", dtype=torch.float32)
    v32 = torch.randn(n, 64, device="cuda", dtype=torch.float32)
    q16 = q32.half()
    k16 = k32.half()
    v16 = v32.half()

    for causal in modes:
        functions = [
            lambda: ai_infra_ops.flashattention(q32, k32, v32, causal),
            lambda: ai_infra_ops.flashattention_fp16(q16, k16, v16, causal),
            lambda: ai_infra_ops.flashattention_v5(q16, k16, v16, causal),
        ]
        t3, t4, t5 = bench_many(
            functions, warmup=args.warmup, iters=args.iters, rounds=args.rounds
        )
        print(
            f"{n},{'causal' if causal else 'full'},"
            f"{t3:.4f},{t4:.4f},{t5:.4f},{t3 / t5:.2f}x,{t4 / t5:.2f}x"
        )
