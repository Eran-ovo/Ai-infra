"""Benchmark the stable D=64 entry against v5, per-head dispatch and SDPA.

This script deliberately keeps one fixed semantic contract: contiguous fp16 BHD
input, fp16 output, and the same full/causal mode.  Only then can the timing be
used to decide whether dispatch/launch overhead or the CUDA kernel is the next
bottleneck.
"""
import argparse
from statistics import median

import torch
import torch.nn.functional as F

import ai_infra_ops


def measure_cuda(fn, iters):
    # CUDA kernel launch 对 CPU 是异步的，time.perf_counter() 直接包住 fn()
    # 通常只会量到 launch 时间。Event 记录在同一条 CUDA stream 上，end 的
    # synchronize 保证 GPU 真正执行完，再读取设备侧经过时间。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def bench_many(functions, warmup, iters, rounds):
    # 先让 CUDA context、PyTorch dispatch 和 GPU 频率进入相对稳定的状态。
    for fn in functions:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples = [[] for _ in functions]
    for round_id in range(rounds):
        # 每轮旋转执行顺序，避免总让同一个实现处在较冷或较热的 GPU 状态。
        # 最后取中位数，降低偶发系统调度和笔记本功耗波动的影响。
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


print(
    "B,H,N,mode,auto_ms,v5_bhd_ms,per_head_dispatch_ms,torch_sdpa_ms,"
    "auto_over_v5,per_head_over_v5,v5_over_sdpa"
)
for causal in (False, True):
    # auto(D=64) 最终会调用同一个 v5 CUDA kernel。二者的差值主要衡量一层
    # host-side head_dim 判断的开销；若出现几个百分点的反转，应先考虑 GPU
    # boost/温度与测量噪声，不能误认为 auto 改变了 kernel 性能。
    functions = [
        lambda c=causal: ai_infra_ops.flashattention_auto(q, k, v, c),
        lambda c=causal: ai_infra_ops.flashattention_v5(q, k, v, c),
        lambda c=causal: per_head_dispatch(c),
        lambda c=causal: F.scaled_dot_product_attention(q, k, v, is_causal=c),
    ]
    t_auto, t_bhd, t_dispatch, t_sdpa = bench_many(
        functions, args.warmup, args.iters, args.rounds
    )
    print(
        f"{args.batch},{args.heads},{args.n},"
        f"{'causal' if causal else 'full'},"
        f"{t_auto:.4f},{t_bhd:.4f},{t_dispatch:.4f},{t_sdpa:.4f},"
        f"{t_auto / t_bhd:.2f}x,{t_dispatch / t_bhd:.2f}x,{t_bhd / t_sdpa:.2f}x"
    )
