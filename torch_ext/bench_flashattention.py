"""FlashAttention v3/v4 performance comparison on the current GPU.

v3 uses fp32 inputs; v4 uses fp16 inputs and fp16 output with fp32 accumulation.
The benchmark intentionally avoids materializing an N x N reference matrix.
"""
import argparse
import torch
import ai_infra_ops


torch.manual_seed(42)
assert torch.cuda.is_available(), "CUDA device is required"


def bench_cuda(fn, warmup=20, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, nargs="+", default=[1024, 4096, 8192])
parser.add_argument("--mode", choices=("all", "full", "causal"), default="all")
args = parser.parse_args()
modes = (False, True) if args.mode == "all" else (args.mode == "causal",)

print("N,mode,v3_fp32_ms,v4_fp16_ms,v3_over_v4")
for n in args.n:
    q32 = torch.randn(n, 64, device="cuda", dtype=torch.float32)
    k32 = torch.randn(n, 64, device="cuda", dtype=torch.float32)
    v32 = torch.randn(n, 64, device="cuda", dtype=torch.float32)
    q16 = q32.half()
    k16 = k32.half()
    v16 = v32.half()

    for causal in modes:
        t3 = bench_cuda(lambda: ai_infra_ops.flashattention(q32, k32, v32, causal))
        t4 = bench_cuda(lambda: ai_infra_ops.flashattention_fp16(q16, k16, v16, causal))
        print(f"{n},{'causal' if causal else 'full'},{t3:.4f},{t4:.4f},{t3 / t4:.2f}x")
