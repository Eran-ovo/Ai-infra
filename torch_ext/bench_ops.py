#!/usr/bin/env python3
"""可复现的 CUDA 算子 benchmark：CUDA Event + 预热 + 交错多轮中位数。"""

import argparse
import statistics

import torch
import torch.nn.functional as F

import ai_infra_ops


def event_ms(fn, iters):
    """只测当前 CUDA stream 上的 GPU 工作，不把 Python 调度时间算进去。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def compare(name, custom, baseline, *, warmup, iters, rounds, detail=""):
    custom_out = custom()
    baseline_out = baseline()
    torch.cuda.synchronize()
    torch.testing.assert_close(custom_out, baseline_out, rtol=2e-2, atol=2e-2)

    for _ in range(warmup):
        custom()
        baseline()
    torch.cuda.synchronize()

    samples = {"custom": [], "baseline": []}
    funcs = [("custom", custom), ("baseline", baseline)]
    for round_id in range(rounds):
        # 每轮反转顺序，减少温度、频率和后台负载随时间漂移造成的偏差。
        order = funcs if round_id % 2 == 0 else list(reversed(funcs))
        for key, fn in order:
            samples[key].append(event_ms(fn, iters))

    custom_ms = statistics.median(samples["custom"])
    baseline_ms = statistics.median(samples["baseline"])
    speedup = baseline_ms / custom_ms
    suffix = f" | {detail}" if detail else ""
    print(
        f"{name:<29} custom {custom_ms:>8.4f} ms | "
        f"baseline {baseline_ms:>8.4f} ms | {speedup:>6.2f}x{suffix}"
    )
    return custom_ms, baseline_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(42)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(
        f"method: CUDA Event, warmup={args.warmup}, iters={args.iters}, "
        f"rounds={args.rounds}, statistic=median"
    )

    eps = 1e-5
    x = torch.randn(1024, 1024, device=device)
    weight = torch.rand(1024, device=device) + 0.5
    compare(
        "RMSNorm fp32 [1024,1024]",
        lambda: ai_infra_ops.rmsnorm(x, weight, eps),
        lambda: F.rms_norm(x, (x.shape[-1],), weight, eps),
        warmup=args.warmup,
        iters=args.iters,
        rounds=args.rounds,
    )
    compare(
        "Softmax fp32 [1024,1024]",
        lambda: ai_infra_ops.softmax(x),
        lambda: torch.softmax(x, dim=-1),
        warmup=args.warmup,
        iters=args.iters,
        rounds=args.rounds,
    )
    compare(
        "LayerNorm fp32 [1024,1024]",
        lambda: ai_infra_ops.layernorm(x, eps),
        lambda: F.layer_norm(x, (x.shape[-1],), None, None, eps),
        warmup=args.warmup,
        iters=args.iters,
        rounds=args.rounds,
    )

    # 严格 fp32 GEMM：显式禁用 TF32，使乘法精度语义与手写 kernel 一致。
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a32 = torch.randn(1024, 1024, device=device)
        b32 = torch.randn(1024, 1024, device=device)
        custom_ms, baseline_ms = compare(
            "GEMM fp32 1024^3",
            lambda: ai_infra_ops.gemm(a32, b32),
            lambda: torch.matmul(a32, b32),
            warmup=args.warmup,
            iters=max(10, args.iters // 5),
            rounds=args.rounds,
            detail="TF32=off",
        )
        custom_tflops = 2 * 1024**3 / (custom_ms / 1000) / 1e12
        baseline_tflops = 2 * 1024**3 / (baseline_ms / 1000) / 1e12
        print(f"{'':29} custom {custom_tflops:.2f} TFLOP/s | baseline {baseline_tflops:.2f} TFLOP/s")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    for size in (1024, 2048, 4096):
        a16 = torch.randn(size, size, device=device, dtype=torch.float16)
        b16 = torch.randn(size, size, device=device, dtype=torch.float16)
        custom_ms, baseline_ms = compare(
            f"GEMM MMA fp16 {size}^3",
            lambda a=a16, b=b16: ai_infra_ops.gemm_mma(a, b),
            lambda a=a16, b=b16: ai_infra_ops.gemm_cublas_fp32(a, b),
            warmup=max(5, args.warmup // 2),
            iters=max(5, args.iters // 10),
            rounds=args.rounds,
            detail="fp32 output",
        )
        custom_tflops = 2 * size**3 / (custom_ms / 1000) / 1e12
        baseline_tflops = 2 * size**3 / (baseline_ms / 1000) / 1e12
        print(f"{'':29} custom {custom_tflops:.2f} TFLOP/s | baseline {baseline_tflops:.2f} TFLOP/s")

    for dim, op in ((64, ai_infra_ops.flashattention_v5), (128, ai_infra_ops.flashattention_v6)):
        q = torch.randn(1, 8, 1024, dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        compare(
            f"FlashAttention D={dim}",
            lambda op=op, q=q, k=k, v=v: op(q, k, v, False),
            lambda q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v),
            warmup=max(5, args.warmup // 2),
            iters=max(5, args.iters // 10),
            rounds=args.rounds,
            detail="B=1,H=8,N=1024,fp16",
        )


if __name__ == "__main__":
    main()
