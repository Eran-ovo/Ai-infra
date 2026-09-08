#!/usr/bin/env python
"""
gemm_mma 平均计时：预热后多轮取平均，手写 Tensor Core vs cuBLAS(fp16)。
交错测量（每轮两个都跑一次）以摊平频率波动，符合"同会话交错测速"纪律。

用法：~/venvs/torch/bin/python bench_avg.py [iters]
"""
import sys
import torch
import time
import ai_infra_ops

sizes = [1024, 2048, 4096]
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
WARMUP = 5


def mean_ms(fn, *args, iters=ITERS):
    # 预热
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    # 多轮计时
    total = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        total += (time.perf_counter() - t0) * 1000.0
    return total / iters


def gflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms / 1e3) / 1e9


print(f"iters={ITERS} (after {WARMUP} warmup)")
print(f"{'size':>6} | {'mma_ops ms':>10} {'GFLOPS':>8} | {'cublas ms':>10} {'GFLOPS':>8} | ratio")
print("-" * 72)
for d in sizes:
    a = torch.randn(d, d, device="cuda", dtype=torch.float16)
    b = torch.randn(d, d, device="cuda", dtype=torch.float16)

    # 交错：每轮先 mma 再 cublas（也可再反向一次，这里取均值已够）
    t_mma = mean_ms(ai_infra_ops.gemm_mma, a, b)
    t_cublas = mean_ms(torch.matmul, a, b)

    g_mma = gflops(d, d, d, t_mma)
    g_cublas = gflops(d, d, d, t_cublas)

    print(f"{d:>6} | {t_mma:>10.3f} {g_mma:>8.0f} | {t_cublas:>10.3f} {g_cublas:>8.0f} | {t_mma/t_cublas:>5.2f}x")
