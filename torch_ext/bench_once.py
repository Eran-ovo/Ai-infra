#!/usr/bin/env python
"""
gemm_mma 单次计时：每个规模只测一次（不取平均），记录手写 Tensor Core 与 cuBLAS 实际用时。
用途：拿到「一次运行」的真实数字，避免多轮平均掩盖频率波动。

用法：~/venvs/torch/bin/python bench_once.py
"""
import torch
import time
import ai_infra_ops

sizes = [1024, 2048, 4096]


def bench_once(fn, *args):
    # 同步后掐表单次
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0  # ms


def gflops(m, n, k, ms):
    return 2.0 * m * n * k / (ms / 1e3) / 1e9


print(f"{'size':>6} | {'mma_ops ms':>10} {'GFLOPS':>8} | {'cublas ms':>10} {'GFLOPS':>8} | ratio")
print("-" * 72)
for d in sizes:
    a = torch.randn(d, d, device="cuda", dtype=torch.float16)
    b = torch.randn(d, d, device="cuda", dtype=torch.float16)

    # 手写 Tensor Core
    t_mma = bench_once(ai_infra_ops.gemm_mma, a, b)
    g_mma = gflops(d, d, d, t_mma)

    # cuBLAS（torch.matmul fp16 -> fp32；先 warm-up 一次让 cuBLAS 走完 autotuning/plan 路径）
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    t_cublas = bench_once(torch.matmul, a, b)
    g_cublas = gflops(d, d, d, t_cublas)

    print(f"{d:>6} | {t_mma:>10.3f} {g_mma:>8.0f} | {t_cublas:>10.3f} {g_cublas:>8.0f} | {t_mma/t_cublas:>5.2f}x")
