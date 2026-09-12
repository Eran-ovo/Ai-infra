#!/usr/bin/env python3
"""面向 Transformer 线性层形状的 GEMM dispatch benchmark。

与只测方阵的 bench_gemm_mma.py 不同，这里让 M 表示 token 数，K/N 表示
hidden/output dimension，用于验证 production API 的 shape policy。
"""

import argparse

import torch

import ai_infra_ops
from bench_ops import compare_variants


DEFAULT_CASES = (
    ("small-token projection", 128, 4096, 4096),
    ("medium-token projection", 512, 4096, 4096),
    ("prefill projection", 1024, 4096, 4096),
    ("square training tile", 2048, 2048, 2048),
)


def expected_policy(m, k, n):
    """镜像 C++ 中可读的离线策略，只用于打印，不参与实际 dispatch。"""
    if 0 < m <= 256 and m % 64 == 0 and n % 64 == 0 and k % 32 == 0:
        return "v10"
    return "v8/fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--large",
        action="store_true",
        help="额外测试 4096^3；RTX 3060 Laptop 上耗时更长",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    cases = list(DEFAULT_CASES)
    if args.large:
        cases.append(("large square", 4096, 4096, 4096))

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")

    for name, m, k, n in cases:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)

        medians = compare_variants(
            f"{name}: [{m},{k}] @ [{k},{n}] | policy={expected_policy(m,k,n)}",
            [
                ("auto", lambda a=a, b=b: ai_infra_ops.gemm_mma_auto(a, b)),
                ("v8", lambda a=a, b=b: ai_infra_ops.gemm_mma_ldmatrix_padded(a, b)),
                ("v10", lambda a=a, b=b: ai_infra_ops.gemm_mma_v10(a, b)),
                ("cuBLAS", lambda a=a, b=b: ai_infra_ops.gemm_cublas_fp32(a, b)),
            ],
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )

        flops = 2 * m * n * k
        print(
            "  TFLOP/s     "
            + " | ".join(
                f"{label} {flops / (ms / 1000) / 1e12:.2f}"
                for label, ms in medians.items()
            )
        )


if __name__ == "__main__":
    main()
