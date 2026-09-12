#!/usr/bin/env python3
"""只测 GEMM MMA 版本链，便于反转尺寸顺序检查温度/频率漂移。"""

import argparse

import torch

import ai_infra_ops
from bench_ops import compare_variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(42)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | sizes={args.sizes}")

    for size in args.sizes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)

        # 所有实现都使用同一输入，并返回 fp32；最后一个 cuBLAS 是速度比基线。
        medians = compare_variants(
            f"GEMM MMA fp16 {size}^3 (fp32 output)",
            [
                ("v4 scalar", lambda a=a, b=b: ai_infra_ops.gemm_mma(a, b)),
                ("v5 vec", lambda a=a, b=b: ai_infra_ops.gemm_mma_vec(a, b)),
                ("v6 async", lambda a=a, b=b: ai_infra_ops.gemm_mma_async(a, b)),
                ("v7 ld", lambda a=a, b=b: ai_infra_ops.gemm_mma_ldmatrix(a, b)),
                ("v8 ld+pad", lambda a=a, b=b: ai_infra_ops.gemm_mma_ldmatrix_padded(a, b)),
                ("v9 async+ld+pad", lambda a=a, b=b: ai_infra_ops.gemm_mma_ldmatrix_async_padded(a, b)),
                ("v10 BK32", lambda a=a, b=b: ai_infra_ops.gemm_mma_v10(a, b)),
                ("cuBLAS", lambda a=a, b=b: ai_infra_ops.gemm_cublas_fp32(a, b)),
            ],
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        print(
            "  TFLOP/s     "
            + " | ".join(
                f"{label} {2 * size**3 / (ms / 1000) / 1e12:.2f}"
                for label, ms in medians.items()
            )
        )


if __name__ == "__main__":
    main()
