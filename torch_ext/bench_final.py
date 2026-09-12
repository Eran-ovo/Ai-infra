#!/usr/bin/env python3
"""生成可提交到仓库、可用于简历数据复核的最终 benchmark 报告。

与探索阶段的脚本不同，本脚本只比较稳定入口、关键实验版本和同语义工业基线，
并同时保存 JSON（含每轮原始样本）、CSV（便于分析）和 Markdown（便于展示）。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import ai_infra_ops


TensorFn = Callable[[], torch.Tensor]


def run_optional(command: list[str], cwd: Path | None = None) -> str:
    """采集环境信息；外部工具不可用时记录 unavailable，不中断 benchmark。"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def gpu_snapshot() -> str:
    """记录测试前后的温度/频率/功耗，帮助解释笔记本 GPU 的结果漂移。"""
    return run_optional(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
    )


def event_ms(fn: TensorFn, iters: int) -> float:
    """使用当前 CUDA stream 上的 Event，只测 GPU 工作而非 Python launch。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def balanced_orders(
    implementations: list[tuple[str, TensorFn]], rounds: int
) -> list[list[tuple[str, TensorFn]]]:
    """同时平衡执行位置和前序实现，减少 GPU 动态频率造成的顺序偏差。

    简单循环移位只能平衡位置，却让每个实现永远跟随同一个前序实现。
    四实现采用 Williams design：4 个序列中每个实现各访问一次每个位置，
    且每个有向相邻组合恰好出现一次。三实现使用全部 6 种排列；二实现
    使用正反两个顺序。默认 12 轮恰好是 2、3、4 三种 case 的公共周期。
    """
    count = len(implementations)
    if count == 4:
        index_orders = [
            (0, 1, 3, 2),
            (1, 2, 0, 3),
            (2, 3, 1, 0),
            (3, 0, 2, 1),
        ]
    elif count in (2, 3):
        index_orders = list(permutations(range(count)))
    else:
        raise ValueError(f"unsupported implementation count: {count}")

    if rounds % len(index_orders) != 0:
        raise ValueError(
            f"rounds={rounds} cannot evenly repeat {len(index_orders)} balanced orders"
        )
    return [
        [implementations[index] for index in order]
        for _ in range(rounds // len(index_orders))
        for order in index_orders
    ]


@torch.inference_mode()
def benchmark_case(
    *,
    suite: str,
    case: str,
    shape: str,
    mode: str,
    implementations: list[tuple[str, TensorFn]],
    baseline: str,
    warmup: int,
    iters: int,
    rounds: int,
    flops: int | None = None,
) -> list[dict]:
    """正确性对拍后，轮换执行顺序并返回每个实现的结构化统计结果。"""
    labels = [label for label, _ in implementations]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate implementation label in {case}")
    if baseline not in labels:
        raise ValueError(f"baseline {baseline!r} is absent from {case}")

    functions = dict(implementations)

    # 先以同一输入做正确性检查。reference 保留 fp32/fp16 原始语义；只在
    # 计算 max error 时转成 fp32，避免 half 减法掩盖微小误差。
    reference = functions[baseline]()
    torch.cuda.synchronize()
    max_errors: dict[str, float] = {baseline: 0.0}
    for label, fn in implementations:
        if label == baseline:
            continue
        output = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
        max_errors[label] = (output.float() - reference.float()).abs().max().item()
        del output

    # 预热 CUDA context、allocator、PyTorch dispatch，并让频率进入相对稳定状态。
    for _, fn in implementations:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {label: [] for label in labels}
    for order in balanced_orders(implementations, rounds):
        for label, fn in order:
            samples[label].append(event_ms(fn, iters))

    baseline_ms = statistics.median(samples[baseline])
    rows = []
    for label in labels:
        values = samples[label]
        median_ms = statistics.median(values)
        min_ms = min(values)
        max_ms = max(values)
        rows.append(
            {
                "suite": suite,
                "case": case,
                "shape": shape,
                "mode": mode,
                "implementation": label,
                "baseline": baseline,
                "iters": iters,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                # 大于 1 表示当前实现比 baseline 更快。
                "baseline_over_impl": baseline_ms / median_ms,
                # range 会暴露最坏离群点；MAD 对偶发系统抖动更稳健。
                "range_pct": (max_ms - min_ms) / median_ms * 100.0,
                "mad_pct": statistics.median(
                    abs(value - median_ms) for value in values
                )
                / median_ms
                * 100.0,
                "tflops": (
                    flops / (median_ms / 1000.0) / 1e12 if flops is not None else None
                ),
                "max_abs_error": max_errors[label],
                "samples_ms": values,
            }
        )

    print(f"\n[{suite}] {case} | {shape} | {mode or '-'}")
    for row in rows:
        throughput = (
            f" | {row['tflops']:.2f} TFLOP/s" if row["tflops"] is not None else ""
        )
        print(
            f"  {row['implementation']:<10} {row['median_ms']:>9.4f} ms"
            f" | {baseline}/{row['implementation']} "
            f"{row['baseline_over_impl']:.3f}x"
            f" | iters {iters} | MAD {row['mad_pct']:.1f}%"
            f" | range {row['range_pct']:.1f}%{throughput}"
        )

    del reference
    return rows


def find_row(rows: list[dict], suite: str, case: str, mode: str, impl: str) -> dict:
    for row in rows:
        if (
            row["suite"] == suite
            and row["case"] == case
            and row["mode"] == mode
            and row["implementation"] == impl
        ):
            return row
    raise KeyError((suite, case, mode, impl))


def markdown_report(metadata: dict, rows: list[dict]) -> str:
    """生成既包含完整长表，也包含简历核心指标摘要的 Markdown。"""
    lines = [
        "# AI Infra Operators — Final Benchmark",
        "",
        "## Environment",
        "",
        f"- Timestamp: `{metadata['timestamp']}`",
        f"- GPU: `{metadata['gpu_name']}` (`sm_{metadata['compute_capability'].replace('.', '')}`)",
        f"- GPU memory: `{metadata['gpu_memory_gib']:.2f} GiB`",
        f"- PyTorch / CUDA runtime: `{metadata['torch_version']}` / `{metadata['torch_cuda']}`",
        f"- NVCC: `{metadata['nvcc']}`",
        f"- Git commit: `{metadata['git_commit']}`; dirty: `{metadata['git_dirty']}`",
        f"- GPU snapshot (start): `{metadata['gpu_snapshot_start']}`",
        f"- GPU snapshot (end): `{metadata['gpu_snapshot_end']}`",
        "",
        "## Method",
        "",
        f"CUDA Event timing; warmup={metadata['warmup']}, base iters={metadata['iters']}, "
        f"rounds={metadata['rounds']}; statistic=median. Large cubic cases reduce "
        "iterations proportionally; the exact count is recorded per row. Implementations use the same "
        "inputs and use position/predecessor-balanced orders. `baseline/impl > 1` means "
        "the implementation is faster than its baseline.",
        "",
        "## Resume-facing summary",
        "",
        "### Reduction operators",
        "",
        "| Operator | Shape | Custom ms | PyTorch ms | Speedup |",
        "|---|---|---:|---:|---:|",
    ]

    norm_cases = sorted({row["case"] for row in rows if row["suite"] == "norm"})
    for case in norm_cases:
        custom = find_row(rows, "norm", case, "", "custom")
        pytorch = find_row(rows, "norm", case, "", "pytorch")
        lines.append(
            f"| {case} | {custom['shape']} | {custom['median_ms']:.4f} | "
            f"{pytorch['median_ms']:.4f} | "
            f"{pytorch['median_ms'] / custom['median_ms']:.2f}x |"
        )

    lines.extend(
        [
        "",
        "### GEMM cubic",
        "",
        "| Shape | v4 ms | auto ms | cuBLAS ms | auto TFLOP/s | v4/auto | auto/cuBLAS perf |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    cubic_cases = sorted(
        {row["case"] for row in rows if row["suite"] == "gemm_cubic"},
        key=lambda value: int(value),
    )
    for case in cubic_cases:
        v4 = find_row(rows, "gemm_cubic", case, "", "v4")
        auto = find_row(rows, "gemm_cubic", case, "", "auto")
        cublas = find_row(rows, "gemm_cubic", case, "", "cublas")
        lines.append(
            f"| {auto['shape']} | {v4['median_ms']:.4f} | {auto['median_ms']:.4f} | "
            f"{cublas['median_ms']:.4f} | {auto['tflops']:.2f} | "
            f"{v4['median_ms'] / auto['median_ms']:.2f}x | "
            f"{cublas['median_ms'] / auto['median_ms']:.2%} |"
        )

    lines.extend(
        [
            "",
            "### GEMM Transformer shapes",
            "",
            "| Shape | Route | auto ms | cuBLAS ms | auto TFLOP/s | auto/cuBLAS perf |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    transformer_cases = sorted(
        {row["case"] for row in rows if row["suite"] == "gemm_transformer"},
        key=lambda value: int(value.split("_")[0][1:]),
    )
    for case in transformer_cases:
        auto = find_row(rows, "gemm_transformer", case, "", "auto")
        cublas = find_row(rows, "gemm_transformer", case, "", "cublas")
        m = int(case.split("_")[0][1:])
        route = "v10" if m <= 256 else "v8"
        lines.append(
            f"| {auto['shape']} | {route} | {auto['median_ms']:.4f} | "
            f"{cublas['median_ms']:.4f} | {auto['tflops']:.2f} | "
            f"{cublas['median_ms'] / auto['median_ms']:.2%} |"
        )

    lines.extend(
        [
            "",
            "### FlashAttention D=64",
            "",
            "| Shape | Mode | Route | Comparison | Comparison ms | auto ms | SDPA ms | comparison/auto | auto/SDPA perf |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    flash_keys = sorted(
        {
            (row["case"], row["mode"])
            for row in rows
            if row["suite"] == "flash_d64"
        },
        key=lambda item: (int(item[0]), item[1]),
    )
    for case, mode in flash_keys:
        auto = find_row(rows, "flash_d64", case, mode, "auto")
        sdpa = find_row(rows, "flash_d64", case, mode, "sdpa")
        comparison_label = "v7" if int(case) < 1024 else "v5"
        route = "v5" if int(case) < 1024 else "v7"
        comparison = find_row(rows, "flash_d64", case, mode, comparison_label)
        lines.append(
            f"| {auto['shape']} | {mode} | {route} | {comparison_label} | "
            f"{comparison['median_ms']:.4f} | {auto['median_ms']:.4f} | "
            f"{sdpa['median_ms']:.4f} | "
            f"{comparison['median_ms'] / auto['median_ms']:.2f}x | "
            f"{sdpa['median_ms'] / auto['median_ms']:.2%} |"
        )

    lines.extend(
        [
            "",
            "### FlashAttention D=128",
            "",
            "| Shape | Mode | Route | auto ms | SDPA ms | auto/SDPA perf |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    d128_keys = sorted(
        {
            (row["case"], row["mode"])
            for row in rows
            if row["suite"] == "flash_d128"
        },
        key=lambda item: (int(item[0]), item[1]),
    )
    for case, mode in d128_keys:
        auto = find_row(rows, "flash_d128", case, mode, "auto")
        sdpa = find_row(rows, "flash_d128", case, mode, "sdpa")
        lines.append(
            f"| {auto['shape']} | {mode} | v6 | {auto['median_ms']:.4f} | "
            f"{sdpa['median_ms']:.4f} | "
            f"{sdpa['median_ms'] / auto['median_ms']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Complete results",
            "",
            "| Suite | Case | Mode | Shape | Implementation | Iters | Median ms | Min ms | Max ms | MAD | Range | Baseline/impl | TFLOP/s | Max abs error |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        tflops = "-" if row["tflops"] is None else f"{row['tflops']:.2f}"
        lines.append(
            f"| {row['suite']} | {row['case']} | {row['mode'] or '-'} | "
            f"{row['shape']} | {row['implementation']} | {row['iters']} | "
            f"{row['median_ms']:.4f} | "
            f"{row['min_ms']:.4f} | {row['max_ms']:.4f} | "
            f"{row['mad_pct']:.1f}% | {row['range_pct']:.1f}% | "
            f"{row['baseline_over_impl']:.3f}x | "
            f"{tflops} | {row['max_abs_error']:.3e} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(output_dir: Path, run_name: str, metadata: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    csv_path = output_dir / f"{run_name}.csv"
    markdown_path = output_dir / f"{run_name}.md"

    json_path.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "suite",
        "case",
        "shape",
        "mode",
        "implementation",
        "baseline",
        "iters",
        "median_ms",
        "min_ms",
        "max_ms",
        "mad_pct",
        "range_pct",
        "baseline_over_impl",
        "tflops",
        "max_abs_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        # Git 仓库统一使用 LF；csv 模块默认 dialect 会写 CRLF。
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})

    markdown_path.write_text(markdown_report(metadata, rows), encoding="utf-8")
    print(f"\nJSON     : {json_path}")
    print(f"CSV      : {csv_path}")
    print(f"Markdown : {markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="只跑每类一个小 shape，用于检查脚本，不作为最终性能数据",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results",
    )
    parser.add_argument("--name", help="输出文件名前缀；默认使用时间戳")
    args = parser.parse_args()

    if args.warmup < 0 or args.iters <= 0 or args.rounds <= 0:
        parser.error("warmup must be >= 0; iters and rounds must be > 0")
    if args.rounds % 12 != 0:
        parser.error("rounds must be a multiple of 12 to balance 2/3/4 implementations")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    run_name = args.name or datetime.now().strftime(
        "smoke_%Y%m%d_%H%M%S" if args.quick else "final_benchmark_%Y%m%d_%H%M%S"
    )

    metadata = {
        "timestamp": timestamp,
        "gpu_name": props.name,
        "compute_capability": f"{major}.{minor}",
        "gpu_memory_gib": props.total_memory / 1024**3,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "nvcc": run_optional(["nvcc", "--version"]).splitlines()[-1],
        "git_commit": run_optional(["git", "rev-parse", "HEAD"], repo_root),
        "git_dirty": bool(run_optional(["git", "status", "--porcelain"], repo_root)),
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
        "quick": args.quick,
        "seed": 42,
        "order_strategy": "Williams/all-permutations position-and-predecessor balance",
        "gpu_snapshot_start": gpu_snapshot(),
    }

    print(f"GPU: {props.name} | sm_{major}{minor} | {props.total_memory / 1024**3:.2f} GiB")
    print(
        f"Method: CUDA Event | warmup={args.warmup} | iters={args.iters} "
        f"| rounds={args.rounds} | median"
    )
    if args.quick:
        print("QUICK MODE: 结果只用于脚本验证，不能写入简历。")

    torch.manual_seed(42)
    rows: list[dict] = []

    # 带宽/归约类算子也进入完整报告，证明库中的五类算子都有统一基线。
    norm_shapes = [(1024, 1024)]
    eps = 1e-5
    for batch, hidden in norm_shapes:
        x = torch.randn(batch, hidden, device="cuda", dtype=torch.float32)
        weight = torch.rand(hidden, device="cuda", dtype=torch.float32) + 0.5
        shape = f"[{batch},{hidden}]"
        rows += benchmark_case(
            suite="norm",
            case="rmsnorm_fp32",
            shape=shape,
            mode="",
            implementations=[
                ("custom", lambda x=x, w=weight: ai_infra_ops.rmsnorm(x, w, eps)),
                ("pytorch", lambda x=x, w=weight: F.rms_norm(x, (hidden,), w, eps)),
            ],
            baseline="pytorch",
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        rows += benchmark_case(
            suite="norm",
            case="softmax_fp32",
            shape=shape,
            mode="",
            implementations=[
                ("custom", lambda x=x: ai_infra_ops.softmax(x)),
                ("pytorch", lambda x=x: torch.softmax(x, dim=-1)),
            ],
            baseline="pytorch",
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        rows += benchmark_case(
            suite="norm",
            case="layernorm_fp32",
            shape=shape,
            mode="",
            implementations=[
                ("custom", lambda x=x: ai_infra_ops.layernorm(x, eps)),
                ("pytorch", lambda x=x: F.layer_norm(x, (hidden,), None, None, eps)),
            ],
            baseline="pytorch",
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        del x, weight

    cubic_sizes = [1024] if args.quick else [1024, 2048, 4096]
    for size in cubic_sizes:
        a = torch.randn(size, size, device="cuda", dtype=torch.float16)
        b = torch.randn(size, size, device="cuda", dtype=torch.float16)
        # 4096^3 的单次 kernel 已足够长；按 size 缩减迭代数，避免 v4 对照
        # 独占绝大部分运行时间，同时让每个 Event 样本仍包含多次 kernel。
        case_iters = max(5, args.iters * 1024 // size)
        rows += benchmark_case(
            suite="gemm_cubic",
            case=str(size),
            shape=f"[{size},{size}] @ [{size},{size}]",
            mode="",
            implementations=[
                ("auto", lambda a=a, b=b: ai_infra_ops.gemm_mma_auto(a, b)),
                ("v4", lambda a=a, b=b: ai_infra_ops.gemm_mma(a, b)),
                ("cublas", lambda a=a, b=b: ai_infra_ops.gemm_cublas_fp32(a, b)),
            ],
            baseline="cublas",
            warmup=args.warmup,
            iters=case_iters,
            rounds=args.rounds,
            flops=2 * size**3,
        )
        del a, b
        torch.cuda.empty_cache()

    transformer_cases = (
        [(128, 4096, 4096)]
        if args.quick
        else [
            (128, 4096, 4096),
            (256, 4096, 4096),
            (512, 4096, 4096),
            (1024, 4096, 4096),
        ]
    )
    for m, k, n in transformer_cases:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        rows += benchmark_case(
            suite="gemm_transformer",
            case=f"m{m}_k{k}_n{n}",
            shape=f"[{m},{k}] @ [{k},{n}]",
            mode="",
            implementations=[
                ("auto", lambda a=a, b=b: ai_infra_ops.gemm_mma_auto(a, b)),
                ("cublas", lambda a=a, b=b: ai_infra_ops.gemm_cublas_fp32(a, b)),
            ],
            baseline="cublas",
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
            flops=2 * m * n * k,
        )
        del a, b
        torch.cuda.empty_cache()

    d64_lengths = [1024] if args.quick else [512, 1024, 2048]
    for n in d64_lengths:
        batch, heads, dim = 2, 8, 64
        q = torch.randn(batch, heads, n, dim, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for causal in (False, True):
            mode = "causal" if causal else "full"
            # auto 在短序列路由 v5、长序列路由 v7。最终报告只加入另一个
            # 版本作为对照，避免把 auto 与它调用的同一 kernel 重复计时。
            comparison = (
                (
                    "v7",
                    lambda c=causal, q=q, k=k, v=v: ai_infra_ops.flashattention_v7(
                        q, k, v, c
                    ),
                )
                if n < 1024
                else (
                    "v5",
                    lambda c=causal, q=q, k=k, v=v: ai_infra_ops.flashattention_v5(
                        q, k, v, c
                    ),
                )
            )
            rows += benchmark_case(
                suite="flash_d64",
                case=str(n),
                shape=f"[{batch},{heads},{n},{dim}]",
                mode=mode,
                implementations=[
                    ("auto", lambda c=causal, q=q, k=k, v=v: ai_infra_ops.flashattention_auto(q, k, v, c)),
                    comparison,
                    ("sdpa", lambda c=causal, q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=c)),
                ],
                baseline="sdpa",
                warmup=args.warmup,
                iters=args.iters,
                rounds=args.rounds,
            )
        del q, k, v
        torch.cuda.empty_cache()

    d128_lengths = [1024] if args.quick else [1024, 4096]
    for n in d128_lengths:
        batch, heads, dim = 2, 2, 128
        q = torch.randn(batch, heads, n, dim, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for causal in (False, True):
            mode = "causal" if causal else "full"
            rows += benchmark_case(
                suite="flash_d128",
                case=str(n),
                shape=f"[{batch},{heads},{n},{dim}]",
                mode=mode,
                implementations=[
                    ("auto", lambda c=causal, q=q, k=k, v=v: ai_infra_ops.flashattention_auto(q, k, v, c)),
                    ("sdpa", lambda c=causal, q=q, k=k, v=v: F.scaled_dot_product_attention(q, k, v, is_causal=c)),
                ],
                baseline="sdpa",
                warmup=args.warmup,
                iters=args.iters,
                rounds=args.rounds,
            )
        del q, k, v
        torch.cuda.empty_cache()

    metadata["gpu_snapshot_end"] = gpu_snapshot()
    write_reports(args.output_dir, run_name, metadata, rows)


if __name__ == "__main__":
    main()
