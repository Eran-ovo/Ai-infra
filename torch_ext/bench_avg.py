#!/usr/bin/env python3
"""兼容入口；新的统一、同语义 benchmark 位于 bench_ops.py。"""

import sys

from bench_ops import main


if __name__ == "__main__":
    # 兼容旧用法 `python bench_avg.py 50`。
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        sys.argv[1:] = ["--iters", sys.argv[1]]
    main()
