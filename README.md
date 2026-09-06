# cuda-kernels - 高性能算子库

学习 AI Infra / CUDA Kernel 优化的练习仓库。每个算子独立 `main()`，编译即可运行，自带 CPU 参考实现对拍验证。

Benchmark on RTX 3060 Laptop 6GB

## 环境

- GPU: NVIDIA GeForce RTX 3060 Laptop (6GB, Ampere / sm_86)
- CUDA Toolkit: 12.4
- OS: WSL2 (Linux)
- 编译: `nvcc -O3`

## 目录结构

```
kernels/        # 算子源码（每个 .cu 独立可编译运行）
benchmarks/     # 编译产物（gitignore）
ncu_flash.txt   # FlashAttention 的 Nsight Compute profiling 记录
```

## 构建与运行

```bash
# 普通算子
nvcc -O3 kernels/gemm_v1.cu -o benchmarks/gemm_v1 && ./benchmarks/gemm_v1

# CUTLASS 版（需额外指定头文件路径）
nvcc -O3 -I/home/eran/cutlass/include kernels/gemm_v3_cutlass.cu -o benchmarks/gemm_v3 && ./benchmarks/gemm_v3
```

## Benchmark 结果（2026-09-05 实测）

| 算子 | 版本 | 规模 | 耗时 | 性能 | 验证 |
|------|------|------|------|------|------|
| GEMM | v0 Naive（一线程一元素） | 1024³ | 3.53 ms | 607 GFLOPS | PASS |
| GEMM | v1 Shared Memory Tiling（32x32，无 Bank Conflict） | 1024³ | 2.77 ms | 779 GFLOPS | PASS |
| GEMM | v2 Coalesced + `__ldg` 只读缓存 | 1024³ | 2.72 ms | 789 GFLOPS | PASS |
| GEMM | v3 CUTLASS 工业级实现（编译期固化 tile/warp/流水线） | 1024³ | 0.408 ms | 5259 GFLOPS | PASS |
| Softmax | v1 Fused（线程粗化 + 树形归约 + warp shuffle） | 1024 x 1024 | 0.046 ms | - | PASS (maxErr=1.7e-08) |
| LayerNorm | v1 Fused（两次归约求均值/方差） | 1024 x 1024 | 0.030 ms | - | PASS (maxErr=4.5e-06) |
| RMSNorm | v1 Fused（LLaMA 标配，一次归约） | 1024 x 1024 | 0.031 ms | - | PASS (maxErr=1.7e-06) |
| FlashAttention | v1（分块 + Online Softmax，S 矩阵不落地 HBM） | N=512, D=64, Br/Bc=32 | 0.199 ms | - | PASS (maxErr=2.4e-07) |
| FlashAttention | v2（v1 + causal mask，模板双模式） | N=8192, D=64, Br=32 | full 9.43 / causal 4.71 ms | 2.00x | PASS (maxErr=6.4e-07) |

> 注：WSL2 下 GPU 频率有波动，数据为多轮运行的代表值；GEMM 计时为 20 次平均，Softmax/LayerNorm/RMSNorm/FlashAttention 为 100 次平均（均含预热）。

## 调优实战记录（ncu 性能分析）

**FlashAttention v2 causal mask 的"负优化"之谜**（2026-09-06）：

初版 causal 实现在 N=512 时 speedup 仅 0.84x（越优化越慢）。ncu 定位：

```
              full      causal
Occupancy     2.08%     2.08%    ← 每 block 仅 1 warp，SM 大量空转
每指令延迟     2.67cyc   3.75cyc  ← warp 内分支分歧惩罚
```

**根因**：一线程一行（Br=32）→ 每 block 只 1 个 warp → 内存延迟期间无其他 warp 可调度（延迟隐藏失效）→ 省下的计算被串行延迟淹没，mask 分支反而添乱。

**解法**：N 放大到 8192（block 数 ≫ 28 个 SM）后 causal 达 2.00x 理论上限；或 Br=256 提高单 block 并行度（N=512 时 causal 从 0.84x → 1.28x）。

**教训**：① 低 occupancy 下谈计算量优化是空中楼阁，先解决延迟隐藏；② 性能对比必须同会话交错测量，WSL2 频率波动可制造 ±50% 假象。

## 实现要点

- **gemm_v0**: PMPP 第 5 章朴素实现，一个线程算 C 的一个元素，全程走全局内存。
- **gemm_v1**: Shared Memory 分块（TILE=32），`As[ty][k] * Bs[k][tx]` 的访问模式天然无 Bank Conflict（源码注释里附了冲突反例对比）。相对 v0 加速 ~1.3x。
- **gemm_v2**: 在 v1 已合并访存的基础上，全局加载改用 `__ldg` 走只读缓存 + `__restrict__`。
- **gemm_v3_cutlass**: 调用 NVIDIA CUTLASS 库的 `cutlass::gemm::device::Gemm`，编译期固化 tile/warp/流水线配置，几乎零运行时开销；相比手写 v2 提速 ~6.5x，展示了工业级库与手写 kernel 的差距。
- **softmax_v1**: 数值安全版（减 max）fused softmax；线程粗化（grid-stride）预扫描 + block 内树形归约到 32 个线程后改用 `__shfl_down_sync` warp 内归约，避免 `__syncthreads` 开销。
- **layernorm_v1**: 一行一 block，线程粗化加载，两次归约（sum → 均值，平方和 → 方差），`rsqrtf(var+eps)` 归一化。
- **rmsnorm_v1**: LLaMA/Qwen 标配的 RMSNorm。相比 LayerNorm 去掉 centering（减均值），只需一次归约（Σx²），且省一次全局显存读写。
- **flashattention_v1**: FlashAttention 前向。Q 行驻留寄存器，K/V 按块搬入 Shared Memory，维护 running max / sum / acc 做 Online Softmax，中间 S 矩阵永不写回 HBM。
- **flashattention_v2**: v1 + causal mask（GPT 自回归必备）。`template<bool IS_CAUSAL>` 编译期双模式零开销；kv 循环上界按 block 粒度截断（`q_row_max/Bc+1`）整块跳过未来信息，对角线块逐元素 mask；N=8192 时 causal 达 2.00x 理论加速。
