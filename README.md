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

## 历史独立程序 Benchmark（2026-09-05 实测）

| 算子 | 版本 | 规模 | 耗时 | 性能 | 验证 |
|------|------|------|------|------|------|
| GEMM | v0 Naive（一线程一元素） | 1024³ | 3.53 ms | 607 GFLOPS | PASS |
| GEMM | v1 Shared Memory Tiling（32x32，无 Bank Conflict） | 1024³ | 2.77 ms | 779 GFLOPS | PASS |
| GEMM | v2 Coalesced + `__ldg` 只读缓存 | 1024³ | 2.72 ms | 789 GFLOPS | PASS |
| GEMM | v3 CUTLASS 工业级实现（编译期固化 tile/warp/流水线） | 1024³ | 0.408 ms | 5259 GFLOPS | PASS |
| GEMM | v4 手写 Tensor Core（fp16 输入 + `mma.sync.m16n8k16`，fp32 累加） | 1024³ | 0.542 ms | 3963 GFLOPS | PASS |
| GEMM | v4 手写 Tensor Core（同上） | 2048³ | 3.889 ms | 4417 GFLOPS | PASS |
| GEMM | v4 手写 Tensor Core（同上） | 4096³ | 26.532 ms | 5180 GFLOPS | PASS |
| Softmax | v1 Fused（线程粗化 + 树形归约 + warp shuffle） | 1024 x 1024 | 0.046 ms | - | PASS (maxErr=1.7e-08) |
| LayerNorm | v1 Fused（两次归约求均值/方差） | 1024 x 1024 | 0.030 ms | - | PASS (maxErr=4.5e-06) |
| RMSNorm | v1 Fused（LLaMA 标配，一次归约） | 1024 x 1024 | 0.031 ms | - | PASS (maxErr=1.7e-06) |
| FlashAttention | v1（分块 + Online Softmax，S 矩阵不落地 HBM） | N=512, D=64, Br/Bc=32 | 0.199 ms | - | PASS (maxErr=2.4e-07) |
| FlashAttention | v2（v1 + causal mask，模板双模式） | N=8192, D=64, Br=256 | full 9.43 / causal 4.71 ms | 2.00x | PASS (maxErr=6.4e-07) |
| FlashAttention | v3（一 warp 一行 + smem padding 消 bank conflict） | N=8192, D=64, Br=256 | full 17.7 / causal 8.8 ms | 2.01x | PASS (maxErr=2.4e-07) |
| FlashAttention | v4（fp16 Tensor Core，P 经 shared-memory `Ps` 中转） | N=8192, D=64 | full 4.2508 / causal 2.4759 ms | - | PASS (N=129 tail maxErr=9.8e-04) |
| FlashAttention | v5（v4 + P fragment 寄存器直连） | N=8192, D=64 | full 3.3810 / causal 2.1193 ms | v4/v5=1.26x/1.17x | PASS (与 v4 误差相同) |
| FlashAttention | v6（v5 + D=128） | B=2,H=2,N=4096,D=128 | full 7.3629 / causal 3.5975 ms | vs SDPA 3.95x/3.57x | PASS |
| FlashAttention | v7（D=64，Q fragment 寄存器缓存 + Q/K smem 复用） | B=2,H=8,N=1024,D=64 | full 0.6500 / causal 0.4254 ms | vs v5 1.13x/1.09x | PASS (v5/v7 逐元素相同) |

> 注：这是各个独立 `.cu` 程序的阶段性历史数据。当前跨实现对比统一使用 [`torch_ext/bench_ops.py`](torch_ext/bench_ops.py)：CUDA Event 计时、预热、交错多轮中位数，并保证 dtype、累加精度和输出 dtype 一致。不要把两种测量口径的数字混用。

## 当前稳定接口与简历主线

实验版本用于展示优化过程，项目使用方优先调用 PyTorch Extension 中的稳定入口：

```python
out = ai_infra_ops.gemm_mma_auto(a, b)
attn = ai_infra_ops.flashattention_auto(q, k, v, causal=False)
```

`gemm_mma_auto` 使用 shape-aware 三层调度：小 token-batch 的完整 tile 选择 BK32 v10，其他形状选择更稳定的 v8，tail 或非对齐 storage 最终回退 v4。Transformer projection 形状由 [`torch_ext/bench_gemm_transformer.py`](torch_ext/bench_gemm_transformer.py) 复现。这个接口标志着 GEMM 主线从“实验 kernel”进入“可用算子 API”阶段。

当前简历表述草稿：

> 基于 CUDA C++/inline PTX 与 PyTorch Extension 开发高性能算子库，实现 GEMM、FlashAttention、Softmax、LayerNorm、RMSNorm；手写 `mma.sync`、`ldmatrix` 与 `cp.async` 流水，通过 Nsight Compute 将 GEMM LDSM bank conflict 从 5.03 亿降至 0，RTX 3060 Laptop 上 `4096³` FP16-input/FP32-output GEMM 相对初版加速约 2.4x、达到约 14 TFLOP/s，并实现 shape-aware fast-path/fallback 调度及完整边界、stream 正确性测试。
>
> 使用 Nsight Compute 定位 FlashAttention 的 shared-memory/MIO 压力，通过寄存器缓存 Q MMA fragment 并复用 Q/K shared tile，将 shared memory 从 24.96 KB 降至 16.64 KB、实测 occupancy 从 22.92% 提升至 28.64%，在 D64、N>=1024 的多组 B×H shape 上相对基线加速约 1.1x–1.14x，并实现 D64/D128 稳定调度。

FlashAttention 也已有第一版稳定入口：`head_dim=64` 且 `N>=1024` 路由到 v7，
较短 D64 路由到 v5，`head_dim=128` 路由到 v6，其他维度明确报错。D64 阈值来自
固定 N 扫描和固定 N 的 B×H 扫描，仍需在更多真实模型 shape 上继续验证。

构建工程化里程碑已完成：`torch_ext/build_and_test.sh` 会检查 Python、CUDA、
GPU 架构和 Ninja，随后执行增量编译与完整正确性测试；同时固定 Ninja
`1.11.1.4`，规避 1.13.2 损坏 `.ninja_deps` 并重复全量编译的已知回归。
连续两次实机验证中，第二次构建已达到 `ninja: no work to do.`。

最终 benchmark 流程也已固化为 `torch_ext/bench_final.py`：使用 CUDA Event、
位置/前序实现双重平衡顺序和多轮中位数，保存环境、每轮原始样本、MAD、
极差、正确性误差及 TFLOP/s。当前 sm_86 实机正式报告见
[`final_benchmark_sm86.md`](torch_ext/benchmark_results/final_benchmark_sm86.md)。
该报告在 clean commit `1d2efa0` 上运行，元数据记录 `git_dirty=false`，原始
样本同时保存为 JSON/CSV。至此第一版简历主线已经收尾，后续 kernel 迭代应
建立在该版本基线上，不能用新的单次数据覆盖这份可复现结果。

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

**FlashAttention v3（warp-per-row）调优三部曲**（2026-09-06/07）：

v3 把"一线程一行"改成"一 warp 一行"（FA2 的重划分思想），每 lane 只存 `D/32=2` 维。过程中踩了三个坑，逐个定位：

| 阶段 | full 耗时 | 关键指标 | 问题 |
|------|----------|---------|------|
| 初版 | 死锁 / FAIL | - | causal 分支发散下用 `__shfl_xor` 归约 → 死锁；`q_row>=N` 提前 return → `__syncthreads` 死锁 |
| 修死锁 | 33.3 ms | occupancy 79%，bank conflict 9.4 亿 | Q 入 smem + 每 lane 独立点积解决发散；但 K/V 列访问 32 路 bank conflict |
| 加 padding | **17.7 ms** | bank conflict 0.67 亿（-14x） | `Ktile/Vtile[Bc][D+1]`，步长 65 与 32 互质 |

**v3 仍比 v2（9.7ms）慢 1.8x 的归因**：
1. **K/V 重复搬运**：block 数 32→1024（每 block Q 行 256→8），每 block 仍搬全量 K/V，DRAM 读取 增加（被 L2 缓存摊薄）
2. **bank conflict**（已修复，贡献 1.9x 提速）
3. **Q 点积冗余**：每 lane 对同一行 Q 独立算 64 维点积，算术量高于 v2
4. **shuffle 广播**：4e 权重收集的额外开销

**教训**：① occupancy、搬运量、算术强度是三角债，拉满一个可能拖累另一个，优化是找平衡点；② shared memory 列访问必查 bank conflict（步长是 32 倍数时全撞）；③ 发散分支里禁用跨 lane shuffle，会死锁。

**GEMM v4-v10 手写 Tensor Core（`mma.sync` / `cp.async` / `ldmatrix`）**（2026-09-08/12）：

v2 的 fp32 FMA 版在 1024³ 是 789 GFLOPS；v4 换成手写 `mma.sync.aligned.m16n8k16.row.col` 后约 5.5–5.8 TFLOP/s。继续加入 16-byte 对齐 fast path 和 `cp.async` 双缓冲后，v6 达到约 8.3–8.9 TFLOP/s，相对 v4 提升 1.49–1.59x。相同 fp16 输入、fp32 累加、fp32 输出的 cuBLAS 为 12.6–22.0 TFLOP/s。核心差异：FMA 是「每周期 32 个 lane 各 1 次乘加」，mma 是「每条指令整个 warp 算完 16×8×16=2048 次乘加」。

**数据流**：`global fp16 A/B → smem tile → fragment(寄存器) → mma.sync → fp32 acc → global C`。为什么要先过 smem：fragment 布局是「乱序」的（每个 lane 读 `(g, 2*gid)` 这种跳变位置），直接从 global 读会打散合并访存；先用 256 线程按 `tid, tid+256...` 顺序接力搬进 smem（coalesced），再从 smem 按 fragment 布局自由索引（bank 带宽高、代价小）。

**fragment 布局（踩坑重灾区）**：mma 的 A/B/C 不是线性数组，而是打散存进 32 个 lane 的寄存器。`g = lane/4`（groupID，0..7）、`gid = lane%4`（threadID_in_group，0..3）：

| 矩阵 | 每 lane 持有 | 布局 |
|------|------------|------|
| A (16×16, row-major) | 4×half2 a0..a3 | a0=(g,2gid) a1=(g+8,2gid) a2=(g,2gid+8) a3=(g+8,2gid+8) |
| B (16×8, col-major) | 2×half2 b0,b1 | b0=(2gid,g)/(2gid+1,g) b1=(2gid+8,g)/(2gid+9,g) |
| C (16×8, fp32) | 4×float c0..c3 | c0=(g,2gid) c1=(g,2gid+1) c2=(g+8,2gid) c3=(g+8,2gid+1) |

**踩坑**：A fragment 的 a1/a2 行索引写反 → **rows 16-63 输出全 0、前 16 行正确**（后 8 行 A 被当前 8 行用，矩阵错位）——**前 16 行对、后 48 行错的不均匀错位（而非全错）是 fragment 映射 bug 的标志**。定位手法：单 block + A=单位阵 + B=行号的确定性用例，一列 dump 出错误模式，一眼定位是哪半块错位。另：CPU 对拍必须和 GPU 吃**同一份 fp16 量化后的输入**（否则 0.2/0.4 这类无法精确表示的值会让两边基准不同、误报 FAIL），阈值用**相对误差**而非绝对误差（fp16 GEMM 的绝对误差随 K 线性增长）。

**性能解释**：2048 比 1024 慢 **8 倍**是健康的线性扩展（工作量 2·M·N·K ∝ D³），判断标准看 **TFLOP/s 是否持平**。v6 在三个尺寸均约 8.3–8.9 TFLOP/s，说明扩展正常；cuBLAS 在大尺寸升至约 22 TFLOP/s，则说明小尺寸尚未充分摊薄固定开销并填满硬件。若手写与 cuBLAS 在同一进程中同时按比例变慢，才更像笔记本 GPU 的频率/TDP 波动。

v7 在相同 `64×64×16` tile 上把标量 fragment load 换成 warp 级 `ldmatrix`，但 `4096³` 仍约 18.01 ms。NCU 显示 5.03 亿次 LDSM bank conflict：A/B 原行跨度分别为 32/128 bytes，映射到 32 个 4-byte bank 后周期性重叠。v8 给每行增加 8 个 half，使跨度变成 48/144 bytes；8 行的 16-byte 段恰好落到互不重叠的 bank 区间。冲突降到 0，`4096³` 降至 11.06 ms（12.43 TFLOP/s），相对 v7 提速 1.63x、达到同轮同语义 cuBLAS 的约 66%。这里的关键不是“换了一条高级指令”，而是指令要求与 shared-memory layout 必须共同设计。

v9 将 v8 的 layout 与 v6 的 `cp.async` 双缓冲组合，但在当前 `BK=16`、`64×64` tile 下没有继续提速：`4096³` 为 10.1097 ms，相比 v8 的 9.9003 ms 慢 2.1%；`2048³` 慢 11.6%。NCU 显示 v9 的 shared memory 从 5.38 KB 增至 10.75 KB，registers/thread 仍为 48，occupancy 仍由寄存器限制；主要原因是每个 tile 的 MMA 计算窗口太短，不足以覆盖 `commit/wait/barrier` 开销。这个负实验明确了下一步应扩大预取后的计算量，而不是盲目增加 pipeline stage。

v10 在独立文件中把 `BK=16` 改为 `BK=32`，每次 stage 搬入 `64×32` 的 A tile 和 `32×64` 的 B tile，并连续执行两个 `ldmatrix + mma.sync` K-slice。`4096³` 达到 9.6866 ms，比 v8 快约 1.0%；但 `2048³` 为 1.2295 ms，比 v8 慢约 26.9%。这说明增加计算窗口确实可能让大矩阵受益，但 shared memory 从 5.38 KB 增至 19.46 KB，单一 BK 配置不能覆盖所有尺寸。

生产入口 `gemm_mma_auto` 将版本链收敛为 shape-aware dispatch：`M<=256` 且满足完整 BK32 tile/16B 对齐时使用 v10，其余调用 v8，并由 v8 对 tail/非对齐输入回退 v4。规则刻意保守，因为 RTX 3060 Laptop 的频率状态会让边界形状结果反转；dispatch 的目标是稳定 API 和安全回退，不是用一次 benchmark 过拟合所有尺寸。

**测量口径差异（重要）**：首轮含 CUDA context、库初始化、冷缓存和未稳态频率，不能作为 kernel 稳态性能。统一脚本先预热，再用 CUDA Event 测当前 stream 上的 GPU 时间，交错多轮取中位数；GEMM 版本链由 [`torch_ext/bench_gemm_mma.py`](torch_ext/bench_gemm_mma.py) 复现。尤其不能直接拿 `gemm_mma` 与 `torch.matmul(fp16)` 比：前者输出 fp32，后者输出 fp16。扩展中的 `gemm_cublas_fp32` 明确调用 `cublasGemmEx`，把输入、累加和输出语义完全对齐。

**FlashAttention v5：删除 P 的 shared-memory 中转**（2026-09-10）：

v4 在 QK MMA 和 online softmax 后，把寄存器中的概率写入 `Ps[64][65]`，随后按 PV MMA 的 A-fragment 布局重新加载。观察到 QK 的两个相邻 `m16n8` 输出 fragment 本身就能拼成 PV 所需的 row-major `m16k16` A fragment，因此 v5 直接把 `S[t]`/`S[t+1]` 转成 `pa0..pa3`，删除 `Ps`。

| 指标（N=8192 full） | v4 | v5 |
|---|---:|---:|
| Static shared memory/block | 33.28 KB | 24.96 KB |
| 可驻留 block/SM（smem 限制） | 2 | 3 |
| 理论 / 实测 occupancy | 16.67% / 15.55% | 25.00% / 21.28% |
| CUDA Event 耗时 | 4.2508 ms | 3.3810 ms |

减少的 `8,320 B = 64×65×sizeof(half)` 与 `Ps` 尺寸严格吻合。v4/v5 使用相同输入、精度、tile、HMMA 数量和交错计时，因而 1.26x 提速可以归因于移除 shared-memory store/load 并提高并发驻留。这个实验也说明：优化应形成“资源假设 → 单变量代码变化 → 正确性 → benchmark → NCU 解释”的证据链。


## 实现要点

- **gemm_v0**: PMPP 第 5 章朴素实现，一个线程算 C 的一个元素，全程走全局内存。
- **gemm_v1**: Shared Memory 分块（TILE=32），`As[ty][k] * Bs[k][tx]` 的访问模式天然无 Bank Conflict（源码注释里附了冲突反例对比）。相对 v0 加速 ~1.3x。
- **gemm_v2**: 在 v1 已合并访存的基础上，全局加载改用 `__ldg` 走只读缓存 + `__restrict__`。
- **gemm_v3_cutlass**: 调用 NVIDIA CUTLASS 库的 `cutlass::gemm::device::Gemm`，编译期固化 tile/warp/流水线配置，几乎零运行时开销；相比手写 v2 提速 ~6.5x，展示了工业级库与手写 kernel 的差距。
- **gemm_v4-v10 + auto**: v4 手写 `mma.sync` 与 fragment；v5 增加 16-byte fast path；v6 使用 `cp.async`；v7/v8 完成 `ldmatrix` bank-conflict 定位与 padding；v9/v10验证流水窗口；最终由 `gemm_mma_auto` 提供 v10/v8/v4 三层稳定调度。实验版本保留证据链，业务入口不暴露版本选择负担。
- **softmax_v1**: 数值安全版（减 max）fused softmax；线程粗化（grid-stride）预扫描 + block 内树形归约到 32 个线程后改用 `__shfl_down_sync` warp 内归约，避免 `__syncthreads` 开销。
- **layernorm_v1**: 一行一 block，线程粗化加载，两次归约（sum → 均值，平方和 → 方差），`rsqrtf(var+eps)` 归一化。
- **rmsnorm_v1**: LLaMA/Qwen 标配的 RMSNorm。相比 LayerNorm 去掉 centering（减均值），只需一次归约（Σx²），且省一次全局显存读写。
- **flashattention_v1**: FlashAttention 前向。Q 行驻留寄存器，K/V 按块搬入 Shared Memory，维护 running max / sum / acc 做 Online Softmax，中间 S 矩阵永不写回 HBM。
- **flashattention_v2**: v1 + causal mask（GPT 自回归必备）。`template<bool IS_CAUSAL>` 编译期双模式零开销；kv 循环上界按 block 粒度截断（`q_row_max/Bc+1`）整块跳过未来信息，对角线块逐元素 mask；N=8192 时 causal 达 2.00x 理论加速。
- **flashattention_v3**: FA2 的核心重划分——一 warp 一行 Q（v2 是一线程一行）。每 lane 只存 DQ=D/32=2 维（`q_reg[2]+acc[2]`），寄存器 255→40/thread，occupancy 2%→79%。Q 行入 smem，每 lane 独立算完整点积（避开 causal 分支下 shuffle 死锁）；`Ktile/Vtile[Bc][D+1]` padding 消除列访问的 32 路 bank conflict（提速 1.9x）。仍比 v2 慢 1.8x：K/V 随 block 数增多而重复搬运 + Q 点积冗余，说明 occupancy 与算术强度需平衡。
- **flashattention_v4**: QKᵀ 与 PV 都改为手写 `mma.sync.m16n8k16`，fp16 输入、fp32 softmax/累加；保留 full/causal 和任意 N 的 K/V tail mask。softmax 概率先写入 padded `Ps[64][65]`，再按 PV 的 A fragment 布局重载，是便于验证 fragment 映射的清晰基线。
- **flashattention_v5**: 复用 QK 输出 fragment 的寄存器布局，把相邻两个 `16×8` 的 P tile 直接拼成 PV 的 `16×16` A fragment，删除 `Ps` 中转。shared memory 33.28→24.96 KB，理论 occupancy 16.67%→25%（实测 15.55%→21.28%），N=8192 full/causal 相对 v4 提速 1.26x/1.17x。
- **flashattention_v5 BHD**: 同一 kernel 兼容 `[N,64]` 和连续 `[B,H,N,64]`；`grid.x` 枚举 Q block，`grid.y` 枚举展平的 batch×head。B=2/H=8/N=1024 时，一次 BHD launch 相比 Python 逐 head dispatch 在 full/causal 分别快 3.11x/5.13x，并与 PyTorch SDPA 完成正确性对拍。
- **flashattention_v6**: 独立 D=128 实验版。QK 归约步数 4→8、PV 输出 tile 8→16、shared memory 24.96→49.54 KB；full 128 registers/thread、causal 163 registers/thread，仍无 spill，说明下一瓶颈是寄存器生命周期和 tile 设计。
- **flashattention_v7**: D=64 优化版。Q 先合作式进入 shared memory，再将每个 lane 的 MMA fragment 缓存到寄存器，使 Q/K 复用同一 tile；shared memory 24.96→16.64 KB，实测 occupancy 22.92%→28.64%，固定 BHD shape 的 full/causal 相对 v5 提速 1.13x/1.09x。完成 N 与 B×H 扫描后，生产 `auto` 在 D64、N>=1024 时选择 v7。
