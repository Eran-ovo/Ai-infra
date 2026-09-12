# AI Infra CUDA Operators

面向 AI Infra 岗位的 CUDA 算子库：使用 CUDA C++、inline PTX 和 PyTorch
Extension 实现 GEMM、FlashAttention、Softmax、LayerNorm、RMSNorm，并保留从
正确性基线、逐版本优化、Nsight Compute 定位到稳定调度接口的完整证据链。

当前里程碑为 **v0.1.0**。正式数据和原始样本见
[`final_benchmark_sm86.md`](torch_ext/benchmark_results/final_benchmark_sm86.md)、
[`JSON`](torch_ext/benchmark_results/final_benchmark_sm86.json) 与
[`CSV`](torch_ext/benchmark_results/final_benchmark_sm86.csv)。

## 项目亮点

- 手写 Tensor Core 数据通路：`cp.async`、`ldmatrix`、
  `mma.sync.m16n8k16`，FP16 输入、FP32 累加与输出。
- 使用 Nsight Compute 将 GEMM 的 LDSM bank conflict 从 **5.03 亿降至 0**；
  `4096³` 相对 v4 加速 **2.45x**，达到 **13.05 TFLOP/s**。
- FlashAttention 支持连续 `[B,H,N,D]`、full/causal、`D=64/128`；通过缓存 Q
  fragment 和复用 Q/K shared tile，将 shared memory 从 **24.96 KB 降至
  16.64 KB**、实测 occupancy 从 **22.92% 提升至 28.64%**。
- 提供 shape-aware `auto` dispatch、tail/非对齐 fallback、current-stream
  执行、完整正确性回归，以及可复现的 CUDA Event benchmark。

## v0.1.0 正式结果

测试环境：RTX 3060 Laptop 6 GB（Ampere / sm_86）、WSL2 Ubuntu、CUDA 12.4、
PyTorch 2.6.0+cu124。以下只摘录简历相关结果；完整报告包含环境快照、原始
样本、median/MAD/range 与最大误差。

| 场景 | 手写实现 | 公平基线 | 结果 |
|---|---:|---:|---:|
| RMSNorm FP32 `[1024,1024]` | 0.0310 ms | PyTorch 0.1116 ms | **3.59x** |
| LayerNorm FP32 `[1024,1024]` | 0.0298 ms | PyTorch 0.0334 ms | **1.12x** |
| Softmax FP32 `[1024,1024]` | 0.0305 ms | PyTorch 0.0290 ms | 0.95x |
| GEMM `1024³` | auto 0.1191 ms / 18.04 TFLOP/s | v4 0.4340 ms | **3.65x** |
| GEMM `4096³` | auto 10.5286 ms / 13.05 TFLOP/s | v4 25.8464 ms | **2.45x** |
| FA D64 `[2,8,1024,64]` full | auto(v7) 0.6618 ms | v5 0.7489 ms | **1.13x** |
| FA D64 `[2,8,2048,64]` full | auto(v7) 2.5258 ms | v5 2.8350 ms | **1.12x** |

GEMM 与 cuBLAS 使用相同的 FP16 输入、FP32 累加和 FP32 输出语义；`1024³`
达到该 cuBLAS 基线的 148.08%，`4096³` 为 63.77%。FlashAttention 当前是明确的
优化证据链而非对 PyTorch SDPA 的性能超越：上述两个 full shape 分别达到
SDPA 的 31.97% 和 32.13%，因此这里只表述为相对自身基线的优化。

计时使用 CUDA Event、20 次预热、12 轮位置/前序实现双重平衡顺序并取中位数。
正式报告在 clean commit `1d2efa0` 上生成，元数据记录 `git_dirty=false`。

## 稳定 API

实验版本用于消融和展示优化过程，使用方只需要调用稳定入口：

```python
import torch
import ai_infra_ops

# FP16 [M,K] @ [K,N] -> FP32 [M,N]
c = ai_infra_ops.gemm_mma_auto(a, b)

# FP16 [B,H,N,D] -> FP16 [B,H,N,D]，D 支持 64/128
o = ai_infra_ops.flashattention_auto(q, k, v, causal=True)

y0 = ai_infra_ops.softmax(x)
y1 = ai_infra_ops.layernorm(x, 1e-5)
y2 = ai_infra_ops.rmsnorm(x, weight, 1e-5)
```

`gemm_mma_auto` 对小 token-batch 完整 tile 选择 BK32 v10，其他完整 tile 选择
v8，tail 或非对齐 storage 回退 v4。`flashattention_auto` 对 D64 按序列长度选择
v5/v7，对 D128 选择 v6；不支持的维度明确报错。

## 快速复现

```bash
cd torch_ext

# 环境预检、按当前 GPU 架构增量编译、运行完整正确性测试
./build_and_test.sh

# 先验证 benchmark 流程；quick 数据不能用于简历
~/venvs/torch/bin/python bench_final.py --quick --warmup 2 --iters 3

# 生成本机正式 JSON / CSV / Markdown 报告
~/venvs/torch/bin/python bench_final.py --name final_benchmark_local
```

依赖安装、全部 API 和单项 benchmark 命令见
[`torch_ext/README.md`](torch_ext/README.md)。

## 工程结构

```text
torch_ext/csrc/              # PyTorch 绑定、稳定 dispatch 与生产 kernel
torch_ext/test_all.py        # 正确性、边界、非对齐 storage、stream 测试
torch_ext/bench_final.py     # 可复现的统一 benchmark 与报告生成
torch_ext/benchmark_results/ # v0.1.0 正式结果和原始样本
kernels/                     # 从 naive 到 Tensor Core 的独立教学版本
docs/                        # 算子原理与可视化说明
```

下面保留逐版本调优记录。它回答的不只是“最终多快”，还包括每次为什么改、
如何验证假设，以及哪些优化在当前硬件上没有收益。

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
