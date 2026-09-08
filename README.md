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
| GEMM | v4 手写 Tensor Core（fp16 + mma.sync.m16n8k16） | 4096³ | 25.9 ms | 5292 GFLOPS | PASS (relErr=2.4e-05) |
| GEMM | v4 手写 Tensor Core（fp16 输入，`mma.sync.m16n8k16`，fp32 累加） | 1024³ | 0.448 ms | ~4800 GFLOPS | PASS (relErr 7.4e-6) |
| GEMM | v4 手写 Tensor Core | 2048³ | 3.66 ms | ~4700 GFLOPS | PASS (relErr 1.3e-5) |
| GEMM | v4 手写 Tensor Core | 4096³ | 26.0 ms | ~5300 GFLOPS | PASS (relErr 2.4e-5) |
| Softmax | v1 Fused（线程粗化 + 树形归约 + warp shuffle） | 1024 x 1024 | 0.046 ms | - | PASS (maxErr=1.7e-08) |
| LayerNorm | v1 Fused（两次归约求均值/方差） | 1024 x 1024 | 0.030 ms | - | PASS (maxErr=4.5e-06) |
| RMSNorm | v1 Fused（LLaMA 标配，一次归约） | 1024 x 1024 | 0.031 ms | - | PASS (maxErr=1.7e-06) |
| FlashAttention | v1（分块 + Online Softmax，S 矩阵不落地 HBM） | N=512, D=64, Br/Bc=32 | 0.199 ms | - | PASS (maxErr=2.4e-07) |
| FlashAttention | v2（v1 + causal mask，模板双模式） | N=8192, D=64, Br=256 | full 9.43 / causal 4.71 ms | 2.00x | PASS (maxErr=6.4e-07) |
| FlashAttention | v3（一 warp 一行 + smem padding 消 bank conflict） | N=8192, D=64, Br=256 | full 17.7 / causal 8.8 ms | 2.01x | PASS (maxErr=2.4e-07) |

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

**GEMM v4 手写 Tensor Core（fp16 mma.sync）**（2026-09-08）：

v2 的 fp32 FMA 版只有 ~40 GFLOPS，v4 换成手写 `mma.sync.aligned.m16n8k16.row.col` 后直接冲到 **~4.8-5.3 TFLOPS，100x+ 跃升**，与 cuBLAS 的差距从 ~250x 缩到 ~4x。核心差异：FMA 是「每周期 32 个 lane 各 1 次乘加」，mma 是「每条指令整个 warp 算完 16×8×16=2048 次乘加」。

**数据流**：`global fp16 A/B → smem tile → fragment(寄存器) → mma.sync → fp32 acc → global C`。为什么要先过 smem：fragment 布局是「乱序」的（每个 lane 读 `(g, 2*gid)` 这种跳变位置），直接从 global 读会打散合并访存；先用 256 线程按 `tid, tid+256...` 顺序接力搬进 smem（coalesced），再从 smem 按 fragment 布局自由索引（bank 带宽高、代价小）。

**fragment 布局（踩坑重灾区）**：mma 的 A/B/C 不是线性数组，而是打散存进 32 个 lane 的寄存器。`g = lane/4`（groupID，0..7）、`gid = lane%4`（threadID_in_group，0..3）：

| 矩阵 | 每 lane 持有 | 布局 |
|------|------------|------|
| A (16×16, row-major) | 4×half2 a0..a3 | a0=(g,2gid) a1=(g+8,2gid) a2=(g,2gid+8) a3=(g+8,2gid+8) |
| B (16×8, col-major) | 2×half2 b0,b1 | b0=(2gid,g)/(2gid+1,g) b1=(2gid+8,g)/(2gid+9,g) |
| C (16×8, fp32) | 4×float c0..c3 | c0=(g,2gid) c1=(g,2gid+1) c2=(g+8,2gid) c3=(g+8,2gid+1) |

**踩坑**：A fragment 的 a1/a2 行索引写反 → **rows 16-63 输出全 0、前 16 行正确**（后 8 行 A 被当前 8 行用，矩阵错位）——**前 16 行对、后 48 行错的不均匀错位（而非全错）是 fragment 映射 bug 的标志**。定位手法：单 block + A=单位阵 + B=行号的确定性用例，一列 dump 出错误模式，一眼定位是哪半块错位。另：CPU 对拍必须和 GPU 吃**同一份 fp16 量化后的输入**（否则 0.2/0.4 这类无法精确表示的值会让两边基准不同、误报 FAIL），阈值用**相对误差**而非绝对误差（fp16 GEMM 的绝对误差随 K 线性增长）。

**性能解释**：2048 比 1024 慢 **8 倍**是健康的线性扩展（工作量 2·M·N·K ∝ D³），判断标准看 **GFLOPS 是否持平**（实测 1024→2048→4096 为 4.8→4.7→5.3 TFLOPS，持平即健康）。若超出 8 倍（GFLOPS 塌陷）则是笔记本 GPU 撞 TDP/温度墙降频——判定铁证是同一进程里 **cuBLAS 也按同比例变慢**（cuBLAS 不受我们的 kernel 影响）。


## 实现要点

- **gemm_v0**: PMPP 第 5 章朴素实现，一个线程算 C 的一个元素，全程走全局内存。
- **gemm_v1**: Shared Memory 分块（TILE=32），`As[ty][k] * Bs[k][tx]` 的访问模式天然无 Bank Conflict（源码注释里附了冲突反例对比）。相对 v0 加速 ~1.3x。
- **gemm_v2**: 在 v1 已合并访存的基础上，全局加载改用 `__ldg` 走只读缓存 + `__restrict__`。
- **gemm_v3_cutlass**: 调用 NVIDIA CUTLASS 库的 `cutlass::gemm::device::Gemm`，编译期固化 tile/warp/流水线配置，几乎零运行时开销；相比手写 v2 提速 ~6.5x，展示了工业级库与手写 kernel 的差距。
- **gemm_v4_mma**: 手写 Tensor Core GEMM——fp16 输入用 `mma.sync.aligned.m16n8k16.row.col` 内联 PTX 指令，绕开 wmma API 与 CUTLASS，亲手管理 fragment 布局 + smem 搬运 + 8-warp 分块。相比 v2 的 fp32 FMA（~40 GFLOPS 量级）直接跃升到 5+ TFLOPS（100x+），与 cuBLAS fp16 差距从 250x 缩到 4x。踩坑见上方"GEMM v4 手写 Tensor Core"调优记录。
- **softmax_v1**: 数值安全版（减 max）fused softmax；线程粗化（grid-stride）预扫描 + block 内树形归约到 32 个线程后改用 `__shfl_down_sync` warp 内归约，避免 `__syncthreads` 开销。
- **layernorm_v1**: 一行一 block，线程粗化加载，两次归约（sum → 均值，平方和 → 方差），`rsqrtf(var+eps)` 归一化。
- **rmsnorm_v1**: LLaMA/Qwen 标配的 RMSNorm。相比 LayerNorm 去掉 centering（减均值），只需一次归约（Σx²），且省一次全局显存读写。
- **flashattention_v1**: FlashAttention 前向。Q 行驻留寄存器，K/V 按块搬入 Shared Memory，维护 running max / sum / acc 做 Online Softmax，中间 S 矩阵永不写回 HBM。
- **flashattention_v2**: v1 + causal mask（GPT 自回归必备）。`template<bool IS_CAUSAL>` 编译期双模式零开销；kv 循环上界按 block 粒度截断（`q_row_max/Bc+1`）整块跳过未来信息，对角线块逐元素 mask；N=8192 时 causal 达 2.00x 理论加速。
- **flashattention_v3**: FA2 的核心重划分——一 warp 一行 Q（v2 是一线程一行）。每 lane 只存 DQ=D/32=2 维（`q_reg[2]+acc[2]`），寄存器 255→40/thread，occupancy 2%→79%。Q 行入 smem，每 lane 独立算完整点积（避开 causal 分支下 shuffle 死锁）；`Ktile/Vtile[Bc][D+1]` padding 消除列访问的 32 路 bank conflict（提速 1.9x）。仍比 v2 慢 1.8x：K/V 随 block 数增多而重复搬运 + Q 点积冗余，说明 occupancy 与算术强度需平衡。
