# torch_ext：把手写 CUDA kernel 封装成 PyTorch 算子

把 `kernels/` 下 5 类手写 CUDA kernel（GEMM / Softmax / LayerNorm / RMSNorm / FlashAttention）统一封装成一个 PyTorch Extension 模块 `ai_infra_ops`。当前暴露 16 个手写算子/调度入口，以及 1 个只用于公平测速的 cuBLAS 基线入口。

## 环境

```bash
# 隔离 venv（不污染系统 Python）
python3 -m venv ~/venvs/torch
~/venvs/torch/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/venvs/torch/bin/pip install numpy ninja
```

## 编译 & 运行

```bash
cd torch_ext
export PATH=~/venvs/torch/bin:/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3060 Laptop = Ampere sm_86
# WSL 内存有限时避免 nvcc 并行编译多个大 CUDA TU 导致 OOM
export MAX_JOBS=1
~/venvs/torch/bin/python setup.py build_ext --inplace
~/venvs/torch/bin/python test_all.py
~/venvs/torch/bin/python bench_ops.py
```

## API

```python
import torch  # 先加载 PyTorch 运行时依赖（libc10/torch_cuda）
import ai_infra_ops

ai_infra_ops.rmsnorm(x, g, eps)          # x [B,N] fp32, g [N]
ai_infra_ops.softmax(x)                  # x [B,N] fp32，沿 axis=-1
ai_infra_ops.layernorm(x, eps)           # x [B,N] fp32，无 affine
ai_infra_ops.gemm(a, b)                  # a [M,K] b [K,N] fp32 -> [M,N]
ai_infra_ops.gemm_mma(a, b)              # a [M,K] b [K,N] fp16 -> [M,N] fp32（手写 Tensor Core）
ai_infra_ops.gemm_mma_vec(a, b)          # GEMM v5：对齐 fast path + 16-byte 同步搬运
ai_infra_ops.gemm_mma_async(a, b)        # GEMM v6：cp.async + shared-memory 双缓冲
ai_infra_ops.gemm_mma_ldmatrix(a, b)     # GEMM v7：ldmatrix 直接加载 fragment（冲突反例）
ai_infra_ops.gemm_mma_ldmatrix_padded(a, b)  # GEMM v8：ldmatrix + 无冲突 padded layout
ai_infra_ops.gemm_mma_ldmatrix_async_padded(a, b)  # GEMM v9：cp.async + ldmatrix + padding
ai_infra_ops.gemm_mma_v10(a, b)            # GEMM v10：BK=32，两个 K-slice/async stage
ai_infra_ops.gemm_mma_auto(a, b)           # 稳定入口：shape-aware v10/v8/v4 dispatch
ai_infra_ops.gemm_cublas_fp32(a, b)      # 同语义 cuBLAS 基线，仅用于 benchmark
ai_infra_ops.flashattention(q, k, v, causal)  # q/k/v [N,64] fp32
ai_infra_ops.flashattention_fp16(q, k, v, causal)  # v4：fp16 Tensor Core，P 经 Ps 中转
ai_infra_ops.flashattention_v5(q, k, v, causal)    # v5：[N,64] 或 [B,H,N,64]，P fragment 寄存器直连
ai_infra_ops.flashattention_v6(q, k, v, causal)    # v6：[N,128] 或 [B,H,N,128]，D=128 实验版
ai_infra_ops.flashattention_auto(q, k, v, causal)  # 稳定入口：按 D=64/128 路由 v5/v6
```

## 实测结果（RTX 3060 Laptop，`bench_ops.py` 复现）

| 算子 | 手写 CUDA | 同语义基线 | baseline/custom |
|------|----------|-------------|--------|
| RMSNorm fp32 `[1024,1024]` | 0.0308 ms | `F.rms_norm` 0.1113 ms | **3.61x** |
| Softmax fp32 `[1024,1024]` | 0.0298 ms | `torch.softmax` 0.0296 ms | 0.99x |
| LayerNorm fp32 `[1024,1024]`（无 affine） | 0.0294 ms | `F.layer_norm` 0.0342 ms | 1.16x |
| GEMM fp32 `1024³`（TF32 off） | 2.4011 ms / 0.89 TFLOP/s | cuBLAS 0.2971 ms / 7.23 TFLOP/s | 0.12x |
| GEMM MMA v8 ldmatrix+padding `1024³`，fp32 输出 | 0.1209 ms / 17.76 TFLOP/s | cuBLAS 0.1779 ms / 12.07 TFLOP/s | 1.47x |
| GEMM MMA v8 ldmatrix+padding `2048³`，fp32 输出 | 0.9964 ms / 17.24 TFLOP/s | cuBLAS 1.0273 ms / 16.72 TFLOP/s | 1.03x |
| GEMM MMA v8 ldmatrix+padding `4096³`，fp32 输出 | 9.9003 ms / 13.88 TFLOP/s | cuBLAS 6.7062 ms / 20.49 TFLOP/s | 0.68x |
| FlashAttention v5 D=64 | 0.4025 ms | PyTorch SDPA 0.1272 ms | 0.32x |
| FlashAttention v6 D=128 | 0.8789 ms | PyTorch SDPA 0.2388 ms | 0.27x |

默认测速使用 CUDA Event、20 次预热、每组 100 次迭代、交错 7 轮取中位数；GEMM 版本链可用 `bench_gemm_mma.py` 指定尺寸顺序、预热、迭代和轮数。正确性测试与性能测试分离：`test_all.py` 不输出性能结论。GEMM MMA 的两侧都使用 fp16 输入、fp32 累加和 fp32 输出；这是关键约束，因为直接比较 `torch.matmul(fp16)` 会得到 fp16 输出，数值语义和写回带宽都不一致。笔记本 GPU 会受温度与功耗墙影响，因此表中结果只表示同一轮、同一语义下的实机观测；尤其 `1024³` 超过该 cuBLAS 入口不能外推成“普遍超过 cuBLAS”。

### GEMM v4 → v8：从搬运流水到 `ldmatrix` layout

五版保持相同的 `64×64×16` block tile、8 warps/block、MMA 数量和 fp32 accumulator，只逐步改变数据搬运路径：

- v4 scalar：每线程多次搬运单个 half，带通用 tail predicate。
- v5 vec：为 tile 整除且输入指针 16-byte 对齐的形状增加 fast path，每线程一次 16-byte 同步搬运；其他情况回退 v4。`contiguous` tensor 仍可能因非零 `storage_offset` 而地址未对齐，所以 dispatch 检查真实 `data_ptr()`。
- v6 async：保持 v5 的字节映射，只把 global→shared 改为 `cp.async` 双缓冲。
- v7 ld：回到单缓冲同步复制，只把 shared→register 的标量 fragment load 替换为 warp 级 `ldmatrix`。这是故意保留的冲突反例。
- v8 ld+pad：只把 A/B 的 shared-memory 行跨度分别从 16/64 half 改成 24/72 half，消除 `ldmatrix` bank conflict。
- v9 async+ld+pad：在 v8 layout 上打开双缓冲 `cp.async`，验证 global→shared 预取是否能覆盖 MMA 和 `ldmatrix` 的执行时间。

[`ldmatrix.sync.aligned.m8n8.x4.shared.b16`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix) 让整个 warp 合作加载四个 `8×8` fp16 子矩阵。A 的 `16×16` fragment 按 `(上左、下左、上右、下右)` 排列，恰好落到 MMA 的 `a0..a3`；B 在 shared memory 中按 `K×N` 行主序保存，使用 `.x2.trans` 将两个 `8×8` 子矩阵转置装入 `b0/b1`，匹配 `mma.sync...row.col` 的列主序 B fragment。所有 32 个 lane 必须一致执行 `.sync.aligned` 指令，且每个行首地址满足 16-byte 对齐；不能把它放进只有部分 lane 进入的分支。

#### 为什么“用了 `ldmatrix`”却没有立刻变快

Ampere shared memory 有 32 个 bank，每个 bank 宽 4 bytes，可用
`bank = floor(byte_address / 4) mod 32` 分析。一次 `m8n8.b16` 的一行是 8 个 half，即连续 16 bytes、覆盖 4 个 bank：

- A 原始行跨度 `16 half = 32 bytes = 8 banks`，8 行起始 bank 为 `0,8,16,24,0,8,16,24`，后四行与前四行重叠。
- B 原始行跨度 `64 half = 128 bytes = 32 banks`，每一行都从同一个 bank 开始，是更严重的冲突。
- 增加 8 个 half 后，A 行跨度为 `48 bytes = 12 banks`，8 行起点为 `0,12,24,4,16,28,8,20`；每行覆盖的 4-bank 区间互不重叠。
- B 行跨度变成 `144 bytes = 36 banks ≡ 4 (mod 32)`，8 行依次覆盖 `0..3, 4..7, ..., 28..31`。

因此 padding 不是经验参数，而是由“每行覆盖 4 个 bank”推导出的最小 16-byte 偏移。行跨度仍是 16-byte 的倍数，也没有破坏向量复制和 `ldmatrix.aligned` 的对齐要求。代价是每 block shared memory 从约 4.10 KB 增到 5.38 KB。

#### 同一会话 CUDA Event 结果

命令：`python bench_gemm_mma.py --sizes 4096 2048 1024 --warmup 30 --iters 20 --rounds 9`。反转尺寸顺序是为了观察笔记本温度/频率漂移；表内每一行仍是所有实现交错执行后取中位数。

| Shape | v4 scalar | v5 vec | v6 async | v7 ld | v8 ld+pad | cuBLAS |
|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 0.4075 ms | 0.2603 ms | 0.2534 ms | 0.2601 ms | **0.1209 ms** | 0.1779 ms |
| 2048³ | 3.4137 ms | 2.1450 ms | 2.0779 ms | 2.0866 ms | **0.9964 ms** | 1.0273 ms |
| 4096³ | 23.8806 ms | 16.5615 ms | 16.2755 ms | 16.2502 ms | **9.9003 ms** | 6.7062 ms |

v7 在 4096³ 只有 8.46 TFLOP/s，说明减少 fragment 指令并不足以抵消冲突序列化；v8 在同一轮达到 13.88 TFLOP/s，相对 v7 提速 **1.64x**、相对 v4 提速 **2.41x**，达到该轮同语义 cuBLAS 的约 **68%**。2048³ 略快于该轮 cuBLAS 调用；1024³ 也快于该轮 cuBLAS 调用，但这些结果受尺寸、算法选择与笔记本频率影响，不作普遍化宣传。

### GEMM MMA 的 NCU/SASS 证据

对 `4096³` 的 v7/v8 使用相同 NCU shared-memory 指标。两版 LDSM 有效数据量均为 12.88 GB，说明计算工作量没有变化；冲突计数从 **503,316,480 降为 0**：

| 指标 | v7 ld | v8 ld+pad | 说明 |
|---|---:|---:|---|
| LDSM shared load bytes | 12.88 GB | 12.88 GB | 有效 fragment 数据不变 |
| LDSM bank conflicts | 503,316,480 | **0** | padding 消除冲突 |
| Static shared memory/block | 4.10 KB | 5.38 KB | 用容量换无冲突布局 |
| CUDA Event `4096³` | 18.0124 ms | **11.0606 ms** | 1.63x |

反汇编可看到 `LDSM.16.M88.4`/`LDSM.16.MT88.2`，证明编译结果确实使用了 `ldmatrix`；v6 中则可看到 `LDGSTS.E.BYPASS.128`，对应 `cp.async`。NCU 的多 pass `Duration` 会受 replay 和采样时 GPU 频率影响，只用于解释瓶颈，最终速度采用 CUDA Event。这个阶段的完整证据链是：**fragment 指令开销假设 → v7 单变量修改 → benchmark 未改善 → NCU 发现 5 亿次 bank conflict → 按 bank 公式推导 padding → v8 冲突归零且提速 1.63x**。

### v9 组合实验：为什么双缓冲没有继续提速

v9 只组合已经验证过的两个机制：v8 的无冲突 `ldmatrix` layout，以及 v6 的 `cp.async` 双缓冲。流水关系如下：

```text
stage 0: 计算当前 tile（ldmatrix + mma）
stage 1: 同时接收下一个 tile 的 cp.async
        ↓ wait_group 0 + __syncthreads
交换 read_stage/write_stage
```

首个 tile 仍然必须 `commit → wait → __syncthreads`，因为它没有前一轮 MMA 可以用来覆盖加载延迟。后续每轮才预取到另一个 buffer；当前 `read_stage` 正被 `ldmatrix` 读取时，绝不能写回同一个 stage。源码中的阶段注释见 [gemm_mma_cuda.cu](/home/eran/cuda-kernels/torch_ext/csrc/gemm_mma_cuda.cu:288)。

| Shape | v8 ld+pad | v9 async+ld+pad | v9 相对 v8 | cuBLAS |
|---:|---:|---:|---:|---:|
| 1024³ | 0.1209 ms | 0.1246 ms | -3.1% | 0.1779 ms |
| 2048³ | 0.9964 ms | 1.1121 ms | -11.6% | 1.0273 ms |
| 4096³ | 9.9003 ms | 10.1097 ms | -2.1% | 6.7062 ms |

这个负结果同样重要。v9 的 static shared memory 从 5.38 KB 增至 10.75 KB，但 registers/thread 仍为 48，理论 occupancy 仍被寄存器限制在 83.33%，所以 shared memory 翻倍没有直接造成 occupancy 下降。当前真正的问题是 `BK=16` 时每个 tile 的计算窗口很短：每轮只有固定数量的 MMA，`cp.async.commit`、`wait_group` 和 block barrier 的额外成本没有被完全隐藏。v9 的 NCU 仍确认 `LDSM`/`LDGSTS` 均存在，但最终性能结论以交错 CUDA Event 为准。

因此当前默认候选应保留 v8，v9 作为有实验价值的对照版本。要让异步流水真正获益，下一轮应扩大单次预取对应的计算量（例如更大的 `BK` 或每次加载后执行更多 MMA），同时重新检查寄存器压力，而不是继续无条件增加 pipeline stage。

### v10：增大 BK，让一次预取覆盖两个 K-slice

v10 将这个假设落实为独立文件 [gemm_mma_v10_cuda.cu](/home/eran/cuda-kernels/torch_ext/csrc/gemm_mma_v10_cuda.cu:1)。它把 `BK=16` 改为 `BK=32`，但 Tensor Core 的基本指令仍然是 `mma.sync.m16n8k16`，因此一个 `BK=32` tile 只是连续执行两个 K-slice：

```text
shared A/B tile: K = 32
    ├── ldmatrix(A[k:k+16]) + ldmatrix(B[k:k+16]) + mma.sync
    └── ldmatrix(A[k+16:k+32]) + ldmatrix(B[k+16:k+32]) + mma.sync
```

这一步的关键不是改变 MMA 的 fragment 形状，而是提高“每次 cp.async 预取后要做的计算量”。copy mapping 也必须重写：A 的 `64×32` tile 和 B 的 `32×64` tile 都是 4096B，256 个线程各搬 A、B 各一个 16B chunk。继续沿用 v9 的前 128 线程搬 A、后 128 线程搬 B 会少搬一半数据。

padding 仍然不能删除：A stride 为 `40 half = 80B = 20 banks`，B stride 为 `72 half = 144B = 36 banks ≡ 4 (mod 32)`。两个 stride 都是 16B 的倍数，既满足 `cp.async`/`ldmatrix` 的对齐，也保持无冲突 layout。双缓冲的静态 shared memory 为：

```text
2 × (64×40 + 32×72) × sizeof(half) = 19,456 B
```

#### v10 实测结果

同一脚本、同一输入、所有版本交错执行：

```bash
python bench_gemm_mma.py --sizes 4096 2048 1024 \
  --warmup 30 --iters 20 --rounds 9
```

| Shape | v8 ld+pad | v9 async+ld+pad | v10 BK32 | cuBLAS |
|---:|---:|---:|---:|---:|
| 1024³ | 0.1228 ms | 0.1267 ms | **0.1160 ms** | 0.1817 ms |
| 2048³ | **0.9690 ms** | 1.0870 ms | 1.2295 ms | 1.1177 ms |
| 4096³ | 9.7819 ms | 10.2161 ms | **9.6866 ms** | 6.7538 ms |

v10 在 4096³ 比 v8 快约 1.0%，在 1024³ 快约 5.5%，但在 2048³ 慢约 26.9%。因此目前不能把 v10 宣称为全面优于 v8；更准确的结论是：扩大 BK 确实让大矩阵的异步流水出现收益，但收益很小且对尺寸敏感。

NCU 资源数据（4096³）：

| 指标 | v8 | v10 |
|---|---:|---:|
| Registers/thread | 48 | 40 |
| Static shared memory/block | 5.38 KB | 19.46 KB |
| Theoretical occupancy | 83.33% | 83.33% |
| Achieved occupancy | 82.39% | 82.29% |
| Shared-memory block limit | 10 | 5 |

v10 的寄存器数反而下降，但 shared memory block limit 从 10 降到 5；occupancy 暂时没有进一步下降，是因为 v8/v10 都已经受到其他资源约束。2048³ 的回退不能仅凭 occupancy 解释，还需要进一步观察 memory replay、L2 命中、kernel launch 频率和指令吞吐。因此下一步应做针对性 NCU 指标采集，而不是继续盲目增大 BK。

### 稳定 GEMM 入口：shape-aware dispatch

实验 API 用于保留学习证据，用户侧则应只依赖 `gemm_mma_auto(a, b)`。host-only 路由实现在 [gemm_dispatch.cpp](/home/eran/cuda-kernels/torch_ext/csrc/gemm_dispatch.cpp:1)，采用三层结构：

```text
M<=256，M/N 为 64 倍数，K 为 32 倍数，指针 16B 对齐
    └── v10 BK32
其他合法 fp16 GEMM
    └── v8 ldmatrix+padding
         └── 非完整 tile / 非对齐 storage 再回退 v4
```

该规则来自 Transformer 形状的多组交错复测，而不是只看 cubic GEMM。`M=128/256, K=N=4096` 时 v10 的多组中位数通常与 v8 持平或略快；`M=512` 的结果会随频率状态反转，因此策略刻意停在 `M<=256`。这里不做首调用现场 autotune，因为 CUDA Event 会强制同步并增加冷启动延迟；未来若扩展到多 GPU，可以在 Python 层建立 `(device, M, N, K)` policy cache。

可用下面的脚本复现 projection/training 形状：

```bash
python bench_gemm_transformer.py --warmup 20 --iters 20 --rounds 7
```

一次 RTX 3060 Laptop 运行中，`[128,4096]@[4096,4096]` 的 auto/v8/v10 分别为 0.3109/0.3363/0.3158 ms；auto 与实际路由版本之间的微小差异来自交错顺序和 GPU 频率，不应解释成 dispatch 本身带来 kernel 加速。这个阶段的工程收益是统一 API、明确回退和真实模型形状验证。

### 2026-09-12 实机验证记录

本次在 RTX 3060 Laptop（sm_86，6GB）上重新编译并运行 `test_all.py`。PyTorch 为 `2.6.0+cu124`，CUDA Toolkit 为 `12.4`。WSL 的 `/usr/lib/wsl/lib` 已加入 `PATH`，`nvidia-smi` 与 PyTorch 均可识别 GPU：`torch.cuda.is_available() == True`、`device_count == 1`。

运行命令：

```bash
cd torch_ext
export TORCH_CUDA_ARCH_LIST="8.6"
~/venvs/torch/bin/python setup.py build_ext --inplace
~/venvs/torch/bin/python test_all.py
```

正确性结果：

```text
[RMSNorm] PASS  maxErr=9.537e-07
[Softmax] PASS  maxErr=7.451e-09
[LayerNorm] PASS  maxErr=9.537e-07
[GEMM] PASS  maxErr=2.136e-04
[GEMM MMA edge suite] PASS  maxErr=2.670e-05
[GEMM MMA misaligned-storage fallback] PASS
[GEMM MMA ldmatrix deterministic mapping] PASS
[FlashAttention full] PASS  maxErr=1.602e-07
[FlashAttention causal] PASS  maxErr=3.576e-07
[FlashAttention v4 full] PASS  maxErr=4.883e-04
[FlashAttention v4 causal] PASS  maxErr=9.766e-04
[FlashAttention v5 full] PASS  maxErr=4.883e-04
[FlashAttention v5 causal] PASS  maxErr=9.766e-04
[FlashAttention v6 D128 full] PASS  maxErr=4.883e-04
[FlashAttention v6 D128 causal] PASS  maxErr=2.441e-04
[GEMM MMA ldmatrix+padded non-default stream] PASS  maxErr=2.861e-06
[GEMM MMA async+ldmatrix+padded non-default stream] PASS  maxErr=2.861e-06
[GEMM MMA v10 BK32 non-default stream] PASS  maxErr=5.722e-06
[GEMM MMA auto non-default stream] PASS  maxErr=5.722e-06
[PyTorch CUDA contract suite] PASS
```

v4/v5 测试使用 `N=129`，专门覆盖不是 64 倍数的 K/V 尾块，同时验证 full 和 causal 两种模式。两版输出为 fp16，reference 使用 fp16 输入、fp32 计算后再转回 fp16，误差阈值为 `2e-2`。v4/v5 的误差逐项完全相同，说明 v5 只改变 P 的搬运路径，没有改变数值语义。

额外边界回归覆盖 `N={1,7,63,64,65,127,128,129,257}` 的 full/causal，结果为 `PASS maxErr=9.766e-04, v4-v5=0`。这些尺寸覆盖 64×64 tile 的边界前、边界上、边界后以及多个 KV tile。

批量多头回归使用 `[B,H,N,D]=[2,3,65,64]`，full/causal 对拍 `torch.nn.functional.scaled_dot_product_attention`，最大误差分别为 `2.441e-04` 和 `1.221e-04`。`N=65` 同时验证每个 head 的地址隔离和 K/V tail。

### 批量多头：二维 CUDA grid

v5 同时接受 `[N,64]` 和连续的 `[B,H,N,64]`。kernel 使用 `grid.x` 枚举 64 行 Q block，使用 `grid.y` 枚举展平后的 `batch×head`：

```cpp
const size_t sequence_offset =
    size_t(blockIdx.y) * size_t(N) * D;
Q += sequence_offset;
K += sequence_offset;
V += sequence_offset;
O += sequence_offset;
```

每个 `(batch,head)` 的 softmax 状态完全独立，但所有 head 在同一次 kernel launch 中进入 GPU。使用 `B=2,H=8,N=1024,D=64`，预热 20 次、每轮 50 次迭代并取交错 7 轮中位数：

| 模式 | auto | v5 BHD | Python 逐 head | PyTorch SDPA | auto/v5 | 逐 head/v5 |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.6583 ms | 0.6741 ms | 2.0552 ms | 0.1958 ms | 0.98x | **3.05x** |
| causal | 0.3958 ms | 0.4010 ms | 2.0308 ms | 0.1321 ms | 0.99x | **5.06x** |

`auto/v5≈1` 说明 head-dimension 调度不是当前瓶颈；二维 grid 消除了 16 次 Python/C++/CUDA launch 的串行提交开销，并把全部 head 的 Q block 一次性暴露给 GPU 调度器。v5 相对工业级 SDPA 仍慢约 3.0–3.4x，下一步应先 profile 内核，再决定优化 fragment load、K/V 搬运还是流水，而不是把该结果包装成“超过 PyTorch”。可用 `python bench_flashattention_bhd.py` 复现。

### v6：D=128 扩展实验

v5 的 D=64 不是把宏改成 128 就结束。D 同时影响三条路径：QK 的归约步数从 `64/16=4` 变成 `128/16=8`；Q/K/V shared-memory 行宽从 65 变成 129；PV 的输出 tile 数从 `64/8=8` 变成 `128/8=16`，于是每个线程的 `o_acc` 也翻倍。v6 保留 register-P 重排，只扩展这些维度。

正确性测试使用连续 `[B,H,N,D]=[2,2,65,128]`，full/causal 均与 SDPA 对齐。性能脚本为 [bench_flashattention_d128.py](bench_flashattention_d128.py)，同一输入、交错 5 轮中位数：

| N | 模式 | v6 D=128 | PyTorch SDPA | v6/SDPA |
|---:|---|---:|---:|---:|
| 1024 | full | 0.8180 ms | 0.2513 ms | 3.25x |
| 1024 | causal | 0.3210 ms | 0.1005 ms | 3.20x |
| 4096 | full | 7.3629 ms | 1.8643 ms | 3.95x |
| 4096 | causal | 3.5975 ms | 1.0083 ms | 3.57x |

ptxas/NCU 资源结果：

```bash
/usr/local/cuda/bin/ncu --set basic \
  --target-processes all \
  --kernel-name 'regex:flash_fp16_mma_v6' \
  --launch-count 1 \
  ~/venvs/torch/bin/python bench_flashattention_d128.py --n 1024 --warmup 1 --iters 1 --rounds 1
```

| 资源 | v6 full | v6 causal |
|---|---:|---:|
| Registers/thread | 128 | 163 |
| Static shared memory/block | 49.54 KB | 49.54 KB |
| Theoretical occupancy | 16.67% | 16.67% |
| Achieved occupancy（NCU full） | 15.98% | - |
| Spill stores/loads | 0 / 0 | 0 / 0 |

原因是 shared memory 已经限制每 SM 只能放 2 个 block；causal 的 163 registers/thread 又显著压缩了寄存器余量。v6 当前定位是“正确的 D=128 结构原型”，下一次优化应围绕减小 `o_acc` 生命周期、降低 PV 输出 tile 的寄存器占用，或重新设计 `Bq`，而不是盲目继续加 unroll。

### 长序列性能：v3 vs v4 vs v5

使用 [bench_flashattention.py](bench_flashattention.py) 进行 CUDA Event 计时。每组进行 20 次预热、50 次迭代、5 轮采样并取中位数；三个版本交错执行且每轮轮换顺序，以降低 GPU boost、温度和固定执行顺序的影响。v3→v5 是端到端升级；v4→v5 的输入、精度、tile 和数学运算相同，是只改变 P 数据通路的受控实验。

```bash
~/venvs/torch/bin/python bench_flashattention.py
```

| N | 模式 | v3 fp32 | v4 `Ps` | v5 register P | v4/v5 |
|---:|---|---:|---:|---:|---:|
| 1024 | full | 0.4543 ms | 0.1484 ms | 0.1329 ms | 1.12x |
| 1024 | causal | 0.1682 ms | 0.1120 ms | 0.0985 ms | 1.14x |
| 4096 | full | 5.1686 ms | 1.3561 ms | 1.0950 ms | 1.24x |
| 4096 | causal | 2.6260 ms | 0.7487 ms | 0.6908 ms | 1.08x |
| 8192 | full | 20.8013 ms | 4.2508 ms | 3.3810 ms | **1.26x** |
| 8192 | causal | 10.5417 ms | 2.4759 ms | 2.1193 ms | **1.17x** |

结论：v5 在所有测试形状上都快于 v4；收益随 full 长序列增大到 1.26x。它没有减少 HMMA 数量，而是删除每个 KV tile 对 `Ps` 的 shared-memory 写回/重载，并允许每个 SM 多驻留一个 block，用更多 active warp 隐藏 shared-memory/L1 延迟。

### Nsight Compute：v4 vs v5（N=8192, full）

为了避免 `N=1024` 只有 16 个 block、无法填满 30 个 SM 的问题，profile 固定使用 `N=8192`：

```bash
/usr/local/cuda/bin/ncu --set basic \
  --target-processes all \
  --kernel-name 'regex:flash_fp16_mma' \
  --launch-count 1 \
  ~/venvs/torch/bin/python bench_flashattention.py --n 8192 --mode full

/usr/local/cuda/bin/ncu --set basic \
  --target-processes all \
  --kernel-name 'regex:flash_fp16_mma_v5' \
  --launch-count 1 \
  ~/venvs/torch/bin/python bench_flashattention.py --n 8192 --mode full
```

两版均为 grid=`128`、block=`128`；profile 的多 pass `Duration` 含工具回放开销，性能结论使用上方独立 CUDA Event benchmark，NCU 在这里用于解释资源变化。

| 指标 | v4 `Ps` | v5 register P | 变化 |
|---|---:|---:|---:|
| Registers/thread | 101 | 99 | -2 |
| Static shared memory/block | 33.28 KB | 24.96 KB | **-8.32 KB** |
| Shared-memory block limit/SM | 2 | 3 | **+1 block** |
| Theoretical occupancy | 16.67% | 25.00% | **+8.33 pp** |
| Achieved occupancy | 15.55% | 21.28% | **+5.73 pp** |
| Memory throughput | 74.51% | 71.94% | -2.57 pp |
| L1/TEX throughput | 85.41% | 86.98% | +1.57 pp |
| Compute (SM) throughput | 31.57% | 32.96% | +1.39 pp |

`Ps[Bq][Bc+1]` 的尺寸正是 `64×65×2 = 8,320 B`，与 ptxas/NCU 的减少量完全吻合。v5 仍由 shared memory 限制 occupancy，但驻留能力从 2 block/SM 提升到 3 block/SM；实测 1.08x–1.26x 说明这个优化成立，同时也说明 occupancy 提升不等于性能线性提升——指令、同步、L1 和 Tensor Core 流水仍然存在。

### v5 原理：为什么可以删除 `Ps`

QK MMA 的一个 `m16n8k16` 输出 fragment 在每个 lane 中是 4 个 fp32：`c0/c1` 对应第 `g` 行的两个相邻列，`c2/c3` 对应第 `g+8` 行的两个相邻列。softmax 后，`S[t][0..3]` 仍保持这个布局。PV MMA 的 A 操作数需要一个 row-major `16×16` fragment；它恰好可由相邻两个 `16×8` 的 S fragment 拼成：

```cpp
const int st = b0 / WMMA_N;
pa0 = half2(S[st][0],     S[st][1]);      // 行 g，   列 0..7
pa1 = half2(S[st][2],     S[st][3]);      // 行 g+8， 列 0..7
pa2 = half2(S[st + 1][0], S[st + 1][1]);  // 行 g，   列 8..15
pa3 = half2(S[st + 1][2], S[st + 1][3]);  // 行 g+8， 列 8..15
```

v4 的路径是 `S(fp32 registers) → fp16 Ps(smem) → half2 A registers → MMA`；v5 变成 `S(fp32 registers) → half2 A registers → MMA`。两者都会做 fp32→fp16 转换，所以数值结果相同；v5 只消除了 shared-memory store/load 与 `Ps` 的容量。

### 为什么是这个结果

- **归一化类必须与融合后的原生算子比较**：RMSNorm 对比 `F.rms_norm` 为 3.61x；LayerNorm 对比 `F.layer_norm` 只有 1.16x。旧的手写 eager 表达式会启动多个 kernel，不能代表 PyTorch 原生 LayerNorm。
- **Softmax 0.97x 不丢人**：B=N=1024 时 torch.softmax 本身已是单个融合 kernel，打平合理；换非 2 的幂 N 或更大 batch，线程粗化版通常反超。
- **GEMM 0.12x 是诚实的差距展示**：v2 tiled 手写 vs cuBLAS 差 9 倍——cuBLAS 用 Tensor Core + 深度流水线。这正是路线 B（FP16 + mma.sync）的动机，也是"知道轮子多快"和"会造轮子"都要会的证据。
- **GEMM v8/v10/auto**：v8 通过 bank 公式推导 padding，将 LDSM 冲突清零；v10 单独重写 `BK=32` copy mapping，让每次预取后执行两个 K-slice；`gemm_mma_auto` 再把实验实现收敛成 v10/v8/v4 三层调度。性能数字受笔记本频率影响，因此 dispatch 使用保守小 M 策略，不宣称某个 tile 全尺寸最优。

## 文件结构

```
torch_ext/
├── csrc/
│   ├── bindings.cpp           # PyBind11 绑定：17 个手写算子/调度入口 + cuBLAS 基线
│   ├── ops.h                  # 入口函数声明
│   ├── rmsnorm_cuda.cu        # 各算子：CUDA kernel + torch::Tensor 包装
│   ├── softmax_cuda.cu
│   ├── layernorm_cuda.cu
│   ├── gemm_cuda.cu
│   ├── gemm_mma_cuda.cu       # GEMM v4-v8：mma/cp.async/ldmatrix/padding
│   ├── gemm_mma_v10_cuda.cu   # GEMM v10：BK=32 双 K-slice 流水
│   ├── gemm_dispatch.cpp       # 稳定 GEMM API：shape-aware v10/v8/v4
│   ├── flashattention_cuda.cu       # FlashAttention v3，fp32 baseline
│   ├── flashattention_mma_cuda.cu  # FlashAttention v4，fp16 Tensor Core + Ps
│   ├── flashattention_v5_cuda.cu   # FlashAttention v5，P fragment 寄存器直连
│   ├── flashattention_v6_cuda.cu   # FlashAttention v6，D=128 实验版
│   └── flashattention_dispatch.cpp # 稳定入口：按 head dimension 路由 v5/v6
├── setup.py                   # CUDAExtension 单模块构建
├── test_all.py                # 全量正确性、边界与 CUDA stream 契约测试
├── bench_ops.py               # 统一同语义 benchmark（CUDA Event/交错/中位数）
├── bench_gemm_mma.py          # GEMM v4-v10 + cuBLAS 受控版本链 benchmark
├── bench_gemm_transformer.py  # projection/training 真实形状 + auto dispatch
├── bench_avg.py               # 兼容旧入口，转到 bench_ops.py
├── bench_flashattention.py    # v3/v4/v5 交错、轮换顺序 benchmark
├── bench_flashattention_bhd.py # v5 BHD vs 逐 head dispatch vs PyTorch SDPA
└── bench_flashattention_d128.py # v6 D=128 vs PyTorch SDPA
```

## 关键点（面试常问）

1. **数据校验**：`TORCH_CHECK` 检查 device / dtype / contiguous / 维度，防御性编程
2. **contiguous**：`is_contiguous()` 保证内存连续，kernel 才能用 `x + row*N` 定位
3. **data_ptr<T>()**：拿到 tensor 裸指针传给 kernel
4. **单一模块 vs 多 .so**：一个 `ai_infra_ops` 暴露多类算子，避免每个算子一个共享库——工业界（如 FlashAttention 官方 repo、DeepSpeed op）都这么做
5. **运行时 bool → 编译期模板**：FlashAttention 的 causal 是运行时 bool，在包装层 `if/else` 派发到 `flash_fwd<true>` / `flash_fwd<false>` 两个编译期实例，消除 kernel 内每轮循环的运行时分支
6. **ABI 一致**：setup.py 编译时 torch 自动对齐 `_GLIBCXX_USE_CXX11_ABI`
7. **TORCH_CUDA_ARCH_LIST**：指定 sm_86，避免对所有架构编译浪费时间
