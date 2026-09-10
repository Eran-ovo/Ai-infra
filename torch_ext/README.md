# torch_ext：把手写 CUDA kernel 封装成 PyTorch 算子

把 `kernels/` 下 5 类手写 CUDA kernel（GEMM / Softmax / LayerNorm / RMSNorm / FlashAttention）统一封装成一个 PyTorch Extension 模块 `ai_infra_ops`，当前暴露 8 个 Python 入口。

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
~/venvs/torch/bin/python setup.py build_ext --inplace
~/venvs/torch/bin/python test_all.py
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
ai_infra_ops.flashattention(q, k, v, causal)  # q/k/v [N,64] fp32
ai_infra_ops.flashattention_fp16(q, k, v, causal)  # v4：fp16 Tensor Core，P 经 Ps 中转
ai_infra_ops.flashattention_v5(q, k, v, causal)    # v5：[N,64] 或 [B,H,N,64]，P fragment 寄存器直连
```

## 实测结果（RTX 3060 Laptop，同会话交错计时，test_all.py 复现）

| 算子 | 手写 CUDA | PyTorch 原生 | 加速比 |
|------|----------|-------------|--------|
| RMSNorm (B=1024 N=1024) | 0.0391 ms | 0.1131 ms | 2.89x |
| Softmax (B=1024 N=1024) | 0.0305 ms | 0.0299 ms | 0.98x |
| LayerNorm (B=1024 N=1024) | 0.0372 ms | 0.1090 ms | 2.93x |
| GEMM (1024^3) | 2.7968 ms | 0.3458 ms (cuBLAS) | 0.12x |
| GEMM mma fp16 (1024^3) | 0.542 ms | 0.237 ms (cuBLAS fp16) | 0.44x |
| GEMM mma fp16 (2048^3) | 3.889 ms | 0.867 ms (cuBLAS fp16) | 0.22x |
| GEMM mma fp16 (4096^3) | 26.532 ms | 6.542 ms (cuBLAS fp16) | 0.25x |
| FlashAttention v3 (N=8192 D=64) | full 16.7522 / causal 8.8380 ms | - | causal 提速 1.90x |
| FlashAttention v4 (N=129 D=64, fp16) | full 0.0267 / causal 0.0611 ms | - | 短序列固定开销主导 |
| FlashAttention v5 (N=129 D=64, fp16) | full 0.0221 / causal 0.0198 ms | - | P fragment 寄存器直连 |

### 2026-09-10 实机验证记录

本次在 RTX 3060 Laptop（sm_86，6GB）上重新编译并运行 `test_all.py`。PyTorch 为 `2.6.0+cu124`，CUDA Toolkit 为 `12.4`。系统没有 `nvidia-smi`，但 PyTorch 成功识别 GPU：`torch.cuda.is_available() == True`、`device_count == 1`。

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
[FlashAttention full] PASS  maxErr=1.602e-07
[FlashAttention causal] PASS  maxErr=3.576e-07
[FlashAttention v4 full] PASS  maxErr=4.883e-04
[FlashAttention v4 causal] PASS  maxErr=9.766e-04
[FlashAttention v5 full] PASS  maxErr=4.883e-04
[FlashAttention v5 causal] PASS  maxErr=9.766e-04
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

每个 `(batch,head)` 的 softmax 状态完全独立，但所有 head 在同一次 kernel launch 中进入 GPU。使用 `B=2,H=8,N=1024,D=64`，交错 5 轮中位数结果：

| 模式 | v5 BHD 单次 launch | Python 逐 head dispatch | PyTorch SDPA | dispatch/BHD |
|---|---:|---:|---:|---:|
| full | 0.7630 ms | 2.3754 ms | 0.2243 ms | **3.11x** |
| causal | 0.4613 ms | 2.3670 ms | 0.1955 ms | **5.13x** |

二维 grid 消除了 16 次 Python/C++/CUDA launch 的串行提交开销，并把全部 head 的 Q block 一次性暴露给 GPU 调度器。与工业级 SDPA 仍有约 2.4–3.4x 差距，后续方向是支持更多 head dimension、减少手写 fragment load 指令并研究异步流水，而不是把该结果包装成“超过 PyTorch”。可用 `python bench_flashattention_bhd.py` 复现。

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

- **归一化类（RMSNorm/LayerNorm）」融合是最大卖点**：PyTorch 原生把它拆成 pow→mean→rsqrt→mul 多个 kernel，中间结果反复写回显存；fused 一次加载一次写回，3-5x。
- **Softmax 0.97x 不丢人**：B=N=1024 时 torch.softmax 本身已是单个融合 kernel，打平合理；换非 2 的幂 N 或更大 batch，线程粗化版通常反超。
- **GEMM 0.12x 是诚实的差距展示**：v2 tiled 手写 vs cuBLAS 差 9 倍——cuBLAS 用 Tensor Core + 深度流水线。这正是路线 B（FP16 + mma.sync）的动机，也是"知道轮子多快"和"会造轮子"都要会的证据。
- **gemm_mma 0.22x-0.44x 是路线 B 的第一步**：手写 `mma.sync.m16n8k16` 后，1024³ 从 v2 fp32 的 ~700 GFLOPS 跃升到 3963 GFLOPS（~6x），大尺寸到 5.2 TFLOPS；与 cuBLAS fp16 差距从 ~13x 缩到 2-4x。剩余差距来自无 cp.async 双缓冲 / ldmatrix / 大 tile，是"追平 cuBLAS"的后续迭代点。注：cuBLAS 对比走 `torch.matmul` 的 fp16 累加路径（比 fp32 累加更快），手写版是 fp32 累加，严格同精度对比见 kernels/gemm_v4_mma.cu 的 cublasGemmEx。

## 文件结构

```
torch_ext/
├── csrc/
│   ├── bindings.cpp           # PyBind11 绑定：统一暴露 8 个 forward
│   ├── ops.h                  # 入口函数声明
│   ├── rmsnorm_cuda.cu        # 各算子：CUDA kernel + torch::Tensor 包装
│   ├── softmax_cuda.cu
│   ├── layernorm_cuda.cu
│   ├── gemm_cuda.cu
│   ├── flashattention_cuda.cu       # FlashAttention v3，fp32 baseline
│   ├── flashattention_mma_cuda.cu  # FlashAttention v4，fp16 Tensor Core + Ps
│   └── flashattention_v5_cuda.cu   # FlashAttention v5，P fragment 寄存器直连
├── setup.py                   # CUDAExtension 单模块构建
├── test_all.py                # 全量正确性对拍 + 基础 benchmark
├── bench_flashattention.py    # v3/v4/v5 交错、轮换顺序 benchmark
└── bench_flashattention_bhd.py # v5 BHD vs 逐 head dispatch vs PyTorch SDPA
```

## 关键点（面试常问）

1. **数据校验**：`TORCH_CHECK` 检查 device / dtype / contiguous / 维度，防御性编程
2. **contiguous**：`is_contiguous()` 保证内存连续，kernel 才能用 `x + row*N` 定位
3. **data_ptr<T>()**：拿到 tensor 裸指针传给 kernel
4. **单一模块 vs 多 .so**：一个 `ai_infra_ops` 暴露多类算子，避免每个算子一个共享库——工业界（如 FlashAttention 官方 repo、DeepSpeed op）都这么做
5. **运行时 bool → 编译期模板**：FlashAttention 的 causal 是运行时 bool，在包装层 `if/else` 派发到 `flash_fwd<true>` / `flash_fwd<false>` 两个编译期实例，消除 kernel 内每轮循环的运行时分支
6. **ABI 一致**：setup.py 编译时 torch 自动对齐 `_GLIBCXX_USE_CXX11_ABI`
7. **TORCH_CUDA_ARCH_LIST**：指定 sm_86，避免对所有架构编译浪费时间
