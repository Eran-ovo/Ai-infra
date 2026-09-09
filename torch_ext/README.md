# torch_ext：把手写 CUDA kernel 封装成 PyTorch 算子

把 `kernels/` 下 5 类手写 CUDA kernel（GEMM / Softmax / LayerNorm / RMSNorm / FlashAttention）统一封装成一个 PyTorch Extension 模块 `ai_infra_ops`，当前暴露 7 个 Python 入口。

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
export PATH=/usr/local/cuda/bin:$PATH
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
ai_infra_ops.flashattention_fp16(q, k, v, causal)  # q/k/v [N,64] fp16，v4 Tensor Core
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
| FlashAttention v4 (N=129 D=64, fp16) | full 0.0253 / causal 0.0238 ms | - | causal 提速 1.06x |

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
```

v4 测试使用 `N=129`，专门覆盖不是 64 倍数的 K/V 尾块，同时验证 full 和 causal 两种模式。v4 输出为 fp16，reference 使用 fp16 输入、fp32 计算后再转回 fp16，误差阈值为 `2e-2`。

### 长序列性能：v3 vs v4

使用 [bench_flashattention.py](bench_flashattention.py) 进行 CUDA Event 计时。v3 使用 fp32，v4 使用 fp16 Tensor Core；因此这是“数据类型 + Tensor Core + kernel 组织”的端到端对比，不是只改变一个变量的微基准。

```bash
~/venvs/torch/bin/python bench_flashattention.py
```

| N | 模式 | v3 fp32 | v4 fp16 | v3/v4 |
|---:|---|---:|---:|---:|
| 1024 | full | 0.3917 ms | 0.1477 ms | 2.65x |
| 1024 | causal | 0.2164 ms | 0.1874 ms | 1.15x |
| 4096 | full | 4.6235 ms | 1.1499 ms | 4.02x |
| 4096 | causal | 2.2017 ms | 0.6787 ms | 3.24x |
| 8192 | full | 17.3622 ms | 3.5999 ms | 4.82x |
| 8192 | causal | 8.7591 ms | 2.0900 ms | 4.19x |

结论：短序列时 kernel launch 和固定 tile 开销占比高，causal 加速不明显；序列长度增大后，v4 的 Tensor Core 路径优势显现，`N=8192` 时 full/causal 分别达到 4.82x/4.19x。

### Nsight Compute baseline（N=8192, full）

为了避免 `N=1024` 只有 16 个 block、无法填满 30 个 SM 的问题，profile 固定使用 `N=8192`：

```bash
/usr/local/cuda/bin/ncu --set basic \
  --target-processes all \
  --kernel-name 'regex:flash_fp16_mma' \
  --launch-count 1 \
  ~/venvs/torch/bin/python bench_flashattention.py --n 8192 --mode full
```

profile 到的 kernel 是 `flash_fp16_mma<false>`，grid=`128`、block=`128`：

| 指标 | 结果 |
|---|---:|
| Registers/thread | 101 |
| Static shared memory/block | 33.28 KB |
| Theoretical occupancy | 16.67% |
| Achieved occupancy | 15.50% |
| Memory throughput | 73.93% |
| L1/TEX throughput | 87.09% |
| Compute (SM) throughput | 31.32% |
| HMMA warp instructions | 4,194,304 |
| FP16→FP32 Tensor path ops | 17,179,869,184 |

Nsight Compute 确认 v4 确实执行了 HMMA Tensor Core 指令。当前主要限制不是 Tensor Core 没有工作，而是 shared memory 导致 occupancy 只有约 16.7%，同时 L1/TEX 利用率高于计算利用率；下一轮优化应优先研究 K/V tile 复用、shared-memory pipeline 和 `cp.async`/`ldmatrix`，而不是继续增加数学计算量。

### 为什么是这个结果

- **归一化类（RMSNorm/LayerNorm）」融合是最大卖点**：PyTorch 原生把它拆成 pow→mean→rsqrt→mul 多个 kernel，中间结果反复写回显存；fused 一次加载一次写回，3-5x。
- **Softmax 0.97x 不丢人**：B=N=1024 时 torch.softmax 本身已是单个融合 kernel，打平合理；换非 2 的幂 N 或更大 batch，线程粗化版通常反超。
- **GEMM 0.12x 是诚实的差距展示**：v2 tiled 手写 vs cuBLAS 差 9 倍——cuBLAS 用 Tensor Core + 深度流水线。这正是路线 B（FP16 + mma.sync）的动机，也是"知道轮子多快"和"会造轮子"都要会的证据。
- **gemm_mma 0.22x-0.44x 是路线 B 的第一步**：手写 `mma.sync.m16n8k16` 后，1024³ 从 v2 fp32 的 ~700 GFLOPS 跃升到 3963 GFLOPS（~6x），大尺寸到 5.2 TFLOPS；与 cuBLAS fp16 差距从 ~13x 缩到 2-4x。剩余差距来自无 cp.async 双缓冲 / ldmatrix / 大 tile，是"追平 cuBLAS"的后续迭代点。注：cuBLAS 对比走 `torch.matmul` 的 fp16 累加路径（比 fp32 累加更快），手写版是 fp32 累加，严格同精度对比见 kernels/gemm_v4_mma.cu 的 cublasGemmEx。

## 文件结构

```
torch_ext/
├── csrc/
│   ├── bindings.cpp           # PyBind11 绑定：统一暴露 7 个 forward
│   ├── ops.h                  # 入口函数声明
│   ├── rmsnorm_cuda.cu        # 各算子：CUDA kernel + torch::Tensor 包装
│   ├── softmax_cuda.cu
│   ├── layernorm_cuda.cu
│   ├── gemm_cuda.cu
│   ├── flashattention_cuda.cu       # FlashAttention v3，fp32 baseline
│   └── flashattention_mma_cuda.cu  # FlashAttention v4，fp16 Tensor Core
├── setup.py                   # CUDAExtension 单模块构建
├── test_all.py                # 全量正确性对拍 + 基础 benchmark
└── bench_flashattention.py    # v3/v4 长序列 CUDA Event benchmark
```

## 关键点（面试常问）

1. **数据校验**：`TORCH_CHECK` 检查 device / dtype / contiguous / 维度，防御性编程
2. **contiguous**：`is_contiguous()` 保证内存连续，kernel 才能用 `x + row*N` 定位
3. **data_ptr<T>()**：拿到 tensor 裸指针传给 kernel
4. **单一模块 vs 多 .so**：一个 `ai_infra_ops` 暴露多类算子，避免每个算子一个共享库——工业界（如 FlashAttention 官方 repo、DeepSpeed op）都这么做
5. **运行时 bool → 编译期模板**：FlashAttention 的 causal 是运行时 bool，在包装层 `if/else` 派发到 `flash_fwd<true>` / `flash_fwd<false>` 两个编译期实例，消除 kernel 内每轮循环的运行时分支
6. **ABI 一致**：setup.py 编译时 torch 自动对齐 `_GLIBCXX_USE_CXX11_ABI`
7. **TORCH_CUDA_ARCH_LIST**：指定 sm_86，避免对所有架构编译浪费时间
