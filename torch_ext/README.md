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
| RMSNorm (B=1024 N=1024) | 0.032 ms | 0.121 ms | 3.7x |
| Softmax (B=1024 N=1024) | 0.031 ms | 0.030 ms | 0.97x |
| LayerNorm (B=1024 N=1024) | 0.030 ms | 0.108 ms | 3.6x |
| GEMM (1024^3) | 2.90 ms | 0.36 ms (cuBLAS) | 0.12x |
| GEMM mma fp16 (1024^3) | 0.542 ms | 0.237 ms (cuBLAS fp16) | 0.44x |
| GEMM mma fp16 (2048^3) | 3.889 ms | 0.867 ms (cuBLAS fp16) | 0.22x |
| GEMM mma fp16 (4096^3) | 26.532 ms | 6.542 ms (cuBLAS fp16) | 0.25x |
| FlashAttention (N=8192 D=64) | causal 9.2 ms | - | causal 提速 1.9x |

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
└── test_all.py                # 正确性对拍 + 性能 benchmark
```

## 关键点（面试常问）

1. **数据校验**：`TORCH_CHECK` 检查 device / dtype / contiguous / 维度，防御性编程
2. **contiguous**：`is_contiguous()` 保证内存连续，kernel 才能用 `x + row*N` 定位
3. **data_ptr<T>()**：拿到 tensor 裸指针传给 kernel
4. **单一模块 vs 多 .so**：一个 `ai_infra_ops` 暴露多类算子，避免每个算子一个共享库——工业界（如 FlashAttention 官方 repo、DeepSpeed op）都这么做
5. **运行时 bool → 编译期模板**：FlashAttention 的 causal 是运行时 bool，在包装层 `if/else` 派发到 `flash_fwd<true>` / `flash_fwd<false>` 两个编译期实例，消除 kernel 内每轮循环的运行时分支
6. **ABI 一致**：setup.py 编译时 torch 自动对齐 `_GLIBCXX_USE_CXX11_ABI`
7. **TORCH_CUDA_ARCH_LIST**：指定 sm_86，避免对所有架构编译浪费时间
