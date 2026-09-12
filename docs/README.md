# docs — 算子原理文档

把 `kernels/` 下的算子按**底层优化主线**归类，每类一个 HTML，讲解「原理」和「版本间区别」。浏览器直接打开 `index.html` 即可浏览，无需构建。

## 目录结构

```
docs/
├── index.html            # 总览：三类算子 + 封装节课导航 + 共同地基（分块/合并访存/bank conflict/归约/occupancy）
├── gemm.html             # GEMM：v0-v10、Tensor Core 数据通路与 auto dispatch
├── ldmatrix.html         # 交互专题：v7/v8 fragment 映射、LDSM bank conflict 与 padding
├── normalization.html    # 归约/归一化类：Softmax / LayerNorm / RMSNorm
├── flashattention.html   # FlashAttention：v1-v7、D64/D128 与稳定 dispatch
├── torch_ext.html        # PyTorch 封装：ai_infra_ops、包装层、绑定、校验与测试
├── assets/
│   └── style.css         # 共享样式（所有分类页共用同一份）
└── README.md             # 本文件
```

## 归类口径

| 类 | 共享主线 | 算子 |
|----|---------|------|
| GEMM | tiling 复用 + 访存优化 + Tensor Core | v0-v10 + shape-aware auto dispatch |
| 归约/归一化 | 块内树形归约 + warp shuffle，fusion 一次写回 | softmax / layernorm / rmsnorm |
| FlashAttention | 分块 + online softmax（S 不落 HBM） | v1-v7 + D64/D128 auto dispatch |
| PyTorch 封装 | 手写 kernel → torch 算子（包装层 + 绑定 + 校验） | 18 个手写入口 + 1 个 cuBLAS 基线 |

## 如何扩展

### ① 新增一个算子版本（最常见）

同一类里加新版本（例：`gemm_v5.cu`）：在对应分类页里加一个 `<div class="op">` 块，拷贝现有版本块改标题/文件名/性能/要点即可。

```html
<div class="op v-highlight">         <!-- 里程碑版本加 v-highlight 高亮 -->
  <div class="op-head">
    <span class="op-title">v5 · ...</span>
    <span class="op-file">kernels/gemm_v5.cu</span>
    <span class="op-perf">xxx GFLOPS @...</span>
  </div>
  <p><strong>核心思想</strong>：...</p>
  <ul><li>...</li></ul>
</div>
```

同时更新该页底部「区别总览」表格加一行。

### ② 新增一个分类（如未来做「通讯/Reduce」「Scan 前缀和」「卷积 im2col」）

1. 新建 `xxx.html`，`<head>` 里引同一份样式：
   ```html
   <link rel="stylesheet" href="assets/style.css">
   ```
2. `<header>` 顶部导航里补一个 `<a href="xxx.html">分类名</a>`（当前页加 `class="active"`）。
3. `index.html` 的网格里加一张 `<div class="card">`，补齐导航链接。
4. 若分类之间有先后，调整每个页面 footer 的「上一类/下一类」链接。

### ③ 换主题

颜色变量全部集中在 `assets/style.css` 顶部的 `:root {}`（`--bg` / `--accent` / 等），改一处全局生效。各 HTML 不内联样式。

## 注意

- 讲「原理」以源码注释和仓库 README 调优记录为准；简历性能数字只以
  `torch_ext/benchmark_results/final_benchmark_sm86.md` 为准。
- 新建算子后，文档里的 GFLOPS / maxErr 等数字需要实测更新，别凭空写。
