# 博客草稿：手撕 FlashAttention —— 从 occupancy 2% 到 79%，为什么反而更慢了？

> 适合发知乎/掘金。配图已生成在本目录：fig1_tuning_trilogy.png / fig2_bank_conflict.png / fig3_gemm_gflops.png
> 代码仓库：https://github.com/Eran-ovo/Ai-infra

## 提纲（每节给了要点和现成素材，你用自己的话扩写）

### 0. 开篇钩子（30 秒抓住读者）
- 一句话反差：「我把 FlashAttention 的 occupancy 从 2% 拉到 79%，速度却慢了 3 倍。」
- 放 fig1，直接上结论图
- 自我介绍一句：本科/研一，在 RTX 3060 上从零手撕 CUDA 算子的学习记录
- 【你来写：为什么想学 AI Infra，30 字即可，真诚最重要】

### 1. 背景：什么是 FlashAttention（给外行看懂，别堆公式）
- 一句话：Attention 的 N² 显存是瓶颈，FA 用"分块 + online softmax"让中间结果不进显存
- online softmax 的 running max / sum / acc 三个量（贴你 v1 的注释图或公式）
- 【你来写：画一张 S 矩阵分块的示意图（手画拍照都行）】

### 2. v1 → v2：causal mask 的"负优化"
- 写了 causal mask，N=512 时反而慢了（0.84x）
- ncu 数据：occupancy 2.08%，一线程一行只有 1 warp/block
- 讲清楚「延迟隐藏」：一个 warp 等显存时，没有别的 warp 顶上
- N 拉到 8192 后 causal 达成 2.0x 理论值（贴你的实测）
- 【你来写：用"工厂流水线"类比延迟隐藏，读者爱看这个】

### 3. v3 重构：一 warp 一行，occupancy 起飞但翻车
- FA2 的重划分思想：线程 → warp 管一行，寄存器 255→40/thread
- 踩坑 1：causal 分支发散下用 __shfl_xor → 死锁（讲为什么 shuffle 要全 warp 参与）
- 踩坑 2：q_row>=N 提前 return → __syncthreads 死锁（部分 warp 先走，留下的干等）
- 修复后 occupancy 79%，但…… 33ms，比 v2 的 9.7ms 还慢
- 【你来写：贴你死锁时的排查过程，printf 大法好，真实感加分】

### 4. 反转：bank conflict 才是真凶之一
- ncu 抓到 shared load 每条指令 7 次 bank conflict
- 讲清楚 32 路冲突：Ktile[c][d] 列访问，步长 64 是 32 的倍数
- 一个 padding [D]→[D+1]，冲突 940M→67M，提速 1.9x（放 fig2）
- 【你来写：画一张 bank 分布图（32 个 bank，列访问全撞 bank0）】

### 5. 为什么 occupancy 79% 还是比不过 16.7%？（全文核心，拔高）
- 三个因素叠加：K/V 重复搬运（block 数 ×4）+ Q 点积冗余 + shuffle 开销
- DRAM 实测只涨 1.5x（L2 缓存救场）——所以搬运不是唯一解释
- 核心观点：**occupancy、内存搬运、算术强度是三角债，优化是找平衡点不是拉满单项**
- FA2 真正的做法：增大 Br 摊销 K/V + Tensor Core 消除点积冗余（预告下篇）
- 【你来写：总结你踩坑的 timeline，真诚 > 完美】

### 6. 结尾 + 引流
- 一句话总结：性能优化没有银弹，只有测量、归因、再测量
- 附 GEMM 对比图（fig3）：手写 vs CUTLASS 差 8.6 倍，说明"知道轮子多快"和"会造轮子"都要会
- 仓库链接 + 求 star + 邮箱/联系方式
- 预告：下一篇《把 CUDA kernel 封装成 PyTorch Extension》

## 写作提醒
- 多用「我以为……结果……」的反转句式，技术博客就爱看踩坑
- 代码片段别贴整文件，贴关键的 5-10 行 + 逐行注释
- 每个结论后面必须跟 ncu 数据或实测耗时（你已经有全套了）
- 字数控制在 3000-4000，太长没人看
