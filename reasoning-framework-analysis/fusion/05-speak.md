# 口头讲稿：新模型要不要做融合

接在系统分析讲稿之后：已经判了 Bound，别人问「那你融什么」。

---

## 30 秒

融合只解决三件事：同一数据多次扫 GM、小 kernel launch 太碎、通信和计算串行。Cube 已经打满就不要融。Ascend 上分内核（Vector 或 CV）、图 pass、通算三层，不要混成一个词。

---

## 2 分钟

先看模型 trace：hidden 上的 add+rms 是否写了中间量；是否一串 &lt;50µs kernel；AllReduce 是否挡着下一跳 RMSNorm。对上 Residual+RMSNorm、SwiGLU、Norm+Quant、MoE TopK、Matmul+激活、通算这几类。

实现上 SIMD 走 CopyIn/Compute/CopyOut；Matmul 后激活走 `__mix__`，C 块必须装进 UB。接到框架先 Python 调通，再 `forward_oot`，跨层序列再上图 pass。msprof 上 GM 要降、UB 要升、流水要重叠，否则只是换了个 op 名。

换 910C 要改口：没有 L1→GM、没有 GM→L0 直连，L0A 用 NZ，Vector 用 Regbase，UB bank 结构变了要重算冲突。

---

## 5 分钟展开

1. 何时融 / 不融，三层分别解决什么。见 [README](README.md)。  
2. 按 Bound 点名 2～3 个模式和 vllm-ascend 入口。见 [01-patterns.md](01-patterns.md)。  
3. tiling 硬约束和 CV 数据流。见 [02-implement.md](02-implement.md)。  
4. A/B/C 接入。见 [03-integrate.md](03-integrate.md)。  
5. 用哪几张 csv 验收；2201→3510 清单。见 [04-profile-arch.md](04-profile-arch.md)。

---

## 自测

1. Residual+RMSNorm 为什么是 Memory Bound 而不是 Compute Bound？  
2. 为什么 AllReduce+RMSNorm 要 `compile_range.start > 512` 才开？  
3. `GetTensorC` 进 UB 装不下时，融合看起来接上了，msprof 会怎样？  
4. 路径 B 和路径 C 各解决什么？RMSNorm+Quant 为什么通常走 C？  
5. 通算和 CV 融合的数据路径有何不同？  
6. 3510 上继续写 L1→GM 会怎样？bank 结构变了为什么要重做冲突分析？  
7. 仓库里六类图 pass 各管什么？为什么 AllReduce+RMSNorm 和 QK-RoPE 的门控条件不同？  
8. 新增算子 1–8 步和 9–10 步，缺哪一段会「kernel 有、图上没有」？

说不出来就回到对应篇，补自己跑过的 trace 截图（不要写未公开指标）。
