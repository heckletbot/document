# 分层观察与 Bound 诊断

基线和路径有了之后，用 Profiler 和日志做根因，不要直接猜「再加大 batch」。

建议顺序：**框架 → 调度 / 通信 → 数据通路 → 算子**。上层能解释的，先不要下到 Kernel。引擎日志和 benchmark 的切片方法见 [log-analyze](../log-analyze/README.md)；算子 msprof 仍走 [fusion](../fusion/README.md)。

---

## 四层分别看什么

| 层级 | 看什么 | 典型问题 |
|------|--------|----------|
| 框架 | 队列、P/D 比例、Continuous Batching、Prefix/KV 命中、batch 参数 | 长短请求混跑抖动；batch 组不满；缓存该命中没命中 |
| 调度 | Device Idle Gap、Launch Gap、算子排队、通信计算 overlap | NPU/GPU 空转；Host 下发跟不上 |
| 数据通路 | 计算 / 访存 / 通信是否重叠；Host 准备是否拖后腿；KV 访问 | Pipeline 阻塞；长上下文把带宽打满 |
| 算子 | 耗时占比、效率、等待；MatMul / Attention / Softmax；Reshape、Transpose、Cast | 热点 Kernel；碎片化 Layout 转换 |

---

## 五种 Bound（用来决定优化方向）

| 类型 | 你看到的现象 | 优先动什么 |
|------|----------------|------------|
| Compute Bound | 计算单元利用率低，或打满但仍不够 FLOPS | Kernel、融合、精度、算法（如 MLA / 量化） |
| Memory Bound | HBM/显存带宽饱和、Cache Miss、KV 访问突出 | Layout、KV block、前缀缓存、减少中间 Tensor |
| Scheduling Bound | Device Idle、Launch Gap、Host 同步 | 异步下发、减少小 Kernel、Runtime 调度 |
| Communication Bound | AllReduce / AllGather / All-to-All 占比高 | 改 TP/EP、切分、通信计算 overlap |
| Framework Bound | batch 效率低、队列空或死锁式堆积、KV miss | 调度参数、PD 切分、缓存策略 |

一张表够用来开口：「我判断是 X Bound，因为看到 Y，所以下一步试 Z。」

---

## 怎么从数推到 Bound（口头逻辑）

1. 利用率低、队列却很长 → 先查 Framework / Scheduling，不是算子算得慢。
2. 利用率低、队列也空 → 流量不够，或调度组不出 batch。
3. 利用率高、带宽也高、TPOT 差 → Memory Bound，盯 KV 和 Layout。
4. 单卡利用率尚可、扩 TP 更慢 → Communication Bound。
5. TOP 算子里 Reshape/Transpose/Cast 很多 → 图和融合，不是再堆 batch。

---

## 诊断之后先写假设再改

每轮只保留一条可证伪的话，例如：

- 「TTFT 高是排队，不是 Prefill Kernel」→ 降并发或改调度后再看 TTFT 分解。
- 「Decode 是 KV 带宽」→ 改 KV block / 量化 KV，看带宽和 TPOT。
- 「TP=8 时通信占比过高」→ 试更小 TP 或更好 overlap，看通信时间。

下一篇：[优化杠杆](05-levers.md)。
