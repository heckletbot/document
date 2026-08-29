# 口头讲稿：五分钟说完分析顺序

下面按「别人问你：新模型怎么看」来组织。先把这段说顺，再往各篇补例子。

---

## 30 秒

推理优化不是调一个 Kernel。我拿到新模型会固定走：定场景 → 跑基线 → 沿请求路径分层看 → 判断 Bound → 只动一类杠杆 → 用同一负载回归。

---

## 2 分钟

场景先问四件事：模型结构（Dense/MoE、精度、上下文）、硬件和框架版本、流量和 SLO、现在的 PD 与并行切法。

基线至少有吞吐、TTFT、TPOT、显存和计算利用率。然后沿路径拆：排队、Prefill、Decode、通信。TTFT 高先看是队列还是 Prefill；TPOT 差再看 Decode 和 KV。

Profiler 从上往下：框架有没有组成有效 batch、设备有没有 Idle、是带宽满了还是通信挡着、最后才是 TOP 算子。用 Compute / Memory / Scheduling / Communication / Framework 五种 Bound 选方向。

杠杆从便宜到贵：Serving 参数 → 调度和执行路径 → 图融合与 Kernel → 改并行。一次验证一个假设。

---

## 5 分钟（按层次展开）

1. **为什么系统看**：batch、TP、融合会互相改 Shape、显存和通信，单点经验换模型就失效。
2. **观察清单**：模型 / 硬件软件 / 负载 SLO / Serving 切法，见 [02-observe.md](02-observe.md)。
3. **路径**：Client → Router → Scheduler → Prefill/Decode → 采样；PD 多一跳 KV。见 [03-request-path.md](03-request-path.md)。
4. **诊断**：四层 Profiling 对五种 Bound。见 [04-diagnose.md](04-diagnose.md)。
5. **动手**：参数、框架、融合、并行。见 [05-levers.md](05-levers.md)。

---

## 自测（说不出来就回去补）

1. 为什么 TTFT 高不一定是 Prefill Kernel？
2. 利用率低时，怎样区分「没流量」「组不出 batch」「Launch 太碎」？
3. 什么现象让你优先改 KV / Prefix Cache，而不是加大 TP？
4. 为什么融合之后还要再看 Device Idle？
5. 给你一个 MoE 长上下文、时延 SLO，你第一轮压测和第一轮 Profiler 看哪些图？

答完再往对应文档里填自己的案例（不要写客户名和未公开指标）。
