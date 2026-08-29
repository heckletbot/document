# 口头讲稿：日志怎么讲到改点

和融合讲稿分开。别人问「decode 有气泡 / TTFT 太高」，先走这里。

---

## 30 秒

先定方向再搜日志：气泡看 idle/gap，TTFT 看 waiting 和 prefill，吞吐看 batch 和 preempt。对上 Scheduler、KV 或 Host 预处理之后，说出文件和改法，确认了再改。不要一上来融算子。

---

## 2 分钟

三类输入：profiling 时间分布、引擎 KV/scheduler 行、benchmark 分位。TTFT 高先拆：排队、下图解码、还是 Prefill 前向。多模态高并发下，线程池过小或串行解析会把 TTFT 打满，NPU 利用率低是结果不是根因。改完同一负载回归；TTFT 降、吞吐抖动几个点，脚本仍可能判有收益（任一指标过 5% 即 OR）。吞吐优先时要自己读 comparison，不要只看退出码。

---

## 自测

1. 为什么 decode 气泡的关键字不能拿去解释所有 TTFT 问题？  
2. `preempted` 高时，你先打开 Scheduler 还是 Model Runner？  
3. 多模态 TTFT 高、但 decode tok/s 正常，Bound 该怎么说？  
4. 为什么只改 Prefill 补丁时可以不重启 Decode？  
5. 和融合笔记如何分工：什么时候从本目录转到 `fusion/`？  
6. 为什么必须 warm up？冷启动会脏掉哪类指标？  
7. `has_improvement` 是 AND 还是 OR？TTFT 大降、QPS 微降时退出码是什么、你怎么向别人解释？

说不出来就回 [01](01-metrics-keywords.md) / [02](02-code-map.md)。
