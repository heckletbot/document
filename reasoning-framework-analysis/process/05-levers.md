# 优化杠杆：判断之后动哪一层

方向来自 Bound，手段分四类。**先参数、后框架路径、再图和 Kernel、最后动并行拓扑**——越往下成本越高、回退越贵。

---

## 1. 参数（Serving 配置）

覆盖调度、batch、缓存、并行。改完必须重新压同一条负载。

| 类 | 常见旋钮 | 主要影响 |
|----|----------|----------|
| 动态批处理 | batch size、max_num_batched_tokens、max_num_seqs、PD 比例 | 吞吐 vs TTFT；组 batch 效率 |
| 显存与 KV | memory utilization、KV block 大小/数量、Prefix Cache | 能吞多少并发、碎片、命中率 |
| 并行 | TP / PP / DP / EP | 单卡计算、通信体积、专家均衡 |

多目标时写清主指标：Throughput、TTFT、TPOT、利用率、显存。不要四个一起「都要最好」。

---

## 2. 框架执行路径

参数改不动，再下到 Scheduler、Executor、Worker、Attention backend：

- Scheduler 里多余同步、锁、串行等待
- Continuous Batching 没有把 Decode 空档填上
- Prefill / Decode 流水线上的空洞（Host 等 Device，或 P 等 D）
- Host 与加速器之间多余的拷贝和同步

PD 分离时还要看 KV 传输是否挡在 Decode 开工之前。对照 Mooncake / Connector 笔记看「KV 怎么走」。

---

## 3. 计算图与算子

框架路径合理，热点仍在 Kernel：

- 融合连续计算，减少中间 Tensor 和 Launch
- 消掉成串的 Reshape / Transpose / Cast，合并 Format Conversion
- 把通用实现换成该硬件上的高性能 Kernel
- Attention、MoE 路由、采样等专用融合

融合会改图，可能改变 Runtime 调度，所以融合后要重新看 Idle Gap，不能只看单算子微秒数。

Ascend 上的分层、模式目录、接入路径和验收指标见 [fusion/README.md](../fusion/README.md)。

---

## 4. 并行与通信

确认是 Communication Bound 再动拓扑：

- 过大的 TP 换来更多 AllReduce
- MoE 的 EP 与 all-to-all 是否成为新墙
- 通信能否与计算 overlap
- PD 的 KV 传输路径是否走对拓扑

---

## 动手时的纪律

1. 一次一类杠杆，保留对照实验。
2. 先用同一 Workload Profile 回归：Throughput、TTFT、TPOT、成功率。
3. 变好就记下配置和 Bound 判断；变差或持平就回退，避免配置堆积。
4. 精度若在范围内，任何 Kernel / 量化改动都要带精度点测。

下一篇：[口头讲稿](06-speak.md)。
