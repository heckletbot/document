# 集合通信算子

并行切分最终都落成 Rank 之间传数据。先分清点对点（一对一）和集合通信（一对多 / 多对一 / 多对多），后面看 TP / PP / EP 才不会把算子记混。

## 总表

| 通信分类 | 操作 | 含义 | 用途 |
|---------|------|------|------|
| 一对一 | Send | 一个 Rank 主动向指定 Rank 发送数据 | PP 中发送激活值、P/D 分离中发送 KV Cache |
| 一对一 | Recv | 一个 Rank 从指定 Rank 接收数据 | PP 中接收激活值、Decode 节点接收 KV Cache |
| 一对多 | Broadcast | Root 将同一份完整数据发送给所有 Rank | 同步模型参数、配置或随机种子 |
| 一对多 | Scatter | Root 将数据切成多个分片，分别发送给不同 Rank | 分发输入、参数或计算任务 |
| 多对一 | Gather | 多个 Rank 将数据发送给 Root，由 Root 拼接完整数据 | 汇总不同 Rank 的分片输出 |
| 多对一 | Reduce | 多个 Rank 的数据经过求和等运算，结果只保存在 Root | 汇总梯度、损失或统计指标 |
| 多对多 | AllGather | 收集所有 Rank 的分片，使每个 Rank 都得到完整数据 | FSDP 参数恢复、TP 数据拼接 |
| 多对多 | AllReduce | 聚合所有 Rank 的数据，使每个 Rank 都得到完整聚合结果 | DDP 梯度同步、TP 结果求和 |
| 多对多 | ReduceScatter | 先聚合所有 Rank 的数据，再将结果切分给不同 Rank | FSDP、ZeRO 的梯度聚合与分片；推理里 SP / EP combine |
| 多对多 | AllToAll | 每个 Rank 将数据切分后，与所有其他 Rank 交换 | MoE/EP 中分发 Token 并回传 Expert 结果 |

## 容易混的几组

**Reduce 不是 Gather。** Reduce 要先收其他 Rank 的数据，再做求和 / max 等规约，Root 上只留一份结果。Gather 是拼接，不做运算。

**Scatter 不是 Broadcast。** Scatter 把一份数据切成互不相同的分片再发给各 Rank；Broadcast 每张卡拿到的是同一份完整数据。

**ReduceScatter 不是「先 Reduce 再 Scatter」两次独立通信。** 实际实现里各 Rank 边交换分片、边执行规约，一次走完。通信量约为 AllReduce 的 `1/N`。

**AllReduce 也是边传边规约。** 常用 Ring 或 Tree：Ring AllReduce 等价于一次 ReduceScatter 再接一次 AllGather。每个 Rank 最终都拿到完整聚合结果。

```text
AllReduce  ≈  ReduceScatter  +  AllGather

Rank0: a0          Rank0: sum(a*)[0]     Rank0: sum(a*) 全量
Rank1: a1    RS →  Rank1: sum(a*)[1]  AG → Rank1: sum(a*) 全量
Rank2: a2          Rank2: sum(a*)[2]     Rank2: sum(a*) 全量
```

**AllGather 是 Gather 的「每人一份」。** Gather 只有 Root 有完整拼接；AllGather 每个 Rank 都有。

**AllToAll 是「每人给每人」。** 不是把同一份数据广播出去。Rank i 把自己的数据切成 N 份，第 j 份发给 Rank j；同时从所有 Rank 收下属于自己的那一份。MoE 的 token 按 expert 归属换卡，走的就是它。

```text
AllToAll（N=3，行是发送方，列是接收方）

        发给 R0    发给 R1    发给 R2
R0 发出   t00        t01        t02
R1 发出   t10        t11        t12
R2 发出   t20        t21        t22

R0 最终拿到 [t00, t10, t20]   ← 所有 Rank 里「该由 R0 处理」的分片
```

## 和切分策略的对应（先混个脸熟）

| 策略 | 前向里最常见的算子 |
|------|-------------------|
| TP | Column 切完可选 AllGather；Row 切完 AllReduce |
| PP | 阶段之间 Send / Recv；最后一阶段 Broadcast 采样 token |
| DP（稠密） | 基本不集体通信，请求在入口分流 |
| DP + MoE | dummy forward 对齐；专家层 AllGather / ReduceScatter 或 AllToAll |
| EP | dispatch AllToAll（或 AllGather），combine AllToAll（或 ReduceScatter） |
| SP | 把 TP 的 AllReduce 改写成 ReduceScatter + AllGather |
| DCP | 小 Q 做 AllGather，本地 KV 算 attention，再 ReduceScatter / AllToAll 合并 |

训练里 AllReduce 梯度、AllGather 参数，推理里同样这几个原语，只是传的对象换成了激活、token 和 KV。
