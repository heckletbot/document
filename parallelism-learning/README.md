# 切分策略学习笔记

先把集合通信算子钉死，再看推理里常用的六种切分：DP / PP / TP / EP / SP / CP。ZeRO / FSDP 是训练切分，这里不展开。

读完应能自己画一张图：每种策略切的是权重、激活、KV 还是请求；对应哪几个集合通信；在 vLLM 启动参数和源码里落在哪。

| 阶段 | 文档 |
|------|------|
| 集合通信 | [01-collectives.md](01-collectives.md) |
| 切分策略 | [02-strategies.md](02-strategies.md) |
| vLLM 落地与效果 | [03-vllm.md](03-vllm.md) |

一句话对照：

| 策略 | 切什么 | 每张卡上的模型 | 主通信 |
|------|--------|----------------|--------|
| DP | 请求 / batch | 权重完整（或 Attention 完整） | 几乎无；MoE 时要对齐 dummy forward |
| PP | 层 | 连续若干层 | Send / Recv 激活 |
| TP | 矩阵 / 头 | 每层都有，权重按列/行切 | AllReduce / AllGather |
| EP | 专家列表（还可再切单个 expert / 复制 MoE 副本） | Attention 按 DP/TP，Expert 子集 | dispatch / combine（A2A 或 AG+RS） |
| SP | token / seq 维激活 | 同 TP，激活沿序列切开 | ReduceScatter + AllGather |
| CP | 上下文 / KV 的序列维 | 同 TP（DCP **不增卡**） | AllGather Q/KV，或 ring Send/Recv |

选策略的粗规则：单卡装得下就别切；单机装不下先 TP；跨节点或没 NVLink 再叠 PP；要吞吐就叠 DP；MoE 用 DP Attention + EP；长上下文 decode 显存不够再开 DCP。

## 自我验证

不看正文，口头回答。卡壳回到对应文档。

1. AllReduce 和 ReduceScatter + AllGather 是什么关系？通信量差在哪？
2. ColumnParallel 和 RowParallel 哪个输入完整、哪个要 AllReduce？MLP 里 gate/up 和 down 各用哪个？
3. `world_size` 和 `world_size_across_dp` 分别乘了哪些维度？DCP 为什么不在乘法里？
4. 开 `--enable-expert-parallel` 之后 EP 大小怎么算？Attention 和 Expert 各按什么切？vLLM 开 EP 后 `moe_tp` 是几？
5. `moe_ep_size` / `moe_tp_size` / `moe_dp_size` 各切什么？「TP 会切到 MoE」问的是哪一个？
6. vLLM 的 SP 改的是权重还是通信图？它和 AsyncTP 谁依赖谁？
7. MLA 模型 `-tp 8` 为什么会 8 倍复制 KV？`-dcp 8` 增不增加 GPU？
8. 从 `vllm serve` 到第一层 AllReduce，进程组和线性层分别在哪两个文件里建出来？
