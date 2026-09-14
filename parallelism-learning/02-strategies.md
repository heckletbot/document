# 切分策略

六种策略切的不是同一个东西。先看「切什么」，再看通信，最后才记启动参数。

```text
一次前向里能被切开的东西

  请求 / batch  ──────────────  DP
  Transformer 层 ────────────  PP
  同一层里的矩阵 / 头 ────────  TP
  MoE 的专家权重 ────────────  EP
  激活的 token / 序列维 ──────  SP（推理里多半是 TP 的通信改写）
  KV / 上下文的序列维 ────────  CP（Prefill CP / Decode CP）
```

## 张量并行 TP

**切什么：** 同一层的权重矩阵。每张卡都跑全部层，但只持有矩阵的一列块或一行块。

原理是把 `Y = X @ W` 拆开算。约定：X 是 `[token, in]`，W 是 `[in, out]`，Y 是 `[token, out]`。TP=2。

**列拆分**（`ColumnParallelLinear`）：W 按**输出维**竖着切成左右两块。

```text
W = [ W0 | W1 ]          每卡一块列，形状 [in, out/2]

Y  = X @ [ W0 | W1 ]
   = [ X@W0 | X@W1 ]     每卡用完整 X，算出 Y 的一半

Rank0 持有 W0，得到 Y 的左半；Rank1 持有 W1，得到 Y 的右半。
输入 X 每张卡都有一份；算完默认不通信。要完整 Y 时再 AllGather。
```

**行拆分**（`RowParallelLinear`）：W 按**输入维**横着切成上下两块，X 也按最后一维切开。

```text
W = [ W0 ]               每卡一块行，形状 [in/2, out]
    [ W1 ]

X = [ X0 | X1 ]

Y  = X0@W0  +  X1@W1     每卡算一份「部分和」，形状都是完整的 [token, out]

Rank0：X0 @ W0
Rank1：X1 @ W1
AllReduce 把两份加起来，每张卡都得到完整 Y。
```

列拆分切的是**输出**，行拆分切的是**输入**。MLP 因此固定成一对：先 Column（gate/up），中间 SiLU 各算各的，再 Row（down）用 AllReduce 拼回去。

```text
MLP 一层（TP=2）

          Rank0                    Rank1
     ┌─────────────┐          ┌─────────────┐
X ──►│ W_up[:, 左] │     X ──►│ W_up[:, 右] │     Column：输入完整，输出切开
     └──────┬──────┘          └──────┬──────┘
            │ SiLU                   │ SiLU          逐元素，不用通信
            ▼                        ▼
     ┌─────────────┐          ┌─────────────┐
     │ W_dn[上, :] │          │ W_dn[下, :] │     Row：输入切开，部分和
     └──────┬──────┘          └──────┬──────┘
            └──────── AllReduce ──────┘
                       ▼
                    完整 Y
```

Attention 同构：`QKVParallelLinear` 按头切（GQA 时 KV 头不够会在 TP 组内复制），本地做 attention，`o_proj` 用 `RowParallelLinear` 再 AllReduce。

```mermaid
flowchart LR
  X[完整 hidden] --> QKV[QKVParallelLinear]
  QKV --> Attn[本地 Attention]
  Attn --> O[RowParallelLinear o_proj]
  O --> AR[AllReduce]
  AR --> Y[完整 hidden]
```

**通信：** 几乎每层一次 AllReduce（decode 时 token 少，AllReduce 延迟很容易成为 TPOT 瓶颈）。Embedding / LM Head 走 `VocabParallelEmbedding`，词表按行切，前后常配 AllReduce 或 AllGather。

**效果：**

- 单卡装不下权重时的第一刀。单机多卡、有 NVLink，优先 TP。
- TP 越大，每卡算力越碎，通信次数不减。GQA / MLA 的 KV 头很少，TP 超过头数后 KV 会在多卡上复制，显存白花。
- 官方经验：先加到「KV cache 行数 / 并发」够用；跨节点再叠 PP，而不是无脑把 TP 拉到集群总卡数。

## 流水并行 PP

**切什么：** 层。Rank 0 拿 embedding + 前几层，中间 Rank 拿中间层，最后 Rank 拿剩下的层 + norm + LM Head。

```text
PP=2，模型 48 层

Rank0 / Stage0                    Rank1 / Stage1
embed + layers[0:24]              layers[24:48] + norm + lm_head
        │ Send hidden, residual
        └────────────────────────►│ Recv 后接着算
                                  │ 采样出 token
        │ Broadcast sampled ids   │
        ◄─────────────────────────┘
```

```mermaid
sequenceDiagram
    participant S0 as PP Rank0
    participant S1 as PP Rank1

    S0->>S0: embed + 前半层
    S0->>S1: Send IntermediateTensors
    S1->>S1: 后半层 + sample
    S1->>S0: Broadcast sampled tokens
```

**通信：** 阶段之间是点对点 Send / Recv，不是 AllReduce。体积是一份 hidden（加 residual），比 TP 每层 AllReduce 轻，所以没 NVLink、跨节点时 PP 往往比纯 TP 更划算。最后一阶段要把采样 token 广播回前面各 stage，下一轮才能继续。

**效果：**

- 单机卡数除不尽、或卡之间没有 NVLink（例如 L40S）：官方建议 `TP=1, PP=卡数`。
- 跨节点经典配方：`TP = 每节点卡数`，`PP = 节点数`。
- Decode 每步只有 1 个 token，流水线很难填满，气泡明显。推理里 PP 主要换的是「装得下」，不是 decode 延迟。Prefill 大 batch 时微批可以掩盖一部分气泡。

## 数据并行 DP

**切什么：** 请求。每张卡（或每个 TP 组）持有同一份 Attention 权重，独立 KV cache，各自跑自己的 batch。

```text
DP=2 TP=2，共 4 卡

        DP rank0                    DP rank1
     TP0        TP1              TP0        TP1
   replica A  replica A        replica B  replica B
   请求 1,3   同左切权重         请求 2,4   同左切权重
   独立 KV                      独立 KV
```

稠密模型：DP rank 之间前向互不通信，入口把请求分出去就行。

MoE 模型：专家层要在 `DP × TP` 范围内同步。任一 rank 还有请求在跑，空闲 rank 必须跟一次 dummy forward，否则 AllToAll / AllGather 对不齐。vLLM 用独立的 DP Coordinator 做这件事。

**通信：** 稠密几乎为零。MoE 时专家层走 EP 或「把 MoE 当成更大的 TP」。调度侧用 ZMQ，不算 NCCL 集合通信。

**效果：**

- 吞吐按 rank 数近似线性扩，前提是负载均衡和 prefix cache 命中。每个 DP rank 一份独立 KV，相同前缀打到同一 rank 才能吃到 APC。
- `--max-num-seqs` 是 **每个 DP rank** 的上限；`--max-num-queued-reqs` 是整机。DP=4 时如果队列上限仍按单 rank 来设，会提前拒请求。
- 不增加单请求算力，不降低 TTFT。单条长请求仍然要靠 TP / CP。

## 专家并行 EP

**切什么：** MoE 里的 M 个专家均分到 N 张卡，所有卡按专家切分 MoE 权重。Attention 仍按 DP/TP。

MoE 适合这么切，因为专家权重是参数大头，但每次只激活 top-k 个；各个 expert 是互相独立的 MLP，不必像稠密 FFN 那样把同一份矩阵按列/行切开。

### 效果

单卡塞不下全部 expert 时，分到多卡，**数学上和单卡算完全部 expert 再加权求和等价**，用通信换显存。

Prefill 时 token 多、专家 GEMM 吃得满，多卡并行算不同 expert，等于扩了算力，TTFT 能降。Decode 每步 token 少，通信占比更高，更吃 AllToAll 的延迟。

### 实现（Attention TP/DP + Expert EP）

```text
Attention TP=2，Attention DP=4，共 8 卡，EP=8（无冗余）

Attention：4 个 DP 组，每组内 TP=2 切 QKV / O
Experts：256 个 expert 摊到 8 张卡，每卡 32 个完整 expert

Token 路由：
  本卡 hidden + topk_ids
       │ dispatch（按 expert 把 token 送到持有它的卡）
       ▼
  持有该 expert 的卡做 GEMM
       │ combine（结果送回 token 原来的卡）
       ▼
  token 回到原卡，和 Attention 输出对齐
```

```mermaid
flowchart TB
  subgraph attn [Attention 按 DP/TP]
    A0[DP0 TP 组]
    A1[DP1 TP 组]
  end
  subgraph moe [MoE 按 EP]
    E0[experts 0-31]
    E1[experts 32-63]
    E2[...]
  end
  attn -->|dispatch| moe
  moe -->|combine| attn
```

### 三套通信实现

前向只有两步语义：**dispatch 送出去，combine 收回来**。底下常见三套实现，不是三种切分。

| 实现 | 实际集合通信 | 在干什么 | 何时用 |
|------|----------------|----------|--------|
| AllToAll | 每个 Rank 按 expert 归属和所有其他 Rank 交换 | 真正的「每人给每人」 | DeepEP / pplx / FlashInfer A2A，大规模 EP |
| AllGather + ReduceScatter | dispatch 用 AllGather 收齐所有 token；combine 用 ReduceScatter 送回 | 用两步模拟 AllToAll，实现简单 | 默认通吃；hybrid EP+TP 时常走这条 |
| dispatch / combine 融合核 | 库自己的一对 API，内部仍是 A2A 或 fused | 把 permute + 通信 + 反 permute 焊在一起 | DeepEP、`ascend_fuseep` 等 |

AllGather+ReduceScatter 的直观过程（EP=2）：

```text
Rank0 的 token 要去 expert 0 和 1
Rank1 的 token 要去 expert 0 和 1

AllGather：每张卡都拿到全部 token（有一份冗余计算/流量）
本地只跑自己持有的 expert
ReduceScatter：按 token 原来的归属把结果加完送回
```

比真 AllToAll 容易写，流量通常更大。vLLM 默认 `--all2all-backend allgather_reducescatter`；DeepEP 才换成真正的 AllToAll。SGLang 默认 `--moe-a2a-backend none`（AR/AG），`--moe-a2a-backend deepep` 才走 dispatch/combine。

Token 按 expert 分布不均时开 EPLB，周期性把热 expert 迁到别的卡。这是权重搬迁，不是前向主路径。

### TP 会切到 MoE 层吗？

先澄清名字：启动里的 `--tp N` **不等于「MoE 层的 TP」**。在 SGLang 里 `--tp` 是这套副本的 world size（总卡数，不含 PP/外部 DP）；MoE 自己还有三个维度，乘起来要等于这个 world：

```text
tp_size(world) = moe_ep_size × moe_tp_size × moe_dp_size
```

| 维度 | 切什么 | 如何设 |
|------|--------|--------|
| `moe_ep_size` | experts 列表（`n_experts` 分到 N 个 rank，每 rank 持 `n_experts/ep` 个**完整** expert） | 用户指定（SGLang `--ep-size`） |
| `moe_tp_size` | **单个** expert 的权重（`W_gate` / `W_up` / `W_down` 再按 intermediate 切） | 自动推导 `= tp / ep / moe_dp` |
| `moe_dp_size` | MoE 副本（world 切成 N 份独立 MoE，**副本之间不交换 token**） | 用户指定（SGLang `--moe-data-parallel-size`，默认 1） |

所以「TP 会切到 MoE 吗」精确版是：`moe_tp_size` 配成 1 还是 >1？

**配置 A：`moe_tp = 1`（默认，专家多而小）**

每个 expert 完整放在单个 rank 上。MoE 层只有 EP 的 dispatch/combine，没有 AllReduce。

例：DeepSeek-V3（256 experts，单 expert 大约 1.5 GB），`--tp 16 --ep-size 16` → `moe_tp = 16/16/1 = 1`，每 rank 持 16 个完整 expert。

单 expert 往往比 Attention 的 `W_q` 还小，再切一份 TP 收益微小，却多一次 AllReduce。大 MoE 默认不切。

**配置 B：`moe_tp > 1`（专家少而大）**

单 expert 太大，一张卡装不下时，把 expert **内部**按 TP 切，语义和稠密 FFN 的 Column + Row 一样。

例：Mixtral-8×7B（8 个大 expert），`--tp 16 --ep-size 8` → `moe_tp = 2`，每 rank 持 1 个 expert 的一半权重。

MoE 层多一次 **MOE_TP AllReduce**（`_MOE_TP` 组，size = `moe_tp`），FFN 后合并 partial。语义同标准 TP AllReduce，只是组更小。SGLang 里 hybrid EP+TP 目前主要是 `--moe-a2a-backend none` 这条路；DeepEP 一类要求 `ep_size = tp_size`，也就是 `moe_tp = 1`。

判断：看**单 expert 能不能塞进单卡**。多而小（DeepSeek 类）→ `moe_tp=1`；少而大（Mixtral 类）→ `moe_tp>1`。启动后看 `mlp.experts.N.gate_proj.weight.shape`：intermediate 维被除过，就是被 MoE TP 切了。

```text
moe_tp=1：gate_proj 是 [intermediate, hidden]     完整 expert
moe_tp=2：gate_proj 是 [intermediate/2, hidden]   列切了
```

### `moe_dp_size` 是独立参数

`moe_dp_size` **不是**从 attn 的 tp/dp/cp 推出来的，是用户主动选的：`--moe-data-parallel-size N`（默认 1）。语义是「把 world 切成 N 份**独立 MoE 副本**，副本之间不做 EP AllToAll」。

约束（SGLang `server_args.py`）：

- `ep_size × moe_dp_size ≤ tp_size`（`ep_size > 1` 时取等号）
- `attn_cp_size != moe_dp_size` 时必须 `moe_dp_size == 1`

| 场景 | `moe_dp_size` | 说明 |
|------|---------------|------|
| 默认：MoE 全域 EP | 1 | 最常见，专家摊最散 |
| MoE A2A 带宽紧，复制权重省通信 | `= attn_dp_size` | 副本级 DP，副本间零 A2A |
| CP + MoE | 保持 1 | CP 的 token 合/切靠 `_MOE_DP` 与 `_ATTN_CP` 别名，不是调这个参数 |

绝大多数部署保持默认 `moe_dp_size = 1`。

### MoE 三个切分参数怎么记

```text
一张卡上的 MoE 看到什么

  moe_ep_size   我持有哪几个完整 expert          ← 列表切开
  moe_tp_size   每个 expert 的矩阵还切不切       ← 矩阵切开
  moe_dp_size   对面那组卡是不是另一份 MoE 副本  ← 请求切开，副本不换 token
```

| 参数 | 谁设 | 默认 | 主通信 | 一句话 |
|------|------|------|--------|--------|
| `moe_ep_size` | 用户 `--ep-size` | 1（等于没开 EP） | dispatch / combine | 专家摊到多卡 |
| `moe_tp_size` | `tp / ep / moe_dp` 自动除出来 | 通常 1 | 组内 AllReduce | 单个 expert 再切一刀 |
| `moe_dp_size` | 用户 `--moe-data-parallel-size` | 1 | 副本间无 A2A | 复制一份 MoE，省跨副本通信 |

和 Attention 侧的 `--tp` / `--dp` 不要对号入座：`--tp 16` 是总卡数；Attention 还可以再拆 `attn_tp × attn_dp × attn_cp`。EP 只回答「专家怎么放」，不回答「QKV 怎么切」。

### 启动参数对照（SGLang vs vLLM）

同一套三维，两套 CLI 暴露方式不同。

| 概念 | SGLang | vLLM |
|------|--------|------|
| 副本 world | `--tp N`（常等于总卡数） | `world_size = TP × PP × PCP`；`--tensor-parallel-size` **不是**总卡数 |
| 开 EP | `--ep-size N`（用户指定） | `--enable-expert-parallel`，然后 `EP = TP × DP`，不能单独设 |
| `moe_tp_size` | `tp/ep/moe_dp`，可以为 >1 | 开 EP 后 **强制 1**（每个 device 持有完整 expert）；不开 EP 则 MoE 整层当 TP 切，所有 expert 每卡都有一份、矩阵按列/行切 |
| `moe_dp_size` | `--moe-data-parallel-size`，副本间不 A2A | 无同名旋钮。`--data-parallel-size` 开 EP 时是把更多卡并进**同一个** EP 组，rank 之间 **要** 换 token |
| A2A 实现 | `--moe-a2a-backend`：`none` / `deepep` / … | `--all2all-backend`：`allgather_reducescatter` / `deepep_*` / … |

vLLM 开 EP 之后走的就是上面的配置 A：`moe_tp = 1`。Mixtral 那种「EP 再叠 MoE TP」在 vLLM 里对应的是 **不开** `--enable-expert-parallel`，让 MoE 层组成大小为 `TP × DP` 的 TP 组去切矩阵。

## 序列并行 SP

**切什么：** 激活的 token 维，不是再切一份权重。推理里它多半是 **TP 通信的改写**，让 LayerNorm 只看到 `1/TP` 的 token，并为后面的 AsyncTP（GEMM 和通信重叠）铺路。

训练 Megatron-SP 是「LayerNorm / Dropout 沿序列切开，省激活显存」。vLLM 的编译期 SP 做的是图替换：

```text
原来（TP）：
  RowParallelLinear ── AllReduce ── RMSNorm ── 下一层

改写后（SP）：
  RowParallelLinear ── ReduceScatter ── 本地 RMSNorm ── AllGather ── 下一层
                       （每卡 1/TP token）
```

ReduceScatter + AllGather 的总字节数等于一次 AllReduce，但 RMSNorm 算在切开的序列上，而且这两段通信可以分别和前后 GEMM 融合（`fuse_gemm_comms` / AsyncTP）。

还有一条 MoE 相关的 SP：`ParallelConfig.use_sequence_parallel_moe`。TP>1 且 DP>1 且开了 EP 时，Attention 的 `o_proj` AllReduce 会让进入专家层的 token 在 TP 组内重复。把专家输入改成 sequence parallel，避免同一 token 被算多遍、AllToAll 再传多遍。

**通信：** ReduceScatter、AllGather，走 TP group。

**效果：**

- 本身不保证更快，是 AsyncTP 的前置。小 batch 时切开再聚合的开销可能更亏。
- vLLM 默认只在大 hidden（H100/Blackwell 上 `hidden_size >= 8192`）且 token 数过阈值时启用。可在 `compilation_config` 里设 `enable_sp`、`sp_min_token_num`。
- 看 profile：AllReduce 很胖、RMSNorm 跟在后面，才值得开。

## 上下文并行 CP

**切什么：** KV / 上下文的 **序列维**。TP 切的是头（H），CP 切的是时间（T）。

vLLM 把 Prefill 和 Decode 拆开，因为 SLO 完全不同：

| | Prefill CP（PCP） | Decode CP（DCP） |
|--|-------------------|------------------|
| 要解决的 | 长 prompt 的 TTFT | 长上下文 decode 的 KV 显存 |
| 切法 | 新 token 切成 N 段，每卡算一段 QKV | KV cache 沿 T 交错切到 TP 组内各卡 |
| 卡数 | `world_size` **乘上** `prefill_context_parallel_size` | **不增卡**，复用 TP 的 GPU |
| 启动 | `--prefill-context-parallel-size` | `--decode-context-parallel-size` / `-dcp` |

Decode 侧为什么需要 DCP：

```text
KV 总量 ≈ H_kv × T

1. 单卡放得下 → 不切
2. 不够 → 先 TP，沿 H 切（`-tp`）
3. TP 超过 H_kv 之后，多出来的卡只能复制 KV
   复制倍数 = tp_size / H_kv
4. 再开 DCP，沿 T 切掉复制
   dcp 范围 [1, tp_size / H_kv]，且 tp 必须能被 dcp 整除
```

交错存储（interleave），后续新 token 自然落到下一张卡，不用重排：

```text
DCP=2，token 下标 0,1,2,3,4,...

Rank0 KV:  0  2  4  6  8
Rank1 KV:  1  3  5  7
```

一步 decode：

```text
各卡算出自己那片 Q
    │ AllGather Q          ← decode 只有 1 个 token，这份通信很便宜
    ▼
各卡用完整 Q × 本地 KV 做 attention
    │ ReduceScatter / AllToAll 合并 output
    ▼
完整 hidden，继续 MLP
```

MLA（DeepSeek）有效 KV 头 = 1，`-tp 8` 就是 8 份 KV 复制，`-dcp 8` 一次切掉。GQA 如 Qwen3-235B 有 4 个 KV 头，`-tp 8` 只复制 2 倍，`-dcp 2` 即可。

Prefill CP 两条路（上游仍在推进）：

1. 部分 Q、完整 KV：AllGather 各卡的 K/V，每卡只算自己那一段 Q。加速 TTFT，显存仍要扛完整 KV。
2. 部分 Q、部分 KV：ring attention，K/V 一块块 Send/Recv 转圈。上下文极长、完整 KV 放不下时才走。

**效果：**

- DCP 换的是 KV 容量和 decode batch，不是单 token 算力。通信随 dcp 变大而增加，所以先把 TP 加到性能够用，再加 DCP 去复制。
- DCP 不增加 `world_size`，和「再加一组 GPU」不是一回事。
- PCP 尚未完全落地；昇腾侧明确 PCP 保持 1。PD 传 KV 时要把 `--cp-kv-cache-interleave-size` 设成 block size，否则 block 对不齐。

## 怎么叠

六种可以同时开，但职责正交：

```text
请求先按 DP 分到副本
  每个副本内部：
    层按 PP 切开
      每一层内部：
        Attention 权重按 TP 切（或 DP 复制）
        KV 序列维按 DCP 切
        MoE：moe_ep 切专家列表，moe_tp 可选再切单个 expert，moe_dp 复制独立 MoE
        激活 token 维可按 SP 改写 AllReduce
```

常见配方：

| 场景 | 配方 |
|------|------|
| 单机稠密 70B | `TP=8` |
| 两机 405B，每机 8 卡 | `TP=8 PP=2` |
| 单机 DeepSeek-V3，H200×8 | vLLM：`TP=1 DP=8 --enable-expert-parallel`；SGLang：`--tp 8 --ep-size 8`（`moe_tp=1`） |
| Mixtral 类少而大的 expert | SGLang：`--tp 16 --ep-size 8`（`moe_tp=2`）；vLLM：不开 EP，让 MoE 走 TP |
| 同上但要去 KV 复制 | 再加 `-dcp`（MLA 可到 8） |
| Prefill / Decode 角色分离 | P、D 各自一套 TP/DP/EP；KV 走 Connector，不是这六种之一 |
