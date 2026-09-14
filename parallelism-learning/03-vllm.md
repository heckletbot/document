# vLLM 里怎么体现切分

配置进 `ParallelConfig`，进程按 world size 拉起来，Worker 建通信组，模型层里真正发集合通信。按这个顺序看，不会在类名里转晕。

## 启动参数落到哪

```text
vllm serve $MODEL \
  --tensor-parallel-size 4 \              # TP
  --pipeline-parallel-size 2 \            # PP
  --data-parallel-size 2 \                # DP
  --enable-expert-parallel \              # EP，大小 = TP×DP，不能单独设
  --decode-context-parallel-size 2 \      # DCP，不增卡
  --prefill-context-parallel-size 1 \     # PCP，会乘进 world_size
  --all2all-backend deepep_low_latency    # EP 的 AllToAll 实现
```

对应 `vllm/config/parallel.py` 的 `ParallelConfig`。两个 world size 不要混：

```text
world_size            = TP × PP × PCP
                        ↑ 一个 EngineCore 拉起来的 Worker 数

world_size_across_dp  = world_size × DP
                        ↑ 整次部署的 GPU 数（external_launcher 时 DP 也会乘进 world_size）

DCP 不出现在乘法里：它把已有的 TP 组再逻辑切一刀。
```

约束在 `__post_init__` 里检查，常见坑：

- `tp_size % dcp_size == 0`，否则直接报错。
- 开 EPLB 必须先开 EP，且 `TP × DP > 1`。
- 稠密模型开外部 DP CLI 不受支持，独立起多个 `vllm serve` 即可。

## 进程怎么长出来

```text
API Server（可 --api-server-count 横向扩）
    │ ZMQ
    ▼
EngineCore × DP                         ← 每个 DP rank 一个调度/执行核
    │
    ├─ DP Coordinator（MoE 才需要）     ← dummy forward、空闲暂停
    │
    └─ Executor
         MultiprocExecutor / Ray
         Worker × (TP × PP × PCP)       ← 真正占 GPU
              GPUWorker.init_device
              GPUWorker.load_model
              GPUWorker.determine_available_memory
```

单机 `DP=2 TP=2`：2 个 EngineCore，每个底下 2 个 Worker，共 4 卡。多节点内部 DP 要在每个节点写 `--data-parallel-size-local`、`--data-parallel-start-rank`、`--headless`；外部 DP 则每个 rank 一次 `vllm serve --data-parallel-rank i`。

## Worker 里建组

入口：`vllm/v1/worker/gpu_worker.py` → `init_worker_distributed_environment`。

```text
1. set_custom_all_reduce(...)                  # 可选自定义 AllReduce kernel
2. init_distributed_environment(world_size, rank, ...)
3. ensure_model_parallel_initialized(
       tp, pp, prefill_cp, decode_cp)
     └─ initialize_model_parallel(...)         # vllm/distributed/parallel_state.py
4. ensure_kv_transfer_initialized(...)         # P/D 分离才有
```

`initialize_model_parallel` 按全局 rank 切出这些 `GroupCoordinator`：

| 组 | getter | 谁在一个组里 | 典型集合通信 |
|----|--------|--------------|--------------|
| TP | `get_tp_group()` | 相邻 `tp_size` 张卡 | AllReduce / AllGather / ReduceScatter |
| PP | `get_pp_group()` | 同一条流水线上的 stage | Send / Recv / Broadcast |
| DP | `get_dp_group()` | 同一 TP 位置、不同数据副本 | MoE dummy、状态 AllReduce |
| EP | `get_ep_group()` | `TP × DP`（MoE 才建） | AllToAll / AllGather / ReduceScatter |
| PCP | `get_pcp_group()` | Prefill 上下文并行 | AllGather KV 或 ring |
| DCP | `get_dcp_group()` | 从 TP 组里再切 | AllGather Q，ReduceScatter output |

官方注释里的 8 卡例子（TP=2，PP=4）：

```text
GPU:  g0 g1 g2 g3 g4 g5 g6 g7

TP 组: [g0,g1] [g2,g3] [g4,g5] [g6,g7]
PP 组: [g0,g2,g4,g6] 和 [g1,g3,g5,g7]
```

封装集合通信的薄层在 `vllm/distributed/communication_op.py`，TP 侧就这几个：

| 函数 | 底层 |
|------|------|
| `tensor_model_parallel_all_reduce` | `get_tp_group().all_reduce` |
| `tensor_model_parallel_all_gather` | `get_tp_group().all_gather` |
| `tensor_model_parallel_reduce_scatter` | `get_tp_group().reduce_scatter` |
| `tensor_model_parallel_gather` | `get_tp_group().gather` |
| `broadcast_tensor_dict` | `get_tp_group().broadcast_tensor_dict` |

PP 不走这层，直接 `get_pp_group().send_tensor_dict` / `recv_tensor_dict`。EP 走 `get_ep_group().dispatch` / `combine`，backend 在 `vllm/distributed/device_communicators/all2all.py`。

## 前向里各策略落在哪

### TP：线性层

`vllm/model_executor/layers/linear.py`

| 类 | 切法 | 前向通信 |
|----|------|----------|
| `ColumnParallelLinear` | W 按列（输出维） | `gather_output=True` 时 AllGather |
| `MergedColumnParallelLinear` | gate/up 熔在一起再按列切 | 同上 |
| `QKVParallelLinear` | 按头切，GQA 自动复制 KV 头 | 通常不 gather |
| `RowParallelLinear` | W 按行（输入维） | `reduce_results=True` 时 AllReduce |
| `ReplicatedLinear` | 不切 | 无 |

词表：`vllm/model_executor/layers/vocab_parallel_embedding.py` 的 `VocabParallelEmbedding` / `ParallelLMHead`。

Llama 一类模型里就是：

```text
qkv_proj = QKVParallelLinear
o_proj   = RowParallelLinear          # AllReduce
gate_up  = MergedColumnParallelLinear
down_proj= RowParallelLinear          # AllReduce
```

自定义 AllReduce 比 NCCL 快时默认开着，`--disable-custom-all-reduce` 关掉。跨节点 IB 要确认 NCCL 日志里是 `NET/IB/GDRDMA` 而不是 `NET/Socket`。

### PP：层列表和中间张量

建层：`vllm/model_executor/models/utils.py` 的 `make_layers`。当前 stage 以外全部换成 `PPMissingLayer()`，`is_pp_missing_parameter` 让 weight loader 跳过。

Llama 骨架（`vllm/model_executor/models/llama.py`）：

```text
first rank:  VocabParallelEmbedding
其他 rank:   PPMissingLayer

make_layers(...)     # 只物化 [start_layer, end_layer)

last rank:   RMSNorm + lm_head
其他 rank:   PPMissingLayer
```

阶段之间传 `IntermediateTensors`（`hidden_states` + `residual`）。Runner V2 里 `PPHandler`：

- 非末 stage：`forward` 返回中间张量，Send 给下一 stage
- 末 stage：采样，再 Broadcast `sampled_token_ids` 给前面各 stage

异步 Send 默认开，`--disable-pp-async-send` 关掉。

### DP：调度而不是层

稠密：每个 EngineCore 独立，模型代码无 DP 分支。

MoE：`vllm/v1/worker/dp_utils.py` 做「还有没有未完成请求」的 AllReduce；Coordinator 让空闲 rank 跟 dummy forward。负载均衡在 API Server 里看各 engine 的 running/waiting 队列，不是 NCCL。

### EP：FusedMoE

三维切分（`moe_ep` / `moe_tp` / `moe_dp`）的概念见 [02-strategies.md](02-strategies.md)。vLLM 的暴露更窄：

```text
不开 --enable-expert-parallel
  moe_ep = 1，moe_tp = 展平后的 TP（含 DP）
  每张卡持有全部 expert，矩阵按列/行切   ← Mixtral 那种「少而大」

打开 --enable-expert-parallel
  moe_ep = TP × DP，moe_tp = 1
  每张卡持有一组完整 expert，不再切单个 expert 的矩阵  ← DeepSeek 那种「多而小」
```

对应 `FusedMoEParallelConfig`：`use_ep=True` 时把 `tp_size` 改写成 1，把原来的 TP rank 当成 EP rank。vLLM **没有** SGLang 那种独立的 `--ep-size` / `--moe-data-parallel-size`。

路径：

```text
Router  →  Quantize + dispatch  →  Expert GEMM  →  combine
```

dispatch/combine 三套实现都走同一对 API：

| 文件 | 看什么 |
|------|--------|
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | 四段流水 |
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | 按 `--all2all-backend` 选 PrepareAndFinalize |
| `.../prepare_finalize/naive_dp_ep.py` | 默认：`get_ep_group().dispatch` / `combine` = AllGather + ReduceScatter |
| `vllm/distributed/device_communicators/all2all.py` | DeepEP / FlashInfer 的真 AllToAll |

`--enable-ep-weight-filter`：每张卡磁盘上只读自己那份 expert，大 MoE 加载会快一截。

### SP：编译期改图

`vllm/compilation/passes/fusion/sequence_parallelism.py` 的 `SequenceParallelismPass`：

```text
AllReduce → RMSNorm (→ FP8 quant)
改成
ReduceScatter → 本地 RMSNorm (→ quant) → AllGather
```

`PassConfig.enable_sp` 打开；`fuse_gemm_comms=True` 会强制连带打开，因为 AsyncTP 建立在这次改写上。只在 token 数超过阈值、且通常 `hidden_size >= 8192` 时应用。

MoE 那条 SP 不是这个 pass，是 `ParallelConfig.use_sequence_parallel_moe`：EP + TP>1 + DP>1 时，专家输入沿序列切开，避免 TP 组内重复 token。

### CP：KV 布局 + Attention backend

| 文件 | 看什么 |
|------|--------|
| `vllm/v1/worker/block_table.py` | 按 `cp_kv_cache_interleave_size` 算本地 slot；`max_model_len / (dcp×pcp)` 才是每卡要存的 token |
| Attention backend（FlashMLA / GQA） | `get_dcp_group().all_gather` 拼 Q 或 KV，再本地 attention |
| 环境变量 `VLLM_DCP_Q_REPLICATE=1` | MLA 可在加载时复制 Q 投影，decode 省掉 Q 的 AllGather |

`--dcp-comm-backend a2a` 把默认的 AllGather+ReduceScatter 换成 AllToAll，MLA 每层 NCCL 次数从 3 降到 2。

## 前向通信地图（一层 Transformer）

稠密 + TP（最常见）：

```text
RMSNorm（无通信，或 SP 时前面刚 AllGather 完）
QKVParallelLinear          无
Attention                  无（DCP 时 AllGather Q / 合并 output）
RowParallelLinear o_proj   AllReduce          ← TP 主成本
RMSNorm
MergedColumnParallelLinear 无
SiLU                       无
RowParallelLinear down     AllReduce          ← TP 主成本
```

MoE + DP + EP：

```text
Attention 同 TP/DP
Router                     无
dispatch                   AllToAll 或 AllGather     ← EP 主成本
Expert GEMM                只算本地 expert
combine                    AllToAll 或 ReduceScatter
```

PP 叠在外面：本 stage 算完上述一整段后，Send hidden 到下一 stage，本层内部的 TP/EP 通信不变。

## 对推理指标的影响

切分不是免费的。看它换到的是显存、TTFT 还是 TPOT。

| 策略 | 主要换到什么 | 典型代价 | 何时会变慢 |
|------|--------------|----------|------------|
| TP | 单卡放得下权重；decode 多卡算一层 | 每层 AllReduce | TP 过大、GQA 头已切光还在复制 KV；跨节点无 NVLink |
| PP | 层拆开装得下；跨节点/无 NVLink | Send/Recv 轻；decode 空泡重 | 量化后单机已能装下时还开 PP，TPOT 往往差过纯 TP |
| DP | 吞吐近线性；每 rank 一份 KV | 调度、MoE dummy | 负载不均、prefix cache 被打散 |
| EP | 专家局部性、Attention 可 DP | 每层 AllToAll | backend 和 P/D 阶段不匹配；expert 热度倾斜（没 EPLB） |
| SP | 大 batch 时 Norm/通信可重叠 | 小 batch 额外切分 | token 数低于阈值仍强制开 |
| DCP | 去掉 KV 复制，decode batch 变大 | AllGather Q + 合并 | dcp 太大，通信吃掉省下的显存收益 |
| PCP | 长 prefill 的 TTFT 摊到多卡 | AllGather KV 或 ring | 中等长度 prompt，通信 > 节省的计算 |

官方推荐的加卡顺序（单副本）：

```text
单卡装得下且延迟够用
    → 停
单机多卡、有 NVLink
    → 加 TP，直到 KV 行数 / 并发够
GQA/MLA 开始复制 KV
    → 加 DCP，范围 [1, tp/H_kv]
单机仍装不下（量化之后），或无 NVLink
    → 才加 PP（跨节点：TP=每节点卡数，PP=节点数）
要吞吐而不是单请求延迟
    → 加 DP（MoE 再加 --enable-expert-parallel）
```

日志里两行是切完之后的体检：

```text
GPU KV cache size: N tokens
Maximum concurrency for max_model_len: Xx
```

KV 行数不够就继续加卡或加 DCP；行数很多但 TPOT 掉下去，多半是 TP/EP 通信，该减 TP 或换 AllToAll backend。

仓库里 Qwen3.5-122B 的例子：SLO 从 TTFT 300ms 放到 500ms 后，部署从 TP8 改成 TP4（Prefill `2P(TP4)`，Decode `1D(TP4DP4)`）。TP 减小 → 单卡 KV 变大、通信变轻，QPS 上来；APC 的 block 对齐被连带打坏，那是另一条优化，不是切分本身。

PD 分离是实例角色切分，不是这六种里的一种。P/D 各自内部仍用 TP/DP/EP；实例之间传 KV 走 Connector（Mooncake / NIXL），用的是 Send 语义的 RDMA，不是 NCCL AllReduce。

## 关键代码索引

| 主题 | 路径 |
|------|------|
| 配置与 world_size | `vllm/config/parallel.py` |
| 进程组 | `vllm/distributed/parallel_state.py` |
| TP 集合通信封装 | `vllm/distributed/communication_op.py` |
| 线性层 TP | `vllm/model_executor/layers/linear.py` |
| 词表 TP | `vllm/model_executor/layers/vocab_parallel_embedding.py` |
| PP 建层 | `vllm/model_executor/models/utils.py` `make_layers` |
| PP 中间量 | `IntermediateTensors`；Runner V2 的 `PPHandler` |
| Worker 初始化 | `vllm/v1/worker/gpu_worker.py` `init_worker_distributed_environment` |
| DP 对齐 | `vllm/v1/worker/dp_utils.py` |
| MoE AllToAll | `vllm/model_executor/layers/fused_moe/`、`device_communicators/all2all.py` |
| 编译期 SP | `vllm/compilation/passes/fusion/sequence_parallelism.py` |
| DCP KV 布局 | `vllm/v1/worker/block_table.py` |

官方文档：

- [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)
- [Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html)
- [Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)
- [Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment.html)
- [Decode Context Parallelism 博客](https://blog.vllm.ai/2026/08/07/decode-context-parallelism.html)
