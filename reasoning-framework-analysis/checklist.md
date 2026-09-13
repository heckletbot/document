# 新模型优化对照清单

拿到新模型时先填场景、跑基线，再扫下面简表。只对「现象对得上」的项展开做；一次只验证一类。不要从上到下全开。

用法：简表里勾「像 / 不像 / 未测」。像 → 翻到同号细则看准入和验收。不像或未测 → 先补观察，不要改。

相关笔记：[观察](process/02-observe.md) · [路径](process/03-request-path.md) · [Bound](process/04-diagnose.md) · [融合](fusion/README.md) · [日志](log-analyze/README.md)

---

## 0. 动手前（不做完不要优化）

记下：Dense/MoE、精度、上下文、是否多模态；卡型卡数；框架/CANN 版本；混合还是 PD；TP/PP/DP/EP；负载（并发、ISL/OSL）；主指标（吞吐或 TTFT/TPOT）。

基线（warm up 后再压）：QPS 或 tok/s、TTFT、TPOT、E2E、成功率、显存、计算利用率。同一负载才能对比。

---

## 简表

| # | 优化点 | 先看什么（像了再往下翻） | Bound | 成本 |
|---|---------|--------------------------|-------|------|
| 1 | Host / 多模态预处理 | TTFT 高，但 Prefill Kernel 不长；下图/解码/tokenize 排队 | Framework / Host | 低 |
| 2 | 排队与 Scheduler | waiting 长、队列堆积、长短请求互堵 | Framework | 低 |
| 3 | Batch / Continuous Batching | 吞吐低、batch 组不满、Decode 填不满 | Framework | 低 |
| 4 | KV / Prefix Cache | preempt、evict、该命中没命中、显存碎片 | Framework / Memory | 低 |
| 5 | PD 切分与 P/D 失衡 | 一跳空一跳堵；KV 传输挡 Decode | Framework / Comm | 中 |
| 6 | 并行度 TP/EP/DP | 扩并行更慢；AllReduce / all-to-all 占比高 | Communication | 中 |
| 7 | Decode 气泡 / Device Idle | decode 段 idle/gap；利用率低但队列不空 | Scheduling | 中 |
| 8 | Prefill 计算本身 | TTFT 分解后 Prefill 前向就是大头 | Compute / Memory | 中 |
| 9 | Vector 融合（Norm/激活/Quant） | 同一 hidden 扫两遍；中间量落 GM | Memory | 中 |
| 10 | 小 kernel 融合（RoPE/MoE 路由） | 一串 &lt;50µs kernel，间隙显眼 | Scheduling | 中 |
| 11 | CV 融合（Matmul+激活） | Matmul 后立刻激活；Vector 闲、Cube 忙 | Memory | 高 |
| 12 | 通算重叠（AllGather/AllReduce+算） | 通信结束后才开算；batch 还不够大 | Communication | 高 |
| 13 | MoE 专家路径 | permute/GEMM 间激活物化；int8 解量化再量化 | Memory / Launch | 高 |
| 14 | 图编译 / 已有融合 pass | 该开的 pass 没开，或门控没过 | Framework / Memory | 低 |
| 15 | 精度与 KV 量化 | 带宽打满、显存先爆，Cube 并未更满 | Memory | 中 |

**不要做：** Cube/计算单元已经打满，再融 elementwise 或加大 batch 只换来更难的 tiling；两段需要不同并行切法时不要硬融。

---

## 逐项：做什么，以及怎样判断该不该上

### 1. Host / 多模态预处理

**做什么：** 加大媒体解码线程池到与并发同量级；去掉已有 `original_bytes` 后的二次读盘；图片/视频 `resolve` 能并发就不要一张一张来。只改 Prefill 预处理时不必重启 Decode。

**该上（要同时满足）：**
- TTFT 分解里「进模型前」占比高，Prefill 前向并不长。
- 日志/代码能看到解码排队、串行 resolve、或死 fallback `open()`。
- NPU 利用率低是结果，不是根因。

**不该上：** 纯文本、没有预处理；或 TTFT 已经在 Scheduler waiting / Prefill Kernel。

**验收：** 同一 conc 下 TTFT / E2E 明显降。QPS 抖动几个点可当噪声。脚本是 OR 判定，吞吐优先时自己看表。

---

### 2. 排队与 Scheduler

**做什么：** 看 waiting 队列、抢占、优先级；调并发或调度策略，避免长短请求互堵。翻 `vllm/scheduler/` 或 `core/scheduler.py`。

**该上：** 日志有 `waiting` / 队列长度涨；利用率低但队列很长（不是没流量）。

**不该上：** 队列空、利用率也低 → 先查流量或组不出 batch（第 3 项），不是加调度复杂度。

**验收：** 排队时间下降，TTFT 里 waiting 段缩短；preempt 次数不升。

---

### 3. Batch / Continuous Batching

**做什么：** 调 `max_num_seqs`、`max_num_batched_tokens`、batch size、PD 比例；确认 Continuous Batching 真的在填 Decode 空档。

**该上：** 吞吐低且平均 batch 远小于上限；Decode 步之间有空档；显存还没打满。

**不该上：** 显存已经顶满或 preempt 很多（先第 4 项）；TTFT SLO 很紧时盲目加大 batch 会伤首 token。

**验收：** 吞吐升、batch 曲线抬高；TTFT 若是主指标，不能用吞吐 OR 过线当成功。

---

### 4. KV / Prefix Cache

**做什么：** `gpu_memory_utilization`、KV block 大小/数量、Prefix/APC；减少无意义 evict/swap。

**该上：** `preempted` / `evict` / `swap` 频繁；多轮或系统 prompt 该命中却 `cache_hit` 低；显存碎片、OOM 边缘。

**不该上：** 单轮短请求、没有共享前缀，开 Prefix Cache 几乎没命中。

**验收：** 命中率升、preempt 降；同样并发下能留下的 KV 更多。TPOT 随带宽压力下降才算 Decode 侧收益。

---

### 5. PD 切分与 P/D 失衡

**做什么：** 看 `queue_P` / `queue_D`、KV 传输是否挡在 Decode 开工前；调 P/D 卡数或比例；KV 路径对照 Connector / 传输引擎笔记。

**该上：** 混合改 PD 后多了一跳时延；一边饿一边堵；TTFT 或 TPOT 坏在「等对端 / 等 KV」。

**不该上：** 单机混合、没有跨节点 KV，先别上 PD。

**验收：** 两队列都不再单边堆积；KV 传输与 Decode overlap 或不再挡关键路径。

---

### 6. 并行度 TP / EP / DP

**做什么：** 降过大的 TP；MoE 看 EP 与 all-to-all；通信与计算 overlap（不够再看第 12 项）。

**该上：** 单卡利用率还行，扩 TP/EP 端到端更慢；通信集合占比高。

**不该上：** 单卡已经 Compute Bound、通信占比很小。乱加 EP 可能只是换一种墙。

**验收：** 通信时间占比下降，或同等时延下吞吐上升。只看单卡 FLOPS 不够。

---

### 7. Decode 气泡 / Device Idle

**做什么：** 从日志搜 `idle` / `gap` / `bubble`；查组 batch 失败、Host 同步、Launch 过碎（第 10 项）、P/D 互抢。

**该上：** Decode 段设备空转，队列里还有活；Launch Gap 明显。

**不该上：** 队列空 → 没流量。Cube 打满的空转不是这类。

**验收：** idle 间隙缩短，TPOT 或吞吐改善；改完仍要看 Idle，融合也可能改调度空隙。

---

### 8. Prefill 计算本身

**做什么：** chunked prefill、图编译（第 14 项）、Prefill 侧融合/量化。先确认 TTFT 分解真的在前向。

**该上：** 排队和 Host 都短，Prefill forward 仍是 TTFT 主因。

**不该上：** 还没拆 TTFT 就改 Kernel。

**验收：** Prefill 段耗时下降，TTFT 跟着降；Decode/TPOT 不无故变差。

---

### 9. Vector 融合（Residual+RMSNorm、SwiGLU、Norm+Quant…）

**做什么：** 内核融合或 `forward_oot` / 图 pass。常见：`npu_add_rms_norm_bias`、`npu_swiglu`、`AddRMSNormQuantFusionPass`。中间量留 UB，少一次 GM 往返。

**该上：**
- Arithmetic 低、GM 带宽高，同一 hidden 读+写两遍。
- trace 里 add 后立刻 rms，或 norm 后立刻 quant，中间有大 GM 分配。
- 架构允许：Norm+Quant 在 310P / W4A4 上应关掉。

**不该上：** Cube 已饱和；或融合后 `baseM*baseN*dtype` 进不了 UB（看起来融了，GM 不降）。

**验收：** msprof 上 GM 降、UB 升；模型 trace 上这类小 op 合并。单算子微秒数下降但 Idle 变大，不算赢。

---

### 10. 小 kernel 融合（QK-RMSNorm+RoPE、MoE TopK）

**做什么：** `QKNormRopeFusionPass`（常要 `head_size==128` 且 BF16）；`moe_gating_top_k`。

**该上：** 一串 &lt;50µs kernel，间隙比计算显眼；QKV split → 两次 RMSNorm → RoPE 分开发射。

**不该上：** head_size / dtype 对不上门控；或热点其实是 grouped GEMM（第 13 项），不是 softmax。

**验收：** kernel 个数下降，Launch Gap 缩小。门控没过时图上不会替换，先查 pass 是否 apply。

---

### 11. CV 融合（Matmul + 激活）

**做什么：** `__mix__(1,2)`，`GetTensorC` 进 UB 再做激活，路径 L0C→UB→GM，不要 Matmul 写 GM 再读。

**该上：** Matmul 后紧跟 ReLU/GELU 等；该 Matmul 的 Vector 利用率接近 0；解耦架构（2201+）。

**不该上：** 没有紧跟 Vector；C 块装不进 UB；SIMT 上做 Matmul。

**验收：** 少一次 GM 往返（Memory.csv）；Vector 开始吃 Matmul 输出。换 3510 要改 L0A NZ、不能再走 L1→GM 直连。

---

### 12. 通算重叠

**做什么：** AllGather+Matmul（MC²，HCCL 高层 API，不能 kernel 直调）；或 `MatmulAllReduceAddRMSNormPass`。常见门控 `compile_range.start > 512`；`flash_comm_v1_enabled` 一类开关。

**该上：** AllReduce/AllGather 结束后才开始下一跳计算；batch/token 数够大，重叠才盖得住通信。

**不该上：** 小 batch（门控都过不了）；或问题其实是 TP 过大（先第 6 项）。不要和 CV `__mix__` 混成一件事。

**验收：** 通信与计算在时间上重叠；端到端通信等待下降。小 shape 下可能持平，按失败方向记下。

---

### 13. MoE 专家路径

**做什么：** TopK 一次做完（第 10 项）；`moe_init_routing_custom` 管 permute；`dequant_swiglu_quant` 管 int8 专家链。瓶颈经常是排列带宽 + GEMM 间激活物化。

**该上：** 模型是 MoE；gating 四次扫描，或专家 int8 解量化/再量化各扫一遍。

**不该上：** Dense 模型。不要只融 softmax 却不管 grouped GEMM。

**验收：** 专家路径 GM 流量或 kernel 数下降；专家负载别更歪。310P 排除 BF16 路径。

---

### 14. 图编译 / 已有融合 pass

**做什么：** 先查仓库里已有 pass 开了没、门控过不过，再写新 kernel。清单见 [fusion/03-integrate.md](fusion/03-integrate.md)。

| Pass | 门控（不过就等于没上） |
|------|------------------------|
| AddRMSNormQuant | 非 310P，且 `enable_custom_op` |
| QKNormRope | `head_size==128` 且 BF16 |
| MatmulAllReduceAddRMSNorm | `compile_range.start > 512` |
| MulsAdd | 非 310P |
| No-op 清理 | 总是；配新 pattern 要按删节后的图 |

**该上：** 图上仍是未融合序列，但门控本该满足。

**不该上：** 门控故意不满足（310P、小 batch、W4A4）；不要复制 naive RMSNorm 去配已经是 `npu_add_rms_norm_bias` 的图。

**验收：** FX/Inductor 图上出现融合 op；再跑第 9–12 项的 msprof 验收。

---

### 15. 精度与 KV 量化

**做什么：** 权重量化、激活/KV 量化、Attention 写出 FP8。带精度点测。

**该上：** HBM 带宽或容量先爆，计算单元没吃满；SLA 允许精度换吞吐。

**不该上：** 已经 Compute Bound；或精度点测挂了。融合+量化要一起看中间是否还落全精度 GM。

**验收：** 显存/带宽下降，吞吐或可服务并发上升；精度在约定范围内。

---

## 每轮收口（所有项共用）

1. 只选简表里「像」的一项，写一句可证伪假设。  
2. 改完同一 Workload、同一 warm up，对比基线。  
3. 主指标过线则留；否则回退并记下失败方向。  
4. `has_improvement` 是任一指标过约 5%，不是主指标 AND。  

口头顺序仍是：定场景 → 基线 → 路径拆分 → 判 Bound → 对简表 → 做一项 → 回归。
