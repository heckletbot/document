# MoE 通信：AllGather + ReduceScatter

- 链路：Prefill（P 实例）MoE token dispatch / combine
- 约束：**只适用于 P 实例 DP=1**
- 收益：开 ag-rs 后 TTFT ↓ 15ms
- 场景：P 为 DP=1 且 TP/EP=8 或 4；本项目后期是 TP4EP4

## 问题怎么识别

默认 MoE 走 **AlltoAll dispatch + AlltoAll combine**：

- Dispatch：按路由把 token 精确发到持有目标 expert 的 rank
- Combine：各 EP rank 再 AlltoAll 把结果送回源 rank，再按 gate 加权

在 **EP 较小、top_k 较大** 时这条路径不稳、也不省：

- Qwen3.5-122B-A10B 的 `top_k=8`
- TP4EP4：`top_k (8) > ep_size (4)`
- AlltoAll 的实际效率还吃专家负载不均（不规则交换，带宽利用率差）

对比通信量：

| 方式 | 算子 | 数据量 | 更合适的时候 |
|------|------|--------|-------------|
| AllGather+ReduceScatter | AG + RS | `(B,S,H) × 2` | ep_size 小，top_k > ep_size |
| AlltoAll | AlltoAll × 2 | `(BS×top_k/ep_rank, H) × 2` | 大 EP，ep_size > top_k |

TP4EP4 下 AG+RS 量更小。TP8EP8 两套量接近，但 AlltoAll 仍更容易被负载不均打歪。

## 具体怎么改实现

新增通信模式 `AllGatherReduceScatter`，替换 dispatch/combine，不改专家计算本身：

**Dispatch（改后）**

1. AllGather 收集所有 EP rank 的 token（完整 `(B,S,H)` 视角）
2. 本地按路由筛自己负责的 expert 的 token

**Combine（改后）**

1. 专用算子做 expert 输出加权
2. ReduceScatter 把结果分回各 rank

为什么能快：小 EP 时规则的 AG/RS 比「按 expert 不规则 AlltoAll」更好占满带宽；同时避开负载不均导致的 AlltoAll 长尾。

限制写进实现：P 侧 DP 必须为 1。DP>1 时 token 布局/通信域变了，这套 AG-RS 不能直接套。

## 学习要点

- 识别依据是 **top_k vs ep_size + 负载不均**，不是「集合通信都慢」。
- 实现是换 MoE 的 dispatch/combine 原语，不是改 gate。
- 第一阶段时间线里「MoE 通信切换为 ReduceScatter」就是这条，后面作为可开关特性（ag-rs）。
