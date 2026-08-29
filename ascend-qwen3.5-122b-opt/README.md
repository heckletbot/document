# 昇腾 Qwen3.5-122B 优化方案

本目录用于整理昇腾（Ascend）A3 上 Qwen3.5-122B-A10B-w8a8 的优化方案总结与学习笔记。

场景：ModelArts AI 搜索（XHS）线上推理，PD 分离，vLLM + vllm_ascend + ascend_vllm。  
材料来源：*ModelArts AI 搜索推理优化系列：Qwen3.5-122B 性能调优实践*（2026-08-03）。配图未随文提供，关键时序用文字/mermaid 还原。

每条优化单独成文，重点记两件事：

1. **问题怎么识别**：现象、定位手段、判断依据。
2. **具体怎么改实现**：改了哪些路径/语义/配置，改前改后差在哪，为什么这样改有效。

## 优化点索引

按请求生命周期排列。本轮已根据原文整理到 KV 控制面；Decode（MTP 零气泡等）与稳定性章节原文被截断，待后续补。

| # | 链路位置 | 笔记 | 原文收益 |
|---|---------|------|---------|
| — | 总览 | [00-overview.md](00-overview.md) | 纯文本单卡 QPS 0.022→0.57；多模态 <0.1→0.425 |
| 1 | 多模态 Worker | [01-image-preprocess-offload.md](01-image-preprocess-offload.md) | TTFT ↓ 80ms+ |
| 2 | 多模态 Worker | [02-vision-token-sparsification.md](02-vision-token-sparsification.md) | TTFT ↓ 40ms+ |
| 3 | Prefill / Decode kernel | [03-ascendc-kernels.md](03-ascendc-kernels.md) | 整体 ↓ 120ms+ |
| 4 | 多模态 ViT | [04-vit-fusion.md](04-vit-fusion.md) | TTFT ↓ 10ms+ |
| 5 | Prefill MoE | [05-moe-ag-rs.md](05-moe-ag-rs.md) | TTFT ↓ 15ms |
| 6 | Decode 投机采样 | [06-argmax-forward.md](06-argmax-forward.md) | 计算量缩 1/TP，通信从 hidden 降到 token_id |
| 7 | Prefill APC | [07-fine-grained-apc.md](07-fine-grained-apc.md) | TTFT ↓ 10ms+（TP4，前缀命中 1k） |
| 8 | Prefill 编译 | [08-prefill-inductor.md](08-prefill-inductor.md) | TTFT ↓ 10ms+（不定长并发） |
| 9 | API Server 分词 | [09-fastokens.md](09-fastokens.md) | TTFT ↓ 20ms |
| 10 | RS + P 调度 | [10-slo-predictor-scheduling.md](10-slo-predictor-scheduling.md) | P95/P99 TTFT ↓ 16.4%/28.4%（vs 最小请求数） |
| 11 | P/D KV 调度 | [11-layerwise-cpcd.md](11-layerwise-cpcd.md) | 单并发 TTFT ↓ 70ms+；纯文本再降 20ms+ |
| 12 | P→D KV 数据面 | [12-kv-batch-sync.md](12-kv-batch-sync.md) | TTFT 606→531ms（↓75ms） |
| 13 | P/D KV 控制面 | [13-zmq-control-plane.md](13-zmq-control-plane.md) | 原文在此截断，实现后半待补 |

待补（原文目录有、本次材料未给出完整实现）：

- Decode：MTP 零气泡调度（同步气泡 5ms+ → ~1ms，TPOT ↓ 1ms+）
- 系统稳定性：绑核、GIL 等
- 昇腾差异化优势 / 可复用性 / 附录工程清单
