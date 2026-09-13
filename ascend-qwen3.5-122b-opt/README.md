# 昇腾 Qwen3.5-122B 优化方案

只总结五项关键优化，不再展开其余条目。

材料：*ModelArts AI 搜索推理优化系列：Qwen3.5-122B 性能调优实践*（2026-08-03），以及后补的先 D 后 P / 先 P 后 D 时序。每条只记问题怎么识别、具体怎么改实现。

主文：[关键优化.md](关键优化.md)

| # | 优化 | 收益 | 细稿 |
|---|------|------|------|
| 1 | 先 P 后 D 调度 | 单并发 TTFT ↓ 70ms+；纯文本再降 20ms+ | [11-layerwise-cpcd.md](11-layerwise-cpcd.md) |
| 2 | KV batch sync | TTFT 606→531ms（↓75ms） | [12-kv-batch-sync.md](12-kv-batch-sync.md) |
| 3 | 细粒度 APC | TTFT ↓ 10ms+（TP4，前缀命中 1k） | [07-fine-grained-apc.md](07-fine-grained-apc.md) |
| 4 | Prefill Inductor 入图 | 不定长并发 TTFT ↓ 10ms+ | [08-prefill-inductor.md](08-prefill-inductor.md) |
| 5 | MTP 零气泡 | 同步气泡 5ms+→~1ms，TPOT ↓ 1ms+ | [14-mtp-zero-bubble.md](14-mtp-zero-bubble.md) |

场景背景见 [00-overview.md](00-overview.md)。其余早期笔记留在目录里，不纳入本次总结。
