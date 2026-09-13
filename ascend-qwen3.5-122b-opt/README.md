# 昇腾 Qwen3.5-122B 优化方案

五项关键优化单独成文；其余特性只记在 [00-overview.md](00-overview.md)。

主文：[关键优化.md](关键优化.md)

| # | 优化 | 收益 | 细稿 |
|---|------|------|------|
| 1 | 先 P 后 D 调度 | 单并发 TTFT ↓ 70ms+；纯文本再降 20ms+ | [11-layerwise-cpcd.md](11-layerwise-cpcd.md) |
| 2 | KV batch sync | TTFT 606→531ms（↓75ms） | [12-kv-batch-sync.md](12-kv-batch-sync.md) |
| 3 | 细粒度 APC | TTFT ↓ 10ms+（TP4，前缀命中 1k） | [07-fine-grained-apc.md](07-fine-grained-apc.md) |
| 4 | Prefill Inductor 图级编译优化 | 不定长并发 TTFT ↓ 10ms+ | [08-prefill-inductor.md](08-prefill-inductor.md) |
| 5 | MTP 零气泡 | 同步气泡 5ms+→~1ms，TPOT ↓ 1ms+ | [14-mtp-zero-bubble.md](14-mtp-zero-bubble.md) |
