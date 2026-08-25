# Mooncake 功能学习计划

压缩后的路线：**概念 2 天 → Transfer Engine 与 vLLM KV Connector 共 1 周 → 再学 Mooncake Store**。

不学部署、K8s、SSD、TENT、EP/PG。

本地源码：`D:\docs\Mooncake`（[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)）。

| 指标 | 安排 |
|------|------|
| 学习阶段 | 3 段（概念 / TE+Connector / Store） |
| 建议周期 | 约 2.5 周 |
| 概念与动机 | 2 天 |
| TE + Connector | 共 1 周 |

**顺序说明：** Connector 放在 Store 之前。本周先打通 `MooncakeConnector` 的 PD 直传；`MooncakeStoreConnector` 只记职责，等学完 Store 的 Put/Get 再回头看共享池。

每个阶段的「要看的内容 / 要掌握的知识 / 关键代码 / 自测题」在 `plan/` 里：

| 阶段 | 时长 | 文档 |
|------|------|------|
| 概念与动机 | 2 天 | [plan/01-concepts.md](plan/01-concepts.md) |
| Transfer Engine | 本周前半 · 3–4 天 | [plan/02-transfer-engine.md](plan/02-transfer-engine.md) |
| vLLM KV Connector | 本周后半 · 3–4 天 | [plan/03-vllm-kv-connector.md](plan/03-vllm-kv-connector.md) |
| Mooncake Store | 随后 5–7 天 | [plan/04-mooncake-store.md](plan/04-mooncake-store.md) |

学完一个阶段，先把该文档底部的自测题口头答完，再进入下一阶段。
