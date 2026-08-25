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

---

## 阶段总览

| 阶段 | 时长 | 你要带走的能力 |
|------|------|----------------|
| 概念与动机 | 2 天 | 两天内搞清 PD 分离、KVCache 池化，以及为什么 vLLM 需要独立 KV Connector |
| Transfer Engine | 本周前半 · 3–4 天 | 掌握 Segment / Buffer / BatchTransfer，能把一次 KV 搬运对应到 TE API |
| vLLM KV Connector | 本周后半 · 3–4 天 | 深挖 MooncakeConnector 的 P2P PD 直传；StoreConnector 只记职责 |
| Mooncake Store | 随后 5–7 天 | 理解 Master 只管元数据、Client 互传数据；回头看 StoreConnector 如何做 prefix cache |

---

## 1. 概念与动机（2 天）

本地先看：`README.md` · `docs/source/design/architecture.md` · `FAST25-release/`

### 待办

- [ ] 第 1 天：读 README Overview + architecture.md，画出 Prefill / Decode / TE / Connector 关系（Store 先只记「后面才学」）
- [ ] 第 1 天：扫 FAST 论文与 slides，只记两件事：PD 分离解决什么、KV 传输为什么不能靠 NCCL/Gloo
- [ ] 第 2 天：读知乎 1–3（架构、调度、KVCache），术语看到 Segment / XpYd 即可，不深挖实现

### 必读

- [官方总览](https://kvcache-ai.github.io/Mooncake/)
- [FAST 论文](https://www.usenix.org/system/files/fast25-qin.pdf)
- [FAST slides](https://www.usenix.org/system/files/fast25_slides-qin.pdf)
- [知乎系列 1](https://zhuanlan.zhihu.com/p/705754254)

---

## 2. Transfer Engine（本周前半 · 3–4 天）

本地先看：`mooncake-transfer-engine/include/transfer_engine.h` · `src/transport/` · `example/transfer_engine_bench.cpp`

### 待办

- [ ] 读 design/transfer-engine：Segment、BatchTransfer、拓扑选路，对照知乎第 4 篇
- [ ] 对照 `transfer_engine.h` 走一遍 `init` / `registerMemory` / `openSegment` / `submitTransfer`
- [ ] 看 RdmaTransport vs TcpTransport，弄清 GPUDirect 何时出现在 PD 路径上
- [ ] 能口述即可：一次 BatchTransfer READ/WRITE 从哪段内存到哪段内存（不必跑满 bench）

### 必读

- [TE 设计文档](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html)
- [Python TE API](https://kvcache-ai.github.io/Mooncake/api-reference/python/transfer-engine.html)
- [知乎第 4 篇](https://zhuanlan.zhihu.com/p/707997501)

---

## 3. vLLM KV Connector（本周后半 · 3–4 天）

本地先看：`docs/source/deployment/integrations/vllm/` · `mooncake-integration/`

### 待办

- [ ] 读 vLLM 集成总览，分清 MooncakeConnector（P2P PD）与 MooncakeStoreConnector（共享池）
- [ ] 跟一条 PD 路径：prefiller 产出 KV → TE 传到 decoder → decode 开始
- [ ] 对照 TE API，标出 Connector 在哪一步 `registerMemory` / `submitTransfer`
- [ ] StoreConnector 只记一句话：用 Store 做跨实例 prefix 复用；实现放到 Store 阶段再看

### 必读

- [vLLM 集成总览](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/index.html)
- [PD 分离指南](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/disagg-prefill-decode.html)
- [vLLM Connector 用法](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/)

---

## 4. Mooncake Store（随后 5–7 天）

本地先看：`mooncake-store/include/` · `src/master*.cpp` · `docs/source/design/store/mooncake-store.md`

### 待办

- [ ] 读 Store 设计：Master vs Client 双角色；跳过 HA / K8s / SSD 部署段落
- [ ] 跟一次 Put：Client 向 Master 申请切片 → TE 写入远端 segment
- [ ] 跟一次 Get：Query 元数据 → BatchTransfer READ → 本地 buffer
- [ ] 回头读 MooncakeStoreConnector：block hash 查 Master，hit 跳过 prefill，miss 写入池

### 必读

- [Store 设计](https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html)
- [Python Store API](https://kvcache-ai.github.io/Mooncake/api-reference/python/mooncake-store.html)
- [StoreConnector](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
- [vLLM 官方博客](https://vllm.ai/blog/2026-05-06-mooncake-store)

---

## 本地仓库该从哪读

| 目录 | 角色 | 学习时盯什么 |
|------|------|----------------|
| `README.md` · `docs/source/design/` | 概念 | 两天内只读架构，不进部署目录 |
| `mooncake-transfer-engine/` | 传输层 | Segment、Transport、RDMA/TCP；Connector 的地基 |
| `docs/source/deployment/integrations/vllm/` | vLLM 胶水 | 本周后半读 PD / MooncakeConnector |
| `mooncake-integration/` | 框架封装 | 对照 Connector 调用 TE 的 Python 层 |
| `mooncake-store/` | 分布式 KV 池 | TE + Connector 之后再学；跳过部署文档 |

---

## 可参考的外部材料

| 材料 | 为什么有用 | 出处 |
|------|------------|------|
| [官方文档 toctree](https://kvcache-ai.github.io/Mooncake/) | 概念 + TE + vLLM 集成即可，跳过 Deployment / K8s | `docs/source/index.md` |
| [官方知乎系列 1–4](https://zhuanlan.zhihu.com/p/705754254) | 两天读 1–3，TE 阶段补第 4 篇 | README.md Updates |
| [FAST 2025 论文 / slides](https://www.usenix.org/system/files/fast25-qin.pdf) | 概念阶段扫一遍动机即可 | README 顶部 Paper / Slides |
| [vLLM Connector 用法](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/) | 本周后半主文档，盯 MooncakeConnector | vLLM 官方文档 |
| [vLLM Store 博客](https://vllm.ai/blog/2026-05-06-mooncake-store) | 放到 Store 阶段，用来接上 StoreConnector | vLLM 官方博客 |

---

## 官方知乎系列怎么用

概念只有两天：读完 1–3 就停。第 4 篇跟 Transfer Engine 一起读。5–7 以及部署类文档整段跳过。

| # | 何时读 | 链接 |
|---|--------|------|
| 1 | 概念第 2 天 | [架构总览](https://zhuanlan.zhihu.com/p/705754254) |
| 2 | 概念第 2 天 | [调度与 SLO](https://zhuanlan.zhihu.com/p/705910725) |
| 3 | 概念第 2 天 | [KVCache 池化](https://zhuanlan.zhihu.com/p/706204757) |
| 4 | TE 前半周 | [传输与工程](https://zhuanlan.zhihu.com/p/707997501) |
| 5–7 | 跳过 | 不在本计划内 |
