# 阶段 1：概念与动机（2 天）

目标：两天内搞清 PD 分离、KVCache 池化，以及为什么 vLLM 需要独立 KV Connector。Store 只记「后面才学」，不进实现。

本地仓库根目录：`D:\docs\Mooncake`。

---

## 要看的内容

### 第 1 天：架构图 + 论文动机

1. [仓库 README Overview](https://github.com/kvcache-ai/Mooncake#overview)（本地 `README.md`）
2. 架构图：`image/architecture.png`、`image/components.png`
3. 设计概述：`docs/source/design/architecture.md`
4. FAST 2025 论文：[Mooncake: Trading More Storage for Less Computation](https://www.usenix.org/system/files/fast25-qin.pdf)
5. [FAST slides](https://www.usenix.org/system/files/fast25_slides-qin.pdf)

论文只记两件事：

- Prefill 和 Decode 为什么要拆开（计算形态不同、互相抢 GPU）
- KV 从 Prefill 搬到 Decode 为什么不能靠 NCCL/Gloo 凑合（对象大、路径要拓扑感知、要和推理流水重叠）

### 第 2 天：官方中文讨论

按顺序读知乎 1–3，术语看到即可，不追实现：

| # | 主题 | 链接 |
|---|------|------|
| 1 | 架构总览 | https://zhuanlan.zhihu.com/p/705754254 |
| 2 | 调度与 SLO | https://zhuanlan.zhihu.com/p/705910725 |
| 3 | KVCache 池化 | https://zhuanlan.zhihu.com/p/706204757 |

第 4 篇放到 Transfer Engine 阶段。5–7 跳过。

配套官方站：[Mooncake Docs](https://kvcache-ai.github.io/Mooncake/) 只扫首页 Overview，不要点进 Deployment。

---

## 要掌握的知识

学完应能自己画一张图，图上只有这几块：

```text
请求 → Proxy
        ├─ Prefill 实例（算 KV）
        └─ Decode 实例（逐 token 生成）
              ↑
         KV 要搬走
              │
     Transfer Engine（传输平面）
              │
     vLLM KV Connector（把「哪几个 block」翻译成 TE 搬运）
              │
     Mooncake Store（本阶段只标名字：跨实例共享池，稍后学）
```

必须能口述的概念：

| 术语 | 一句话 |
|------|--------|
| Prefill | 一次吃完整 prompt，产出 KV cache，算力密集、吞吐优先 |
| Decode | 按 token 自回归，延迟敏感，不能被 Prefill 抢 GPU |
| PD 分离 | Prefill / Decode 跑在不同实例或节点上，资源按负载比独立扩 |
| XpYd | X 个 Prefill 实例 + Y 个 Decode 实例的拓扑记法，如 1P2D |
| KV Connector | vLLM 的插件接口：调度侧决定「要远程拉/推哪些 block」，Worker 侧真正搬数据 |
| Transfer Engine | Mooncake 的传输库：注册内存、选网卡、RDMA/TCP 批量读写 |
| Mooncake Store | 带 Master 的分布式对象池；和 P2P Connector 不是同一条路 |
| P2P vs 共享池 | Connector 是 Prefill→Decode 直传；Store 是多实例按 hash 复用 prefix KV |

论文里的调度器、early rejection、SLO 预测：**知道存在即可**，本计划不实现、不部署。

---

## 关键代码所在的文件

本阶段以文档为主，源码只用来对上「仓库里有哪些子系统」，不要顺着函数往下钻。

| 文件 | 看什么 |
|------|--------|
| `README.md` | 三块：TE / Store / EP&PG。本计划只用前两块，且 EP/PG 整段跳过 |
| `docs/source/index.md` | 官方文档目录。记住 Getting Started / Design，跳过 Deployment |
| `docs/source/design/architecture.md` | Master 管元数据、数据面走 TE 零拷贝 |
| `CMakeLists.txt` | `WITH_TE`、`WITH_STORE` 对应两个目录 |
| `mooncake-transfer-engine/` | TE 源码根；第 2 阶段再进 |
| `mooncake-wheel/mooncake/mooncake_connector_v1.py` | vLLM Connector 入口；第 3 阶段再进 |
| `mooncake-store/` | Store 源码根；第 4 阶段再进 |

---

## 自我验证

不看书，口头回答。卡壳就回到对应文档，不要往下阶段逃。

1. Prefill 和 Decode 的计算形态差在哪？合在一个 GPU 实例上会互相伤害什么指标？
2. PD 分离之后，Decode 实例开跑之前缺的是什么数据？这块数据大概有多大（可用「长上下文、大模型」量级描述）？
3. `MooncakeConnector` 和 `MooncakeStoreConnector` 各解决哪一类问题？一条请求能不能只走其中一个？
4. Transfer Engine 在这张图里是「调度器」还是「搬运工」？它知不知道 prompt 语义？
5. XpYd 里的 X、Y 分别指什么？为什么说可以按流量比独立扩？
6. 为什么论文强调「用存储换计算」？这里的「存储」主要指 GPU HBM 里的 KV，还是集群里闲置的 DRAM/SSD？
7. 画出：用户请求 → Proxy → Prefill →（什么组件）→ Decode → 回包。标出 KV 走哪条路径。
8. 本计划为什么先学 TE 和 Connector、后学 Store？如果倒过来，会卡在什么概念上？
