# vLLM KV Connector V1：怎么学、有什么接口、怎么自己写一个

目标：不依赖 Mooncake / NIXL 的传输细节，先把 **vLLM V1 的 KV Connector 插件接口** 学透，并能自己挂一个最小实现。

仓库里已有 [Mooncake 学习计划](../mooncake-learning-plan/README.md)。那份计划把 Connector 当成「把 block 列表翻译成 TE 搬运」的一层，**刻意不翻** 上游 `KVConnectorBase_V1`。本笔记补这一层：基类、调度/Worker 分工、最小 DIY。

对照文档：

| 文档 | 看什么 |
|------|--------|
| [01-interfaces.md](01-interfaces.md) | 生命周期、必实现 / 选实现接口、Scheduler ↔ Worker 怎么传元数据 |
| [02-minimal-diy.md](02-minimal-diy.md) | NoOp → Dummy Fill → 磁盘共享 三级最小实现，以及怎么挂进 vLLM |

上游源码（vLLM 仓库，不在本仓库）：

- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` — `KVConnectorBase_V1`
- `vllm/distributed/kv_transfer/kv_connector/factory.py` — 注册 / 动态加载
- `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py` — 官方调试实现（磁盘存取）
- `vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py` — 用假 KV 填 cache，测 Decode
- `vllm/config/kv_transfer.py` — `KVTransferConfig`

---

## Connector 是什么

vLLM 自己的 paged KV 只活在**当前实例的 GPU block pool** 里。一旦 KV 要离开这个池——写到磁盘、写到 CPU、写到另一台 Decode、写进共享对象池——引擎不自己搬，而是调用插件：

```text
Scheduler 决定「这次要远程命中 / 写出哪些 token、哪些 block」
        ↓  KVConnectorMetadata（随 SchedulerOutput 下发）
Worker  在 forward 前后真正 load / save
```

同一套基类覆盖几类产品：

| 产品形态 | 典型实现 | KV 去哪 |
|----------|----------|---------|
| PD 直传 | MooncakeConnector、NixlConnector | Prefill GPU → Decode GPU |
| 共享池 / prefix 复用 | MooncakeStoreConnector、LMCache | 按 block hash Put/Get |
| Offload | SimpleCPUOffloadConnector | GPU ↔ 本机 CPU |
| 调试 / 学习 | ExampleConnector、DecodeBenchConnector | 磁盘 或 填假值 |

**TE / RDMA 不是接口的一部分。** 接口只认「匹配了多少 token、写进哪几个 paged block」。传输库是 Worker 方法内部的实现细节。

---

## 怎么学（建议顺序）

先接口，后某一种传输。倒过来会把 Mooncake 的 ZMQ/TE 误当成 vLLM 的契约。

1. **钉死两个进程、一份元数据。** Scheduler 进程不算 KV、不碰 RDMA；Worker 进程才碰 tensor。两边只能通过 `build_connector_meta` → `bind_connector_metadata` 对话。Python `dict`、模块级 `Map` 跨进程无效。
2. **把 7 个 abstract 方法的调用时机背下来。** 见 [01-interfaces.md](01-interfaces.md)。能口述「一次请求从 waiting 到 finished，引擎调了哪些钩子」再往下。
3. **写一个 NoOp。** 所有 abstract 方法空实现，`get_num_new_matched_tokens` 返回 `(0, False)`。能 `vllm serve` 起来，证明工厂、构造函数、HMA 开关都通了。
4. **看 `DecodeBenchConnector`。** 它几乎是最小「做了事」的例子：Scheduler 谎称 prompt 已在外部命中，Worker 往 block 里填假数。没有网络、没有磁盘，但走完 load 路径。
5. **看 `ExampleConnector`。** 官方 debug 实现：按 prompt hash 把每层 KV 存成 safetensors。这是最小「真的 save + load」。
6. **再学一种真实传输。** PD 直传说 Mooncake / NIXL；共享池说 Store / LMCache。走 [plan/03-vllm-kv-connector.md](../mooncake-learning-plan/plan/03-vllm-kv-connector.md)。

官方文档只当配置手册，不当接口手册。接口以 `base.py` 模块 docstring 为准；API 仍标 experimental。

---

## 一句话对照

| 问题 | 答案 |
|------|------|
| 接口在哪 | `KVConnectorBase_V1`，V0 基类已经是它的别名 |
| 谁必须实现 | 见下一篇的 7 个 `@abstractmethod` |
| 怎么挂自己的类 | `--kv-transfer-config` 里 `kv_connector` + `kv_connector_module_path` |
| 最简可运行 | NoOp + `kv_role=kv_both` + `--disable-hybrid-kv-cache-manager` |
| 最简有数据面 | 仿 DecodeBench：Scheduler 报命中，Worker 写 paged KV |
| 最简有存取 | 仿 ExampleConnector：共享目录 + 按层文件 |

学完应能自己画：

```text
请求进入 Scheduler
  get_num_new_matched_tokens  → 外部还能补多少 token（可 defer / 可 async）
  CacheManager 分配 GPU block
  update_state_after_alloc    → 记下 local block_ids
  build_connector_meta        → 打包本 step 的 load/save 计划（并清空 scheduler 暂存）

Worker 本 step
  bind_connector_metadata
  start_load_kv               → 当前 forward 需要的 KV 必须在此前提交
  wait_for_layer_load         → 分层流水才有用；整请求搬运可空
  模型 forward
  save_kv_layer               → 分层写出；整请求搬运可空
  wait_for_save
  get_finished / get_transfer_results

请求结束
  request_finished            → 要不要 delay_free_blocks，以及回给 Proxy 的 kv_transfer_params
```
