# 阶段 3：vLLM KV Connector（本周后半 · 3–4 天）

目标：深挖 `MooncakeConnector` 的 P2P PD 直传，能把一次请求从 Scheduler 勾到 TE `WRITE`。`MooncakeStoreConnector` 本阶段只记职责，实现放到 Store 阶段。

---

## 要看的内容

1. [Mooncake × vLLM 集成总览](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/index.html)（本地 `docs/source/deployment/integrations/vllm/index.md`）
2. [PD 分离指南](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/disagg-prefill-decode.html)（本地 `docs/source/deployment/integrations/vllm/disagg-prefill-decode.md`）——只看 V1 用法和配置，不搭集群
3. [vLLM MooncakeConnector 用法](https://docs.vllm.ai/en/latest/features/mooncake_connector_usage/)
4. **主文件通读**：`mooncake-wheel/mooncake/mooncake_connector_v1.py`（约 900 行，本阶段最重要的一份代码）
5. Proxy 参考：`mooncake-wheel/mooncake/vllm_v1_proxy_server.py`（注释写明拷自 vLLM nixl toy proxy）

Store 相关文档本周只读标题和第一段，区分两条产品线即可：

- [KV Cache Storage](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/kv-cache-storage.html) ← 下周再读正文

---

## 要掌握的知识

### 两个 Connector 先分清

| | MooncakeConnector | MooncakeStoreConnector |
|--|-------------------|------------------------|
| 本周深度 | **深挖** | 只记职责 |
| 数据路径 | Prefill Worker **RDMA WRITE** 到 Decode Worker 已注册的 KV buffer | 经 Mooncake Store 的 Put/Get，按 block hash 跨实例复用 |
| 典型角色 | `kv_producer`（P）/ `kv_consumer`（D） | 单机 offload 用 `kv_both`；和 PD 叠用时以后再说 |
| 需要 Store Master 吗 | 否 | 是 |

### 一次 PD 请求的控制面 + 数据面

```text
用户 → Proxy
         ├─ 把请求打到 Prefill，带 do_remote_decode
         └─ Prefill 结束后把 do_remote_prefill + remote_host/port
            交给 Decode

Scheduler（MooncakeConnectorScheduler）
  get_num_new_matched_tokens   # Decode：发现要远程拉整段 prompt KV
  update_state_after_alloc     # 记下 local block_ids 和远端地址
  request_finished             # Prefill：延迟释放 block，把 host/port 传给 Decode
  build_connector_meta         # 打包成 MooncakeConnectorMetadata 给 Worker

Worker（MooncakeConnectorWorker）
  register_kv_caches           # TE.batch_register_memory(KV tensor 基址)
  start_load_kv
      Decode: ZMQ 把 MooncakeAgentMetadata 发给 Prefill
      Prefill: 按 block 算出 src/dst 指针，TE.batch_transfer_sync_write
  get_finished                 # 告诉调度器哪些请求发完/收完
```

必须钉死的三点：

1. **TE 只在 Worker 里创建**。Scheduler 进程不碰 RDMA，只传 metadata。
2. **握手走 ZMQ side channel，KV 走 TE**。`VLLM_MOONCAKE_SIDE_CHANNEL_PORT`（默认 6557）+ tp_rank；TE 自己还有 `get_rpc_port()`。Decode 发给 Prefill 的是「我的 hostname、TE rpc_port、各层 KV base addr、我的 block_ids」。
3. **数据方向是 Prefill WRITE 到 Decode**，不是 Decode READ。见 `send_kv_to_decode` → `_send_blocks` → `engine.batch_transfer_sync_write`。

### vLLM 配置在干什么

Prefill：

```text
--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

Decode：

```text
--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

若用 Mooncake 仓库自带的实现（而不是 vLLM 树内那份），还要加 `"kv_connector_module_path":"mooncake.mooncake_connector_v1"`。

`kv_both`：同一实例既当 P 又当 D，本周知道即可。

环境变量（Worker `__init__` 顶部）：

| 变量 | 默认 | 作用 |
|------|------|------|
| `VLLM_MOONCAKE_PROTOCOL` | `rdma` | 传给 `engine.initialize(..., protocol, "")` |
| `VLLM_MOONCAKE_SIDE_CHANNEL_PORT` | `6557` | ZMQ 握手端口基址 |
| `VLLM_MOONCAKE_SENDER_WORKERS` | `10` | Prefill 侧发送线程池 |

### 和 TE 阶段的接口对照

| Connector 调用 | TE |
|----------------|-----|
| `TransferEngine()` + `initialize(hostname, "P2PHANDSHAKE", protocol, "")` | `init`，P2P 握手，无 etcd |
| `batch_register_memory(kv_data_ptrs, kv_data_lens)` | `registerLocalMemory` |
| `batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)` | `openSegment` + `submitTransfer(WRITE)` |
| `get_rpc_port()` | TE RPC，给对端 `openSegment` 用的名字组成 `host:port` |

`save_kv_layer` / `wait_for_layer_load` 在这份 Connector 里是空的：它不做逐层流水搬运，而是请求粒度整块传。

---

## 关键代码所在的文件

路径相对 `D:\docs\Mooncake`。

### 必读

| 文件 | 盯的类 / 函数 |
|------|----------------|
| `mooncake-wheel/mooncake/mooncake_connector_v1.py` | 整文件。下面按类拆 |
| 同上 · `MooncakeConnector` | 把 vLLM `KVConnectorBase_V1` 的 Scheduler/Worker 方法转发给两个内部类 |
| 同上 · `MooncakeConnectorScheduler` | `get_num_new_matched_tokens`、`update_state_after_alloc`、`request_finished`、`build_connector_meta` |
| 同上 · `MooncakeConnectorWorker` | `__init__`（创建 TE）、`register_kv_caches`、`start_load_kv`、`send_kv_to_decode`、`receive_kv`、`get_finished` |
| 同上 · `MooncakeAgentMetadata` | ZMQ 上序列化的「对端 KV 基址 + block 列表」 |
| 同上 · `MooncakeConnectorMetadata` | Scheduler → Worker 的 `reqs_to_recv` / `reqs_to_send` |
| `mooncake-integration/transfer_engine/transfer_engine_py.h` | `initialize`、`batchTransferSyncWrite`，对上 Worker 里的 Python 调用 |
| `mooncake-wheel/mooncake/vllm_v1_proxy_server.py` | Proxy 如何把一次用户请求拆成 Prefill + Decode，并填 `kv_transfer_params` |

### 对照文档（不是源码，但配置字段在这里）

| 文件 | 看什么 |
|------|--------|
| `docs/source/deployment/integrations/vllm/disagg-prefill-decode.md` | `--kv-transfer-config`、`kv_producer` / `kv_consumer`、toy proxy 命令 |

vLLM 上游的 `KVConnectorBase_V1` 不在本仓库。接口形状已经由 `MooncakeConnector` 的方法名体现：不必去翻 vLLM 也能学完本阶段。

`MooncakeStoreConnector` 的实现在 **vLLM 仓库**，不在 Mooncake 仓库。本周不要去搜它的源码。

---

## 自我验证

1. 一次 PD 请求里，谁在 Scheduler 里、谁在 Worker 里？TE 实例存在哪个进程？
2. `kv_producer` 和 `kv_consumer` 分别对应 Prefill 还是 Decode？`request_finished` 为什么要 `delay_free_blocks`？
3. Decode 的 `get_num_new_matched_tokens` 在 `do_remote_prefill=True` 时返回的 token 数代表什么？第二个 bool 为什么是 True（异步）？
4. ZMQ side channel 传的是 KV 本体还是元数据？列出 `MooncakeAgentMetadata` 的字段，并说明 Prefill 收到后如何构造 TE 的 src/dst。
5. 数据是 Prefill `WRITE` 到 Decode，还是 Decode `READ` Prefill？指出函数名。
6. `register_kv_caches` 里 `block_len = tensor_size_bytes // num_blocks` 之后，第 `i` 个 block 的地址怎么算？
7. `remote_session = f"{hostname}:{rpc_port}"` 里的 port 是 side channel 还是 TE RPC？和 `side_channel_port + tp_rank` 各干什么？
8. 为什么 `save_kv_layer` 是空实现？这和 NIXL 一类 layerwise connector 有什么差别？
9. Proxy 不存在时，Decode 怎么知道 Prefill 的 `remote_host/port`？`request_finished` 返回的 dict 里哪几个 key 是必须的？
10. 用一段话区分：MooncakeConnector 的 P2P 直传 vs MooncakeStoreConnector 的共享池。本周如果被问「prefix 跨实例命中」，你的正确回答是什么（「还没学 / 走 Store」）？
