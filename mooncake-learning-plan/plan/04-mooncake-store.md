# 阶段 4：Mooncake Store（随后 5–7 天）

目标：理解 Master 只管元数据、Client 互传数据；回头看 `MooncakeStoreConnector` 如何用 Put/Get 做 prefix cache。跳过 HA、K8s、SSD offload、多租户调优。

---

## 要看的内容

1. [Mooncake Store 设计](https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html)（本地 `docs/source/design/store/mooncake-store.md`）——读 Introduction + Architecture + Client API 概述；遇到 HA / Snapshot / SSD 章节直接跳
2. [Python Store API](https://kvcache-ai.github.io/Mooncake/api-reference/python/mooncake-store.html)
3. Quick Start 里的单机 Put/Get 示例：`docs/source/getting_started/quick-start.md`（`MooncakeDistributedStore.setup` / `put` / `get`）
4. 回头读：[KV Cache Storage & Sharing](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/kv-cache-storage.html)
5. [MooncakeStoreConnector 用法](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
6. [vLLM × Mooncake Store 博客](https://vllm.ai/blog/2026-05-06-mooncake-store)

不要读：`docs/source/deployment/mooncake-store-deployment-guide.md`、K8s、SSD 专章。

---

## 要掌握的知识

### Store 不是 Redis

- Key 由上层（vLLM block hash）给定，value 写入后视为不可变，直到 Remove。
- 定位是 **KV cache 对象池**，不是通用缓存。Get 保证读到完整一致的一版；副本是 best-effort。

### 两个进程角色

| 组件 | 干什么 | 不干什么 |
|------|--------|----------|
| Master（`mooncake_master`） | 对象→切片→segment 的映射、分配、租约、淘汰 | **不搬 KV 字节** |
| Client | 双角色：① 作为调用方 Put/Get；② 作为存储节点贡献一段 DRAM segment，给别人用 TE 读写 | 不自己当中心交换机 |

Client 三种用法（设计文档原话，本阶段记住 embedded 即可）：

- **Embedded**：推理进程 in-process 链入，既发请求也可能贡献 `global_segment_size`
- Dummy / Real：TP 多 rank 时 dummy 转发到本实例唯一 real client（知道即可）
- Standalone store service：独立进程持有内存池（部署向，跳过）

`global_segment_size=0` → 纯客户端；`local_buffer_size=0` → 纯存储端。

### Put / Get 数据路径（必须能画）

**Put**

```text
Client.Put(key, slices)
  → MasterClient.PutStart(key, slice_lengths)   # Master 在某几个 segment 上划片
  → TransferSubmitter 用 TE WRITE
        把本地 slices 写到持有这些 segment 的远端 Client
  → MasterClient.PutEnd(key)                    # 对象对 Get 可见
```

**Get**

```text
Client.Get(key, local_slices)
  → MasterClient.GetReplicaList(key)            # 只要元数据：副本在哪些 buffer
  → TransferSubmitter 用 TE READ
        从远端 Client 的 segment 读到本地已注册 buffer
```

和 Connector 阶段对比：

| | MooncakeConnector | Store Put/Get |
|--|-------------------|---------------|
| 谁决定地址 | Decode 经 ZMQ 把 KV base+block 告诉 Prefill | Master 分配 / 查询 replica descriptor |
| 谁搬数据 | Prefill Worker 的 TE | 发起 Put/Get 的那个 Client 的 TE |
| 生命周期 | 跟这一次 PD 请求 | 对象可被后来的实例按 key 命中 |

### 回头看 MooncakeStoreConnector

职责一句话：**vLLM 用 block hash 当 Store 的 key，命中则跳过对应 prefill 计算，未命中则把算出来的 KV Put 进池。**

本阶段应能解释，不必把 vLLM 里那份 connector 逐行读完（实现在 vLLM 仓库，不在 Mooncake）：

- Scheduler 侧：用 token block hash 问 Master（或 Store API）有没有这些 key
- Worker 侧：每个 GPU rank 内嵌 Mooncake client；GPU KV 仍要 `register` 成 TE buffer，才能 GPUDirect 进出池
- 和 P2P Connector 可叠加（`MultiConnector`）：P 既直传给 D，也写入共享池。知道这层关系即可，不配集群

Python 最小闭环（理解字段，不一定真跑）：

```python
from mooncake.store import MooncakeDistributedStore
store = MooncakeDistributedStore()
store.setup(..., metadata_server="P2PHANDSHAKE", master_server_addr="127.0.0.1:50051", protocol="tcp")
store.put("hello_key", b"...")
store.get("hello_key")
```

`P2PHANDSHAKE` 仍然只服务 TE 元数据；**Master 地址是另一回事**（默认 `50051`）。

---

## 关键代码所在的文件

路径相对 `D:\docs\Mooncake`。跳过 `mooncake-store/include/ha/`、`local_ssd/`、`nvme_kv_*`。

### 必读

| 文件 | 盯什么 |
|------|--------|
| `mooncake-store/include/client_service.h` | `class Client`：`Put` / `Get` / `Remove` 的对外语义 |
| `mooncake-store/include/master_client.h` | Client→Master RPC：`PutStart` / `PutEnd` / `GetReplicaList` |
| `mooncake-store/include/master_service.h` | Master 进程侧：集群资源池、分配与元数据（文件很大，只搜 Put/Get 相关方法名） |
| `mooncake-store/include/replica.h` | 副本描述符：Get 之后 TE 该写/读哪段 buffer |
| `mooncake-store/include/types.h` | `ObjectKey`、`Slice`、`ErrorCode`、`ReplicateConfig` |
| `mooncake-store/include/segment.h` | Store 视角的 segment（Client 贡献的那块池） |
| `mooncake-store/include/transfer_task.h` | `TransferSubmitter`：把 replica descriptor + slices 变成 TE 请求 |
| `mooncake-store/src/transfer_task.cpp` | `submit` / `submitTransferEngineOperation`：真正调用 `engine_.submitTransfer` |
| `mooncake-store/include/pyclient.h` | Python / dummy-real 那一层的 C++ 包装入口 |
| `mooncake-integration/store/store_py.cpp` | `py::class_<MooncakeStorePyWrapper>(m, "MooncakeDistributedStore")` |

### Connector 对照（只读文档 + 点到为止）

| 位置 | 作用 |
|------|------|
| `docs/source/deployment/integrations/vllm/kv-cache-storage.md` | vLLM 如何把 Store 配成 `MooncakeStoreConnector` |
| `mooncake-wheel/mooncake/mooncake_connector_v1.py` | **不要和 StoreConnector 搞混**；这是 P2P Connector。对照完「它为什么不调 Put/Get」即可 |

---

## 自我验证

1. Master 宕了但两个 Client 还在：已经 Put 成功的对象，还能 Get 到吗？新的 Put 还能分配空间吗？由此说明 Master 在数据面上的位置。
2. 画出 Put 的三步（Start / TE WRITE / End）。若 TE WRITE 成功但 `PutEnd` 没调用，Get 会怎样？
3. Get 时 Client 已经知道 key，为什么还要先问 Master，而不是直接连某个 Peer？
4. `TransferSubmitter` 和 `MooncakeConnectorWorker._send_blocks` 都调用 TE。它们的「远端地址」分别从哪来？
5. `global_segment_size` 和 `local_buffer_size` 各控制什么？纯客户端该怎么设？
6. 为什么说 Store 的 value 写入后不可变？这对 vLLM prefix cache 的 key 设计意味着什么？
7. MooncakeStoreConnector 用什么当 Store 的 key？两个 vLLM 进程 `PYTHONHASHSEED` 不一致时会发生什么（文档里的坑）？
8. P2P Connector 的一次 PD 直传，会不会自动把 KV 写进 Store？要跨实例复用 prefix，缺了哪一步？
9. `setup(..., metadata_server="P2PHANDSHAKE", master_server_addr="127.0.0.1:50051")` 这两个地址为什么都要？能不能合成一个？
10. 用一段话讲完：用户第二条带相同 system prompt 的请求进来后，StoreConnector 如何让 Prefill 少算、MooncakeConnector 又如何把（可能不完整的）KV 交给 Decode。
