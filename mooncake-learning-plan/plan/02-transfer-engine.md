# 阶段 2：Transfer Engine（本周前半 · 3–4 天）

目标：掌握 Segment / Buffer / BatchTransfer，能把一次 KV 搬运对应到 TE API。这是 Connector 的地基。不学 TENT、不学 EP/PG、不跑满集群 bench。

---

## 要看的内容

按这个顺序，不要一上来扎进全部 Transport 实现：

1. 设计文档：[Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html)（本地 `docs/source/design/transfer-engine/index.md`）
2. 知乎第 4 篇：[传输与工程](https://zhuanlan.zhihu.com/p/707997501)
3. Python API：[Transfer Engine Python](https://kvcache-ai.github.io/Mooncake/api-reference/python/transfer-engine.html)
4. 头文件走读（见下一节关键文件）
5. 示例：`mooncake-transfer-engine/example/transfer_engine_bench.cpp` 里 `registerLocalMemory` → `allocateBatchID` → `submitTransfer` → `getTransferStatus` 这一段

协议清单 `docs/source/getting_started/supported-protocols.md` 扫一眼即可：本阶段只需分清 **RDMA（主路径）** 和 **TCP（无 RDMA 时的回退）**。

---

## 要掌握的知识

### 核心抽象

| 抽象 | 含义 |
|------|------|
| Segment | 一块可被远程读写的地址空间。每个进程通常有且仅有一个以 `local_hostname` 命名的 RAM Segment |
| Buffer | Segment 里真正注册出去的连续区间（同一设备、同一权限）。`BatchTransfer` 的每一笔必须落在某个 Buffer 内 |
| BatchTransfer | 一组 READ/WRITE 请求：本地若干不连续区间 ↔ 远端 Segment 若干区间，异步提交、轮询完成 |
| Transport | 真正干活的后端：`RdmaTransport`、`TcpTransport` 等。TE 对外统一 API，对内按协议分发 |

一次完整搬运的心智模型：

```text
init(metadata, local_server_name)
  → registerLocalMemory(本地 KV / DRAM)
  → openSegment(对端 hostname)          # 拿到 SegmentID
  → allocateBatchID(N)
  → 填 TransferRequest{opcode, source, target_id, target_offset, length}
  → submitTransfer(batch_id, requests)
  → getTransferStatus / getBatchTransferStatus 直到 COMPLETED
```

### 必须分清的细节

- **READ vs WRITE 的主语是本地**：`WRITE` 是把本地 `source` 写到远端 `target_id + target_offset`；`READ` 是从远端读到本地。
- **元数据服务 ≠ 数据面**。etcd / HTTP / Redis / `P2PHANDSHAKE` 只交换 Segment、rkey、拓扑；payload 不走 Master。vLLM Connector 用的是 `P2PHANDSHAKE`，不单独起 etcd。
- **拓扑选路**：多 NIC 时按 NUMA / GPU 亲和选 preferred HCA；大块会切 slice，多网卡并行。失败会换备选路径。
- **注册才可远程访问**：没 `registerLocalMemory` 的地址，对端 `openSegment` 也写不进去。GPU 路径通常还要 GPUDirect。
- **Python 层是薄封装**：`mooncake.engine.TransferEngine.initialize` / `batch_register_memory` / `batch_transfer_sync_write` 最终落到同一套 C++ API。

能口述即可，不必自己实现 RDMA QP。

---

## 关键代码所在的文件

路径均相对 `D:\docs\Mooncake`。

### 必读（按走读顺序）

| 文件 | 盯什么 |
|------|--------|
| `mooncake-transfer-engine/include/transfer_engine.h` | 对外 API：`init`、`openSegment`、`registerLocalMemory`、`allocateBatchID`、`submitTransfer`、`getTransferStatus` |
| `mooncake-transfer-engine/include/transport/transport.h` | `TransferRequest`（`opcode/source/target_id/target_offset/length`）、`TransferStatusEnum` |
| `mooncake-transfer-engine/include/transfer_metadata.h` | 集群里如何发现对端 Segment / Buffer / handshake |
| `mooncake-transfer-engine/include/topology.h` | `TopologyEntry.preferred_hca`：为什么不是随便挑一张网卡 |
| `mooncake-transfer-engine/include/multi_transport.h` | 多协议时请求如何分发给具体 Transport |
| `mooncake-transfer-engine/src/transfer_engine.cpp` | `TransferEngine` 对 `TransferEngineImpl` 的转发 |
| `mooncake-transfer-engine/include/transfer_engine_impl.h` | 真正的引擎状态（segment 表、transport 安装） |
| `mooncake-integration/transfer_engine/transfer_engine_py.h` | Python 看见的名字：`initialize`、`batchTransferSyncWrite/Read` |
| `mooncake-integration/transfer_engine/transfer_engine_py.cpp` | `PYBIND11_MODULE(engine, …)`，对应 `from mooncake.engine import TransferEngine` |
| `mooncake-transfer-engine/example/transfer_engine_bench.cpp` | 最短可跟的调用序列 |

### 选读（建立「RDMA 和 TCP 差在哪」）

| 文件 | 盯什么 |
|------|--------|
| `mooncake-transfer-engine/include/transport/rdma_transport/rdma_transport.h` | RDMA 路径入口 |
| `mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp` | `registerLocalMemoryInternal`、`submitTransferTask` |
| `mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp` | 无 RDMA 时的回退；理解后就知道 Connector 里 `VLLM_MOONCAKE_PROTOCOL=rdma\|tcp` 改的是什么 |

不要进 `mooncake-transfer-engine/tent/`（下一代引擎，本计划跳过）。

---

## 自我验证

1. 用自己的话定义 Segment 和 Buffer。一次 `submitTransfer` 里的 `source` 和 `target_offset` 分别必须落在哪里？
2. 写出 `TransferRequest` 的五个核心字段，并解释 `WRITE` 时数据从哪到哪。
3. `openSegment("192.168.0.3:12345")` 成功意味着什么？它会不会把 KV 拷过来？
4. 为什么必须先 `registerLocalMemory` 再让对端来写？不注册会失败在哪一层？
5. `P2PHANDSHAKE`、etcd、HTTP 三种 metadata 后端的共同点是什么？数据面走它们吗？
6. 从 Python `engine.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)` 往下，对应到 C++ 的哪几个调用？
7. 一台机器两张 RDMA 网卡、两块 GPU。TE 根据什么决定用 `mlx5_0` 还是 `mlx5_1`？相关结构在哪个头文件？
8. TCP Transport 和 RDMA Transport 对「本地 VRAM ↔ 远端 DRAM」的支持差异是什么？这如何影响无 RDMA 环境的学习方式？
9. `allocateBatchID` 和 `freeBatchID` 为什么成对出现？`getTransferStatus` 返回 `PENDING` 和 `COMPLETED` 时调用方该怎么做？
10. 若一次搬运 128 个不连续 KV block，你会构造 128 个 `TransferRequest` 还是 1 个？Batch 的意义是什么？
