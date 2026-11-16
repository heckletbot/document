# KV Connector V1 接口

源码：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`。

引擎会为**同一个类**构造两次：一次 `role=SCHEDULER`（调度进程），一次 `role=WORKER`（每个 Worker 进程）。常见写法是外壳类按 role 把调用转给内部的 `*Scheduler` / `*Worker`，Mooncake、DecodeBench、NIXL 都是这个结构。

```text
KVConnectorFactory.create_connector(config, role, kv_cache_config)
        │
        ├─ Scheduler 进程：只跑 scheduler-side 方法，产出 KVConnectorMetadata
        └─ Worker 进程：只跑 worker-side 方法，搬 tensor
```

---

## 必须实现的 7 个方法

`@abstractmethod`，漏一个不能实例化。

### Scheduler 侧（3）

| 方法 | 引擎在问什么 | 最小合法返回 |
|------|----------------|--------------|
| `get_num_new_matched_tokens(request, num_computed_tokens) -> (int \| None, bool)` | 在本地已经算过 / 命中的 token 之外，外部 KV 还能再补多少 | `(0, False)` |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | CacheManager 已经把 GPU block 分好了，把要 load 的请求记下来 | `pass` |
| `build_connector_meta(scheduler_output) -> KVConnectorMetadata` | 本 step 打包给 Worker 的计划；**调用后应清空 scheduler 暂存** | 空的 metadata 子类实例 |

`get_num_new_matched_tokens` 的契约最容易写错：

- 第一个返回值是 **「比 `num_computed_tokens` 多出来」的 token 数**，不是外部缓存的总长。
- 只统计**当前确实能 load 的最长 prompt 前缀**。连不上、被淘汰的 token 不能算命中。
- `None` = 这次还不知道，调度器稍后会再问（defer）。
- 第二个 `bool`：`True` = 异步 load（跨 step）；返回 0 个 token 时必须是 `False`。
- 该方法可能对同一请求调用多次，**应无副作用**。真正记下 block 的地方是 `update_state_after_alloc`。
- 若上次返回了异步 load，`update_state_after_alloc` 可能被调两次：先为异步 load 分配 block，完成后再为后续计算分配。

`update_state_after_alloc` 看不看 `num_external_tokens`，不要看 `blocks` 是否为空。`MultiConnector` 里落选的子 connector 仍会收到真实 block，但 `num_external_tokens == 0`，这时**不要 load**。

`build_connector_meta` 不要改 `scheduler_output`（文档里唯一例外：同步 READ 且本 step 会整块覆盖的 attention block，可从 `new_block_ids_to_zero` 拿掉）。

### Worker 侧（4）

| 方法 | 引擎在问什么 | 最小合法行为 |
|------|----------------|--------------|
| `start_load_kv(forward_context, **kwargs)` | forward 前开始把外部 KV 写入 paged buffer。当前 forward 要用的必须先提交；其它异步 load 可以随后再发 | `pass` |
| `wait_for_layer_load(layer_name)` | 进入该层 attention 前，这一层的 load 必须完成 | `pass` |
| `save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)` | 该层算完，可以把这一层 KV 往外写（可异步） | `pass` |
| `wait_for_save()` | 本 step 所有 forward 之后提交写出并做必要同步；空 step 也会调。没活就直接返回 | `pass` |

分层接口（`wait_for_layer_load` / `save_kv_layer`）是给 **layerwise 流水** 用的。Mooncake P2P 整请求搬运时这两个是空的：它在 `start_load_kv` 里一次 WRITE 完。若你真的在 attention 层里做异步拷贝，通常还要让 `requires_piecewise_for_cudagraph` 返回 `True`，否则 CUDA Graph 回放会跳过 Python 同步。

---

## 元数据三类

| 类型 | 方向 | 作用 |
|------|------|------|
| `KVConnectorMetadata` | Scheduler → Worker | 本 step 的 load/save 计划。`build_connector_meta` 生产，model runner 在 forward 前 `bind_connector_metadata`，forward 后 `clear_connector_metadata` |
| `KVConnectorWorkerMetadata` | Worker → Scheduler | Worker 回传；多 Worker 时用 `aggregate()` 合成一份再交给 scheduler |
| `KVConnectorHandshakeMetadata` | P/D Worker 带外握手 | 进程起来时交换地址 / rkey / 端口，不走每 step 的 meta |

自己实现时至少要有一个空的 `KVConnectorMetadata` 子类，否则 `build_connector_meta` 没东西可返回。Worker 里用 `_get_connector_metadata()` 取，不要自己藏一份。

---

## 一次请求的调用顺序

```text
Scheduler.schedule()
  │
  ├─ (新请求) on_new_request                  # 可选
  ├─ get_num_new_matched_tokens               # 可返回 None 推迟
  ├─ KVCacheManager 分配 GPU blocks
  ├─ update_state_after_alloc
  └─ build_connector_meta  ──►  随 SchedulerOutput 到 Worker

Worker.forward
  │
  ├─ bind_connector_metadata
  ├─ register_kv_caches                       # 进程生命周期内，不是每 step
  ├─ handle_preemptions                       # 若有抢占 / 驱逐
  ├─ start_load_kv
  ├─ 对每一层: wait_for_layer_load → attention → save_kv_layer
  ├─ wait_for_save
  ├─ get_transfer_results / get_finished
  └─ clear_connector_metadata

Scheduler 收到 KVConnectorOutput
  ├─ update_connector_output                  # 可选
  └─ 请求结束: request_finished
         返回 (delay_free_blocks, kv_transfer_params)
```

PD 分离时，Prefill 的 `request_finished` 常返回 `delay_free_blocks=True`，并带上 `remote_host` / `remote_port` / `remote_block_ids` 这类 `kv_transfer_params`。Proxy 把这份 dict 贴到 Decode 请求上；Decode 的 `get_num_new_matched_tokens` 看到 `do_remote_prefill=True` 才去拉。

---

## 常用可选方法（按需要实现）

不是 abstract，默认空。真做传输时这几条几乎总会碰到。

### Worker

| 方法 | 什么时候要实现 |
|------|----------------|
| `register_kv_caches(kv_caches)` | RDMA / NIXL / Mooncake：把每层 KV tensor 基址注册给传输引擎 |
| `get_finished(finished_req_ids) -> (sending_ids, recving_ids)` | 异步 save/load。返回的 sending id 必须属于某次传入的 `finished_req_ids` |
| `get_transfer_results(...)` | 新接口：一次给出 send 完成、recv 完成、recv 失败。默认转调 `get_finished` |
| `get_block_ids_with_load_errors()` | load 失败的 block。异步路径最迟要在该请求出现在 `finished_recving` 的那一轮报出来 |
| `handle_preemptions(meta)` | 异步 save 时，block 被抢占覆盖前把没写完的处理掉 |
| `shutdown()` | Worker 退出前排空异步操作 |
| `get_handshake_metadata()` / `set_xfer_handshake_metadata(...)` | P/D 带外握手 |
| `build_connector_worker_meta()` | 需要 Worker→Scheduler 回传时 |
| `kv_cache_access()`（classmethod） | 默认 `IN_PROCESS`。要用 RDMA 注册内存则返回 `REGISTERS`，引擎会按「每段单一物理分配」来分配 KV |

### Scheduler

| 方法 | 什么时候要实现 |
|------|----------------|
| `request_finished(request, block_ids) -> (bool, dict \| None)` | PD producer：`True` 表示你暂时接管这些 block，直到 `get_finished` 报发送完成。dict 是给下游 Decode / Proxy 的 `kv_transfer_params` |
| `update_connector_output(output)` | 根据 Worker 的完成 / 失败集合更新本地状态 |
| `take_events()` | 对外发 KV cache 事件（命中、淘汰） |
| `may_load_async(request)` | 默认看 `do_remote_prefill`。和 `--max-num-inflight-kv-loads` 一起决定能不能在 running batch 已满时先发起 load |
| `has_pending_push_work()` | Prefill 还在等把 KV WRITE 到 Decode 时返回 True，让主循环继续转 |
| `get_loaded_kv_cache_group_ids(request)` | 默认 prefix-cacheable groups。PD 直传通常覆盖全部 transfer groups |
| `get_required_kvcache_layout(vllm_config)` | 需要固定 `HND` / `NHD` 时 |

异步 save 的 block **在 `get_finished` 报完成之前不能被引擎回收**。这就是 `request_finished` 返回 `True`（delay free）的原因。

---

## 配置：`KVTransferConfig`

命令行 `--kv-transfer-config` 是一段 JSON，对应 `vllm/config/kv_transfer.py`。

| 字段 | 作用 |
|------|------|
| `kv_connector` | 类名。内置走 registry；配了 `kv_connector_module_path` 则按这个名字从模块里 `getattr` |
| `kv_connector_module_path` | 动态 import 外部模块，优先于内置表 |
| `kv_role` | `kv_producer` / `kv_consumer` / `kv_both`。设了 connector 就必须设 role |
| `kv_connector_extra_config` | 插件私有 dict，用 `self._kv_transfer_config.get_from_extra_config(key, default)` 读 |
| `engine_id` | 实例身份，默认 uuid |
| `kv_port` | 部分 connector 的监听端口，默认 14579 |
| `kv_load_failure_policy` | `fail`（默认）或 `recompute` |

`kv_role` 怎么选：

| role | 典型场景 |
|------|----------|
| `kv_producer` | Prefill：算完往外推 |
| `kv_consumer` | Decode：启动前从外拉 |
| `kv_both` | 单实例 offload / prefix 共享 / 学习用的 ExampleConnector、DecodeBench |

工厂还强制一件事：**默认开着 Hybrid KV Cache Manager (HMA)**。connector 若没混入 `SupportsHMA`（并实现 `request_finished_all_groups`），启动会直接报错。学习阶段最省事的办法是命令行加 `--disable-hybrid-kv-cache-manager`。

构造函数签名必须是：

```python
def __init__(self, vllm_config, role, kv_cache_config):
    super().__init__(vllm_config, role, kv_cache_config)
```

缺 `kv_cache_config` 的旧两参数签名会被工厂拒绝。

---

## 内置 connector 对照（学习时选谁看）

`factory.py` 末尾的 registry。只挑几条建立地图，不必全读。

| 名字 | 适合用来学什么 |
|------|----------------|
| `ExampleConnector` | save + load 的最小真实路径（磁盘） |
| `DecodeBenchConnector` | 只有 load 计划、Worker 填假 KV；看 meta 怎么从 Scheduler 走到 Worker |
| `MooncakeConnector` | PD P2P：ZMQ 握手 + TE WRITE |
| `NixlConnector` / Pull / Push | 另一套 PD；能对照 pull vs push |
| `MooncakeStoreConnector` / `LMCacheConnectorV1` | 按 hash 的共享池 |
| `SimpleCPUOffloadConnector` | 本机 GPU↔CPU，还示范了 HMA |
| `MultiConnector` | 一个实例同时挂直传 + 共享池 |

自己的类**不必**改 registry。外部模块用 `kv_connector_module_path` 即可。

---

## 和传输库的边界

Connector 接口保证的是：

1. Scheduler 选出要 load/save 的 token / block；
2. 这些信息能完整到达 Worker；
3. Worker 在 forward 前后把 paged KV 写成对端期望的布局；
4. 异步完成通过 `get_finished` 回到 Scheduler，决定何时释放 block。

它**不保证**：用 RDMA 还是 TCP、谁 READ 谁 WRITE、要不要 etcd、layerwise 还是整请求。那些是 Worker 内部的选择。所以最小实现可以是 `pass`，也可以是 `tensor.copy_`，不必先有 Transfer Engine。
