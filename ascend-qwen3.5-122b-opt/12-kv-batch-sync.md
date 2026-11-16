# KV Cache 批量传输（batch sync）

- 链路：Prefill 每层 forward 之后 → 把该层 KV 写到 Decode（与计算共享 NPU stream）
- 场景：2P1D，P: tp4dp1，D: tp4dp4，输入均长 8k，不定长并发
- 收益：TTFT 606ms → 531ms（↓75ms）
- 开关：`VLLM_CLOUD_KV_BATCH_SIZE`，默认 0=禁用；>0 表示一次 sync 覆盖的层数

开源仓库里**搜不到** `VLLM_CLOUD_KV_BATCH_SIZE`。下面对照的是社区已经公开的同名原语：`Event.record` → 发送线程 `Event.synchronize` → `batch_transfer_sync_write`。社区粒度是**每层 1 次**，不是每 N 层攒一次。

主文件：[vllm-ascend `mooncake_layerwise_connector.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)。Attention 侧 hook 在 [`vllm_ascend/attention/utils.py`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/utils.py)。

## 问题怎么识别

分层传输策略：每层 forward 完立刻传 KV，和计算共用 NPU stream，用 `Event::record` → `Event::sync` 对齐。

profiling 结论：

- 每次 `aclrtSynchronizeEvent` 约 **4ms**
- 阻塞后续 kernel 下发
- **record→record 的 enqueue 间隙占 84.4%** —— 大头不是 DMA 拷贝，是同步把 launch 管道掐断

48 层满同步的话，光 sync 墙钟就不可接受。所以问题定义为 **同步粒度太细**，不是传输协议太慢。

## sync 该怎么理解

名字里有两个「sync」，不是一回事。

**`Event.record()`**：在计算 stream 上钉一个章，意思是「stream 执行到这里时，章被盖上」。CPU 立刻返回，NPU 还在往后算。每层都可以 record，本身几乎不挡人。

**`Event.synchronize()` / `aclrtSynchronizeEvent`（这里说的 sync）**：CPU **坐等这枚章被盖上**。章一盖上，说明 record 之前、同一条 stream 上的 kernel 都算完了——对本路径就是「这层 paged KV 已经写进显存，RDMA 可以读」。

它**不是**：

- 等 Decode 收到了
- 等发送线程和计算线程握手
- 等网络 ACK

发送线程调用 `wait_event.synchronize()`，语义只是：**在确认本卡这层 KV 写完之前，不准发起 `batch_transfer_sync_write`**。否则 DMA 会读到旧数据。

`batch_transfer_sync_write` 名字里的 sync 是另一件事：这次 RDMA 提交在引擎返回前会卡住调用方。优化砍的是前面那个 **Event sync**，不是改 RDMA 协议。

同 stream 是 FIFO：等第 N 层的 event，等于前 N 层的写 KV 都已经完成。所以可以每层都 record（便宜），但只在第 N 层 `synchronize()` 一次，再把 N 层一起传出去。

```text
改前（每层）:
  layer_i 算完 → record_i → synchronize(event_i)  ← CPU 在这空等 ~4ms
               → 传 layer_i → 才能安心下发 layer_{i+1}

改后（每 N 层）:
  layer_i   算完 → 写入 buffer → record_i → 立刻算下一层   （不 synchronize）
  layer_i+1 算完 → 写入 buffer → record_{i+1} → …
  layer_i+N 算完 → 写入 buffer → synchronize(event_{i+N})  ← 只等这一次
               → 一次传这 N 层
```

你的理解对的部分：减的是 **synchronize 次数**，不是 record。中间那几层只打点、不等；攒满 N 层再用最后一枚 event 对齐，然后一批送走。CPU 少被掐几次，后面的计算 kernel 才能连着 enqueue——profiling 里 84.4% 的 record→record 间隙，指的就是被这次等待掐断的下发。

## 具体怎么改实现

Prefill 侧加 KV buffer：每层写入后立刻返回，本层不做 `Event::synchronize`；每 N 层才 sync 一次，再 `batch_transfer_sync_write` 一次送出这 N 层。

N 由 `VLLM_CLOUD_KV_BATCH_SIZE` 控制。N 越大，sync 次数越少、下发越连；但 buffer 更大，D 侧看到 KV 的延迟也更批量化（分层 overlap 的窗口变粗）。原实验在不定长并发下 606→531ms。

## 社区开源实现（每层 1 次，不是每 N 层）

### 调用链

```text
reshape_and_cache 写完本层 paged KV
    → notify_kv_cache_written(layer_name)          # attention 立刻打 Event.record
        → MooncakeLayerwiseConnector.on_kv_cache_written
            → Worker.on_kv_cache_written
                → cache_write_event.record()        # 挂在计算 stream，不 wait

本层 o_proj / attention 收尾
    → maybe_save_kv_layer_to_connector(layer, kv)
        → Connector.save_kv_layer
            → Worker.save_kv_layer
                → 组 SendTask(wait_event=cache_write_event)
                → send_queue.put(task)              # 计算线程到此返回
                → current_layer += 1

KVCacheSendingLayerThread（后台）
    → send_queue.get()
    → _transfer_kv_cache
        → wait_event.synchronize()                  # 这里才 Event::sync
        → engine.batch_transfer_sync_write(...)     # 本层（可含同物理层多个 tensor）一次发出
```

Attention 入口：

- `notify_kv_cache_written`：`reshape_and_cache` 之后立刻调。`attention_v1.py` / MLA / SFA / DSA 都有。
- `maybe_save_kv_layer_to_connector`：本层计算收尾再调（MLA/SFA/DSA 在 `o_proj` 之后）。它只是把 `connector.save_kv_layer` 包一层。

`wait_for_save()` 在 Mooncake layerwise 上是空实现：保存已经按层丢给发送线程，计算路径不再挡一层总 barrier。

### 1. 计算 stream 只 record，不 sync

`Worker.on_kv_cache_written`：

```python
cache_write_event = torch.npu.Event()
cache_write_event.record()
self._cache_write_events[self.current_layer] = cache_write_event
```

`Worker.save_kv_layer` 把这个 event 塞进 `SendTask.wait_event`，`put` 进发送队列后立刻 `current_layer += 1` 返回。计算线程**不**在这里 `synchronize()`。

若 attention 没走到 early hook，`save_kv_layer` 会自己再 `Event()` + `record()`，注释写明：正确性还在，但丢掉和计算的 overlap。

### 2. 发送线程里才 sync，然后 batch write

`KVCacheSendingLayerThread._transfer_kv_cache`（`pd_head_ratio == 1` 的主路径）：

```python
send_task.wait_event.synchronize()   # aclrtSynchronizeEvent
ret = self.engine.batch_transfer_sync_write(
    session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
)
```

`batch` 在这里的意思是：**同一 session 上，本层若干 src/dst/length 描述符一次提交**（多 request、K/V 两块、连续 block 合并）。不是「N 个 transformer 层攒成一次」。

`get_transfer_meta` 会按 `group_concurrent_contiguous` 把连续 block 合成一条 length，再按 session 合并进 `TransferMeta`。

`SendTask.layer_names` 可以带多个名字，但它们都属于 `index_to_name[current_layer]`——同一物理层的多个 tensor（例如 hybrid 里 attn + mamba），不是跨 N 层。

### 3. `k_buffer` / `v_buffer` 不是攒层 buffer

`create_kv_buffer` 只在 `pd_head_ratio > 1`（P/D TP 不等，要 alltoall reshard）或 KV/C8 量化时分配，尺寸是**单层** cache 对齐到 2MB 后注册给 Mooncake TE。

发送线程里的用法是：resharding stream 上 `k_buffer[:].copy_(key)`，再拿 `k_buffer.data_ptr()` 当 RDMA src。TP 相等、无量化时，直接用各层已注册的 `kv_caches_base_addr`，不走这块 staging。

队列深度也说明没有攒层：`pd_head_ratio == 1` 时 `Queue(maxsize=0)`（无界），`put` 不挡计算；`pd_head_ratio != 1` 时 `maxsize=1`，上一层没被发送线程取走，`put` 会堵住下一层。

### 4. 和这篇优化差在哪

| | 社区 Mooncake layerwise | 这篇的 KV batch sync |
|--|-------------------------|----------------------|
| 开关 | `--kv-transfer-config` 选 `MooncakeLayerwiseConnector` | `VLLM_CLOUD_KV_BATCH_SIZE=N`（开源搜不到） |
| Event.record | 每层，计算 stream | 每层写入 buffer |
| Event.sync | **每层**，在发送线程 | **每 N 层**一次 |
| `batch_transfer_sync_write` | 每层 1 次（该层所有 descriptor） | 每 N 层 1 次（N 层 KV） |
| buffer | 单层 staging（reshard/量化） | 多层 KV buffer |

社区已经把 sync 从计算线程挪到 `KVCacheSendingLayerThread`。这篇要再砍的是：**发送侧仍然每层一次 `synchronize()`**。若 `aclrtSynchronizeEvent` 仍干扰 launch 管道（或 `pd_head_ratio != 1` 时 `put` 反压），48 层就会叠出那 4ms × N 和 84.4% enqueue 间隙。解法是延迟同步 + 攒层传输，不是换 RDMA 协议。

### 不要和这两条搞混

1. **上游 Mooncake Cross-layer**（[vllm#41093](https://github.com/vllm-project/vllm/pull/41093)）：一块连续 buffer，**一次传所有层**。粒度是「整网」，不是可调的 N。
2. **AscendStore layerwise pool**：`layerwise_max_transfer_blocks` / `layerwise_max_transfer_bytes` 限制的是单次传输的 block/字节，防止一层太大占满总线；`layerwise_prefetch_layers` 是 Decode 侧预取层数。都不是「每 N 层才 sync」。

## 学习要点

- sync = CPU 等本卡 KV 写完，不是等 Decode 收到。`batch_transfer_sync_write` 是另一次（RDMA）等待。
- 识别手段是 **Event sync 耗时 + enqueue 间隙占比（84.4%）**，直接指向 host/device 同步，而不是 RDMA 带宽。
- 实现是 **延迟同步 + 攒层传输**：record 仍可每层打，synchronize 改成每 N 层一次。
- 社区公开实现已经是「计算只 record、发送线程再 sync + batch write」，但 batch 的单位仍是**一层**。`VLLM_CLOUD_KV_BATCH_SIZE` 是在这之上把 sync/write 粒度从 1 层改成 N 层。
- 和先 P 后 D（[11](11-layerwise-cpcd.md)）正交：那条改调度与首 token 路径，这条改数据面每层 sync 粒度。
