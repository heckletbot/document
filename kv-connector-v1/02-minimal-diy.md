# 自己实现一个最简单的 KV Connector

分三级。每一级都能单独跑；不要一上来写 RDMA。

| 级别 | 做什么 | 证明什么 |
|------|--------|----------|
| 0 NoOp | 全部空实现 | 工厂能加载、进程能起来 |
| 1 Dummy Fill | Scheduler 报「prompt 已命中」，Worker 往 block 填常数 | meta 通道通了，load 路径能改 paged KV |
| 2 共享目录 | 按 prompt 前缀把每层 KV 写成文件，下次读回来 | 真的有 save + load；仍不需要网络 |

三级共用一件事实：**Scheduler 进程和 Worker 进程不共享 Python 内存。** 级别 0/1 只通过 `KVConnectorMetadata` 传计划；级别 2 用磁盘当双方都看得到的存储。不要在外壳类里放一个 `self.cache = {}` 指望 Prefill 写入、Decode 读出。

---

## 怎么挂进 vLLM

把模块放到 `PYTHONPATH` 上，类名和配置里的 `kv_connector` **完全一致**：

```text
--kv-transfer-config '{
  "kv_connector": "NoOpConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "my_connectors.noop"
}'
--disable-hybrid-kv-cache-manager
```

等价 Python：

```python
from vllm.config.kv_transfer import KVTransferConfig

KVTransferConfig(
    kv_connector="NoOpConnector",
    kv_role="kv_both",
    kv_connector_module_path="my_connectors.noop",
)
```

学习阶段加上 `--disable-hybrid-kv-cache-manager`，否则工厂会要求 `SupportsHMA`。以后真要跟 hybrid / Mamba 一起用，再混入 `SupportsHMA` 并实现 `request_finished_all_groups`。

外部模块优先于内置表。类名如果和内置撞车（例如也叫 `LMCacheConnectorV1`），新一点的 vLLM 用 `kv_connector_class_name` 做真正的 getattr 名，`kv_connector` 只当逻辑别名。

---

## 级别 0：NoOp

文件 `my_connectors/noop.py`。这是能通过 abstract 检查的最短骨架。

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class NoOpMetadata(KVConnectorMetadata):
    pass


class NoOpConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

    # ----- Worker -----
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return

    # ----- Scheduler -----
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        return

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return NoOpMetadata()
```

启动后日志应有 `Creating v1 connector with name: NoOpConnector`。行为与没开 connector 几乎一样：不报外部命中、不搬数据。若这里就起不来，先查类名、模块路径、三参数构造函数、HMA 开关，不要怀疑接口本身。

---

## 级别 1：Dummy Fill（推荐作为「最小有行为」）

思路抄 `DecodeBenchConnector`，但剥到只剩一件事：

1. Scheduler 对每个新请求返回「除最后一个 token 外全部算外部命中」，且同步（`is_async=False`）。
2. `update_state_after_alloc` 在 `num_external_tokens > 0` 时把该请求的 block id 放进 `_pending`。
3. `build_connector_meta` 把 `_pending` 拷进 metadata 后清空。
4. Worker 的 `start_load_kv` 按 block id 把对应 KV 行填成常数。`save_kv_layer` 仍为空。

这已经覆盖接口里最关键的四步：**匹配 → 分配后记账 → 打包 meta → Worker 写 paged KV**。输出文本没有语义，只适合确认 Decode 路径和 TTFT 量级。

外壳按 role 拆两个内部类，避免在 Scheduler 进程里碰到 GPU tensor：

```python
class DummyFillConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._sched = DummyFillScheduler(vllm_config) if role == KVConnectorRole.SCHEDULER else None
        self._worker = DummyFillWorker() if role == KVConnectorRole.WORKER else None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        n = max(0, request.num_tokens - num_computed_tokens - 1)
        return n, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens == 0:
            return
        # 学习用：只取第 0 组 full-attention 的 block id
        block_ids = [b.block_id for b in blocks.blocks[0] if not b.is_null]
        self._sched.pending[request.request_id] = block_ids

    def build_connector_meta(self, scheduler_output):
        meta = DummyFillMetadata(reqs=dict(self._sched.pending))
        self._sched.pending.clear()
        return meta

    def register_kv_caches(self, kv_caches):
        self._worker.kv_caches = kv_caches

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        for block_ids in meta.reqs.values():
            self._worker.fill(block_ids)

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        return
```

Worker.fill 里对每个 layer 的 KV tensor 做 `kv[block_ids] = 常数`。真实代码还要处理多 KV group、MLA 的 shape、fp8 的 view，那些是 `DecodeBenchConnector` 的篇幅，最小实验可以先固定单组 full-attention。

配置：

```text
--kv-transfer-config '{
  "kv_connector": "DummyFillConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "my_connectors.dummy_fill"
}'
--disable-hybrid-kv-cache-manager
```

自测：发一个很长的 prompt，prefill 计算量应接近「只算最后一个 token」，GPU 时间会掉一截，生成内容是垃圾。这就是成功。

---

## 级别 2：共享目录 save / load

对应官方 `ExampleConnector`。单机、`kv_both`、两次请求同一 prompt 前缀时，第二次应跳过大部分 prefill。

Scheduler 逻辑：

1. `get_num_new_matched_tokens`：用 `prompt_token_ids` 对齐到 `block_size` 后查目录在不在。命中则返回 `aligned_len - num_computed_tokens`，`is_async=False`。
2. `update_state_after_alloc`：`num_external_tokens > 0` 的请求记入 `_requests_need_load`。
3. `build_connector_meta`：
   - 在 load 集合里的新请求 → metadata 带 `is_store=False` 和 `block_ids` / `slot_mapping`；
   - 否则若磁盘还没有这份前缀 → `is_store=True`（第一次写入）；
   - 然后清空 `_requests_need_load`。

Worker 逻辑：

1. `start_load_kv`：对 `is_store=False` 的请求，按层读 safetensors，用 `slot_mapping` 写进 paged KV。
2. `save_kv_layer`：对 `is_store=True` 的请求，用 `slot_mapping` 从 `kv_layer` 抽出 token，写文件。
3. `wait_for_layer_load` / `wait_for_save` 仍可空（同步 IO）。

slot 计算公式（和 paged KV 一致）：

```text
slot = block_id * block_size + offset_in_block
```

`ExampleConnector` 里对普通 attention 是 `dst[block_idxs, :, offsets] = src`；MLA 要先把 `[B, 1, N, C]` reshape 成 `[B*N, C]` 再按 slot 索引。抄官方时盯这两处 shape，不要盯文件 IO。

目录路径必须 Scheduler 和 Worker 都能读。默认 `/tmp/<hash>/layer_name.safetensors`，多机就换成 NFS。`kv_connector_extra_config.shared_storage_path` 是官方用的键名。

配置：

```text
--kv-transfer-config '{
  "kv_connector": "ExampleConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {"shared_storage_path": "/tmp/kv-diy"}
}'
```

内置就能跑，不必自己再写一份。自己写的意义是改 hash 规则、改只存 prefix 的哪些 token。官方 debug 实现会覆盖 GPU 上已有的 prefix cache，生产不能直接用。

---

## 从「最小」走到「能传 PD」还缺什么

级别 2 仍是**单实例、同步、整层文件**。PD 直传要在 Worker 里补这些，接口不用换：

| 缺口 | 典型补法 |
|------|----------|
| 两个进程要交换「对端 KV 基址 + block 列表」 | ZMQ / HTTP side channel；或引擎的 handshake metadata |
| GPU 内存要给网卡 DMA | `register_kv_caches` → 传输引擎 `registerLocalMemory` |
| Prefill 算完 Decode 才有地方写 | Decode 先 `get_num_new_matched_tokens` 占住 block，把地址发给 Prefill；Prefill `request_finished` 里 `delay_free_blocks=True` |
| 数据面 | Prefill `WRITE` 到 Decode（Mooncake）或 Decode `READ` Prefill（部分 NIXL pull） |
| 异步完成 | Worker `get_finished`；Scheduler `update_connector_output` 后再释放 block |

这些逐步加在级别 0 的骨架上即可。Mooncake 的对照见 [plan/03-vllm-kv-connector.md](../mooncake-learning-plan/plan/03-vllm-kv-connector.md)：它的 7 个 abstract 方法一个都没少，只是 Worker 里换成 TE。

---

## 写的时候容易踩的坑

1. **在 Scheduler 里创建 TransferEngine / 打开 GPU。** role 分流，TE 只活在 Worker。
2. **`get_num_new_matched_tokens` 里改内部状态。** 它可能被多次调用；记账放 `update_state_after_alloc`。
3. **返回的命中长度没对齐 `block_size`。** V1 调度按 block 对齐 `num_computed_tokens`，你返回的也要对齐。
4. **`num_external_tokens == 0` 仍去 load。** MultiConnector 会把真实 block 传给每个子 connector。
5. **`build_connector_meta` 不清空 pending。** 下一 step 会把旧请求再下发一次。
6. **异步 save 却 `request_finished` 返回 `False`。** 引擎立刻回收 block，传输读到被覆盖的页。
7. **忘了 HMA 开关。** 新版本默认 hybrid manager，最小类会启动失败。
8. **模块级 dict 当跨实例缓存。** 只对「同进程、同 Worker」成立；PD 和多 replica 都假。

---

## 自测题

1. 为什么 NoOp 必须返回 `(0, False)` 而不能返回 `(0, True)`？
2. Dummy Fill 的 `n = num_tokens - num_computed_tokens - 1`，这个 `-1` 是为了留下什么？
3. 把 Dummy Fill 的 `self.cache = {}` 写在外壳类上，Scheduler 往里面塞 KV tensor，Worker 能读到吗？
4. `ExampleConnector` 为什么用磁盘而不是内存 dict？这和「两个进程」有什么关系？
5. 若你要做 PD，Decode 的 `get_num_new_matched_tokens` 应该看 `request` 上的哪份字段才知道去哪拉？这份字段是谁在 Prefill 的哪个钩子里填的？
6. 列出从「请求进 waiting」到「Worker 第一次 `start_load_kv`」引擎会调你哪些方法，顺序不能错。
