# 05 · 持久化状态与无同步写入

目标：读懂 MRV2 的地基三件套——`RequestState`（每请求一行的持久状态）、`BlockTables`（大张量的增量写入）、`buffer_utils`（UVA + StagedWriteTensor + 缓冲池）。第 01 章讲的四条设计主张（persistent batch、消除竞态、StagedWriteTensor、UVA）全部在这里落地。

---

## 要看的内容

1. `vllm/v1/worker/gpu/buffer_utils.py`——**先读这个**，它是另外两个文件的词汇表
   - `UvaBuffer` / `NonUvaBuffer` / `UvaBufferPool` / `UvaBackedTensor` / `StagedWriteTensor` / `FusedStagedWriter`
   - Triton kernel `_apply_write_kernel`
2. `vllm/v1/worker/gpu/states.py`（约 130 行，很短，建议逐行读）
3. `vllm/v1/worker/gpu/block_table.py` · `BlockTables` 及其两个 kernel（`_gather_block_tables_kernel` / `_compute_slot_mappings_kernel`）
4. 对照组：`vllm/v1/worker/gpu_input_batch.py`（V1 的 `InputBatch` + `CachedRequestState`）

---

## 要掌握的知识

### `RequestState`：槽位分配器 + 一堆并行维护的状态列

核心是三个字段构成的槽位分配器：

```text
req_id_to_index: dict[str, int]
index_to_req_id: dict[int, str]
free_indices:    list[int]        # 初始为 range(max_num_reqs)
```

`add_request()` 从 `free_indices` 弹一个槽，`remove_request()` 还回去。**没有任何张量重排**——这就是第 01 章「每请求固定一行」的全部实现。

状态列按存储策略分三类，看清楚每一列为什么选这个策略：

| 列 | 类型 | 为什么 |
|----|------|--------|
| `all_token_ids` | `StagedWriteTensor`，默认 `uva_instead_of_gpu=True` | 形状是 `(max_num_reqs, max_model_len)`，可能几个 GB，默认放 UVA 省显存；但 `ngram_gpu` 需要反复扫活跃行，所以 `use_dense_all_token_ids` 时改放显存 |
| `prompt_len` / `prefill_len` | `UvaBackedTensor` | 小、CPU 写、GPU 读 |
| `total_len` / `num_computed_tokens` | `StagedWriteTensor` | GPU 常驻 + 增量写；`num_computed_tokens` 是 CPU/GPU 混合写的典型 |
| `num_computed_prefill_tokens` / `num_computed_tokens_np` / `max_seq_len` | 裸 numpy | 纯 CPU 决策用 |
| `last_sampled_tokens` / `draft_tokens` / `next_prefill_tokens` | 裸 GPU 张量 | 纯 GPU 产出、GPU 消费 |

`num_computed_tokens` 同时有 GPU 版和 `_np` 版，注释写明 numpy 那份是**乐观镜像（上界）**。理解这一点很重要：CPU 不能等 GPU 告诉它真实值（那就同步了），所以它维护一个保守的上界用于调度决策，GPU 上的那份才是真值。

### `prompt_len` vs `prefill_len`：最容易搞混的一对

代码注释专门强调：

- `prompt_len`：用户给的 prompt 长度。
- `prefill_len`：喂进 model runner 的 token 数，**可能包含已产出的部分 output token**，所以 `prefill_len >= prompt_len`。
- 典型差异来源：抢占后恢复——恢复时已经生成的 output 会被当作 prefill 重算。
- 为什么必须区分：prompt logprobs、frequency penalty 这类特性必须把 prompt 和 output 分开对待。

再加上 `total_len = prompt_len + output_len`，随请求推进增长。三个长度别混。

### `StagedWriteTensor`：把「写」和「生效」分开

用法是两段式：

```python
state.stage_write(row=2, start=3, value=[3, 1, 2])   # 只暂存到 CPU
state.stage_write(row=0, start=1, value=[-1, -2, -5])
state.apply_write()                                   # 打包 → H2D → 一个 kernel 落盘
```

关键在 `apply_write()`：把这一轮所有暂存的 ragged 写入**打包成连续缓冲**，一次 H2D，再用一个 Triton kernel（`_apply_write_kernel`）散射到基张量。好处是 ragged 写入不需要 CPU-GPU 同步，kernel launch 次数也被压到 1。`FusedStagedWriter` 再进一步，把多个 `StagedWriteTensor` 的写入合并。

`RequestState.apply_staged_writes()` 就是把这一轮所有列的 staged 写入一次性兑现——在第 04 章的 step 流程里，它和 `block_tables.apply_staged_writes()` 是成对出现的。

### `UvaBufferPool`：第 01 章「临时 pinned 副本」的工程化版本

设计文档里的示意写法是每步 `self.states.pin_memory()` 现分配一份。实际实现换成了**轮转缓冲池**：

- `UvaBackedTensor.cpu` 是真值，**不 pin**（`pin_memory=False`）。
- `UvaBufferPool` 预分配 `max_concurrency` 个 pinned `UvaBuffer`，`copy_to_uva()` 每次轮转到下一个。
- `_DEFAULT_MAX_CONCURRENCY = 2`，注释写明它必须 ≥ engine 的 `batch_queue_size`（并发在途 step 数）。

这解决的是同一个竞态问题，但避免了每步分配 pinned 内存的开销。**轮转深度必须覆盖在途步数**——这是理解它正确性的唯一要点：如果池子比在途步数浅，就会回绕到一个 GPU 还在读的 buffer。

还要注意平台兜底：没有 pinned memory 的平台走 `NonUvaBuffer`，它退化成「CPU 张量 + 显存副本 + 显式 copy」。

`UvaBuffer` 的核心是 `get_accelerator_view_from_cpu_tensor()`——把 pinned CPU 张量包成 GPU kernel 能直接寻址的 view，kernel 就能直接读 CPU 内存，不需要显存副本。

### `BlockTables`：大张量 + 两个 kernel

- `append_block_ids()` 走 staged write，`apply_staged_writes()` 兑现。
- `gather_block_tables()` 按本 step 的请求顺序从持久表里 gather（第 01 章「按顺序 gather」的落地）。
- `compute_slot_mappings()` 算出每个 token 写进 KV cache 的哪个槽。
- `_make_ptr_tensor()` / `init_block_table_layout_tensors()`：多 KV cache group（不同层可能有不同的 block 配置）时，用指针张量把多张表交给同一个 kernel。
- `get_dummy_block_tables()` / `get_dummy_slot_mappings()`：给 CUDA graph 捕获和 profiling 用。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲透 MRV2 的持久状态层和无同步写入机制。

第一部分 · buffer_utils.py（先讲这个）
1. 逐个讲 UvaBuffer / NonUvaBuffer / UvaBufferPool / UvaBackedTensor /
   StagedWriteTensor / FusedStagedWriter，每个给：职责、内存住在哪、谁写谁读。
2. 重点讲 UvaBufferPool 的轮转机制：
   - _DEFAULT_MAX_CONCURRENCY 为什么默认是 2？它和 engine 的 batch_queue_size 是什么关系？
   - set_default_max_concurrency() 被谁在什么时候调用？
   - 如果轮转深度小于在途 step 数，会发生什么具体的错误？
   - copy_to_uva() 里的动态扩容逻辑（1 << (n-1).bit_length()）为什么安全？注释里的
     "callers must retire a slot's GPU readers before reuse" 这条约束由谁保证？
3. 把 UvaBackedTensor 的写法和 docs/design/model_runner_v2.md 第 3 节的
   "tmp_states = self.states.pin_memory()" 示例对应起来，说明实现为什么用池子而不是现场 pin。
4. 讲 StagedWriteTensor 的完整生命周期：stage_write / stage_write_elem / apply_write /
   clear_staged_writes，以及 _apply_write_kernel 这个 Triton kernel 具体干了什么
   （输入的打包布局是什么样的、怎么散射回基张量）。
5. 讲 get_accelerator_view_from_cpu_tensor() 的作用，说明 UVA 在这里省掉了什么。
6. NonUvaBuffer 这条兜底路径在什么平台上会被选中？它丢掉了 UVA 的什么好处？

第二部分 · states.py
7. 讲 RequestState 的槽位分配器（req_id_to_index / index_to_req_id / free_indices），
   说明 add_request / remove_request 为什么完全不需要张量重排。
   对比 vllm/v1/worker/gpu_input_batch.py 里 V1 的做法，指出 V1 在哪一行做了重排。
8. 列一张表：RequestState 的每个字段 → 存储类型（StagedWriteTensor / UvaBackedTensor /
   裸 numpy / 裸 GPU 张量）→ 为什么选这个 → 谁写谁读。
9. 专门讲 prompt_len / prefill_len / total_len 三者的区别，照着代码注释讲，
   并举出 prefill_len > prompt_len 的具体场景。再说明哪些特性必须依赖这个区分。
10. 讲 all_token_ids 的 uva_instead_of_gpu 开关：默认走 UVA 的理由（算一下
    max_num_reqs × max_model_len × 4 字节的量级），以及 use_dense_all_token_ids
    在什么情况下被打开、为什么 ngram 需要它。
11. 解释 num_computed_tokens（GPU）和 num_computed_tokens_np（CPU）为什么要并存，
    以及注释说 CPU 那份是"乐观上界"是什么意思，谁在用这个上界。
12. apply_staged_writes() 把哪几列一次性兑现？它在 step 流程里的调用时机是什么
    （回到 vllm/v1/worker/gpu/model_runner.py 找调用点）。

第三部分 · block_table.py
13. 讲 BlockTables 的每个方法：append_block_ids / apply_staged_writes /
    gather_block_tables / compute_slot_mappings / get_dummy_block_tables /
    get_dummy_slot_mappings。
14. 讲 _make_ptr_tensor() 和 init_block_table_layout_tensors()：
    多个 KV cache group 是怎么被塞进同一个 kernel 的？
15. 讲清楚两个 Triton kernel（_gather_block_tables_kernel、_compute_slot_mappings_kernel）
    的输入输出形状和并行维度划分。
16. 说明为什么 block table 是 StagedWriteTensor 最典型的使用场景
    （和 docs/design/model_runner_v2.md 第 4 节对应）。

第四部分 · 对照 V1
17. 用一张表对比 V1 和 V2 在"请求加入 / 请求结束 / 请求被抢占"三个事件上的状态操作，
    指出 V2 省掉了哪些工作。
18. 在 V1 里找到 CachedRequestState，说明 V2 用什么消掉了它。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 讲 Triton kernel 时要说清楚 program_id 的维度含义和每个 block 处理的数据范围。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- 区分 CPU numpy 视图（.np）、CPU 张量（.cpu）、UVA view（.uva()）、纯 GPU 张量，
  每次提到都要写清楚是哪一个。
```

---

## 自测题

1. `RequestState` 里一个请求结束时发生了什么？有没有张量被重排？
2. `prompt_len`、`prefill_len`、`total_len` 分别是什么？举一个 `prefill_len > prompt_len` 的场景。哪些特性会因为混淆这两者而出错？
3. `all_token_ids` 默认放在哪里？为什么？什么时候会改放显存？
4. `num_computed_tokens` 为什么要 GPU 和 numpy 各存一份？numpy 那份为什么是「上界」而不是精确值？
5. `StagedWriteTensor` 的 `apply_write()` 做了哪几步？为什么能避免 CPU-GPU 同步？
6. `UvaBufferPool` 的轮转深度默认是多少？它必须 ≥ 什么？小了会怎样？
7. `UvaBackedTensor.cpu` 为什么**不**设 `pin_memory=True`？如果设了会发生什么？
8. `get_accelerator_view_from_cpu_tensor()` 省掉了什么开销？代价是什么？
9. `BlockTables.gather_block_tables()` 为什么是「gather」而不是「直接用」？这对应设计文档的哪条主张？
10. 一个请求被抢占又恢复，它在 `RequestState` 里还是原来那一行吗？`prefill_len` 会变吗？
