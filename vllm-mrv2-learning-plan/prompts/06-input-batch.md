# 06 · `InputBatch` 与 GPU 原生输入准备

目标：读懂 MRV2 每一步是怎么从持久状态「长出」这一步的模型输入的。本章对应设计文档第 5 条（GPU 原生输入准备）和第 7 条（把输入收进 `InputBatch`）。

---

## 要看的内容

1. `vllm/v1/worker/gpu/input_batch.py` 全文
   - `InputBuffers`（预分配的复用缓冲）
   - `InputBatch`（dataclass，字段注释写得很细，**逐字段读**）
   - Triton kernels：`_prepare_prefill_inputs_kernel`、`_prepare_pos_seq_lens_kernel`、`_combine_sampled_and_draft_tokens_kernel`、`_get_num_sampled_and_rejected_kernel`、`_post_update_kernel`、`_post_update_num_computed_tokens_kernel`、`_expand_idx_mapping_kernel`
2. `vllm/v1/worker/gpu/model_runner.py` · `prepare_inputs()`（约 200 行，本章主线）
3. 同文件 · `gather_batch_req_state()`、`BatchReqState`
4. 对照组：`vllm/v1/worker/gpu_input_batch.py`（V1）

---

## 要掌握的知识

### `InputBuffers` vs `InputBatch`

- `InputBuffers`：构造一次、长期复用的 GPU 缓冲（`input_ids`、`positions`、`is_padding`、`query_start_loc`、`seq_lens`、`dcp_local_seq_lens`）。按 `max_num_tokens` / `max_num_reqs` 预分配。
- `InputBatch`：每步新建的 **dataclass 视图**，字段大多是上面缓冲的切片，加上一堆 CPU 侧 numpy。

关键是理解这是**两层**：持久缓冲（不动）→ 每步切片+填充（`InputBatch`）。这和第 05 章的 `RequestState`（持久状态）又是一层，一共三层，别混。

### `InputBatch` 字段按用途分组

读的时候按这几组归类，比按顺序读高效得多：

| 组 | 字段 |
|----|------|
| 身份与映射 | `req_ids`、`num_reqs`、`num_reqs_after_padding`、`idx_mapping` / `idx_mapping_np`、`expanded_idx_mapping`、`expanded_local_pos` |
| token 计数 | `num_scheduled_tokens`、`num_tokens`、`num_tokens_after_padding`、`num_draft_tokens`、`num_draft_tokens_per_req` |
| 注意力形状 | `query_start_loc` / `_np`、`seq_lens`、`seq_lens_cpu_upper_bound`、`dcp_local_seq_lens`、`max_query_len`、`max_seq_len_np` |
| prefill 进度 | `num_computed_tokens_np`、`prefill_len_np`、`num_computed_prefill_tokens_np`、`is_prefilling_np`、`has_prefill`、`prefill_runs_as_decode_np`、`decode_graph_eligible` |
| 模型输入 | `input_ids`、`positions`、`is_padding` |
| 采样入口 | `logits_indices`、`cu_num_logits` / `_np` |
| 特性开关 | `has_structured_output_reqs`、`prompt_lens`（R-SWA）、`fast_prefill`（KV sharing） |

### `idx_mapping` 是全书最重要的一个概念

三层索引要分清：

- **batch_idx**：本 step 批内第几个请求（顺序由 `sort_batch_req_ids()` 决定）。
- **req_state_idx**：这个请求在 `RequestState` 里的常驻行号。
- `idx_mapping[batch_idx] = req_state_idx`。

`expanded_idx_mapping` 是 logits 粒度的版本：投机解码下一个请求对应多条 logits，这个数组把**每条 logits** 指回它的 `req_state_idx`，`expanded_local_pos` 则给出它在请求内的位置。设计文档第 6 节说的「用 indirection 代替把采样状态展开成 logits 形状」，就是这两个数组。

无投机的常见情况下有快路径：`expanded_idx_mapping` 直接等于 `idx_mapping`，`cu_num_logits` 就是 `arange(num_reqs+1)`。

### `prepare_inputs()` 的几个难点

读这个函数时按下面几个关键点拆：

1. **padding 的两套**：token 级（`num_tokens_after_padding`，由 CUDA graph 描述符决定）和 request 级（`num_reqs_after_padding`，PIECEWISE 图不需要）。`VLLM_MOE_SKIP_PADDING` 还会改 `is_padding` 的填法。
2. **logits 行数的计算**：无草稿时 `total_num_logits = num_reqs`；有草稿时 `num_reqs * num_bonus_tokens + total_num_draft_tokens`。代码里有一句 `assert (num_scheduled_tokens_np >= num_logits).all()`，注释解释得很清楚：logits 行取的是 `[query_end - num_logits, query_end)`，query 行数不够会**静默取到前一个请求的 hidden states**。这是个很好的「为什么要这个 assert」的例子。
3. **`query_start_loc` 的 padding**：尾部填 `num_tokens` 而不是 0，因为 FA3 之类的后端要求它非递减。
4. **自适应验证的两段式**：`compact_batch()` 先在 CPU 侧压缩批，`reallocate_drafts()` 再在 **GPU 侧**不均匀地重分配草稿预算。这是「CPU 不知道真实值，让 GPU 自己决定」的典型例子——`num_scheduled_tokens` 此时只是**上界**。
5. **prefill token 的来源**：从 `RequestState.all_token_ids`（可能在 UVA 上）取，kernel 直接读 CPU 内存。

### 那几个 Triton kernel 各管什么

| kernel / 函数 | 职责 |
|---------------|------|
| `prepare_prefill_inputs` | 填 prefill 部分的 `input_ids` |
| `prepare_pos_seq_lens` | 算 `positions` 和 `seq_lens` |
| `combine_sampled_and_draft_tokens` | 把上一步采样出的 token 和本步的草稿 token 拼成 `input_ids`——**这是 async-first 的关键**：上一步的采样结果还在 GPU 上，CPU 从没见过它 |
| `get_num_sampled_and_rejected` | 统计接受/拒绝数量 |
| `post_update` / `post_update_num_computed_tokens` | 步末在 GPU 上更新持久状态 |
| `expand_idx_mapping` | 生成 `expanded_idx_mapping` / `expanded_local_pos` |

`combine_sampled_and_draft_tokens` 值得单独停一下：V1 必须把上一步的 sampled token 拷回 CPU 才能拼下一步输入，这就是一个同步点；MRV2 让 GPU 自己拼，CPU 全程不需要知道采样结果。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲透 InputBatch 的结构和每步输入是怎么在 GPU 上被造出来的。

第一部分 · 数据结构
1. 对比 InputBuffers 和 InputBatch（vllm/v1/worker/gpu/input_batch.py）：
   谁是长期复用的、谁是每步新建的、后者的字段有多少是前者的切片。
2. 把 InputBatch 的所有字段按用途分组列表，每个字段给：形状、dtype、住在 CPU 还是 GPU、
   谁产出谁消费。字段的代码注释要原样转述，不要改写。
3. 专门讲三层索引：batch_idx / req_state_idx / logits 行号，以及
   idx_mapping、expanded_idx_mapping、expanded_local_pos 三者的关系。
   给一个具体例子：3 个请求、其中 2 个带 2 个草稿 token，把这三个数组的实际数值写出来。
4. 说明 InputBatch.make_dummy() 和 set_dummy_context() 的用途与调用方。

第二部分 · prepare_inputs 逐段精读
5. 逐段讲 vllm/v1/worker/gpu/model_runner.py 的 prepare_inputs()，每段给行号范围。
   必须覆盖：
   a. token 级 padding（num_tokens_after_padding）和 request 级 padding
      （num_reqs_after_padding）的来源，以及 PIECEWISE 图为什么不需要后者。
   b. envs.VLLM_MOE_SKIP_PADDING 改变了 is_padding 的什么填法，为什么 MoE 需要。
   c. 无草稿快路径 vs 一般路径的分叉条件，以及快路径省掉了什么。
   d. total_num_logits 的两种算法，num_bonus_tokens 是什么。
   e. 那句 assert (num_scheduled_tokens_np >= num_logits).all() 的注释在说什么风险，
      去掉它会出现什么 bug。
   f. query_start_loc 尾部为什么填 num_tokens 而不是 0（提示：FA3 的非递减要求）。
   g. 自适应验证的 compact_batch() / reallocate_drafts() 两段式：
      哪一段在 CPU、哪一段在 GPU，为什么 num_scheduled_tokens 在这种模式下只是上界。
   h. prefill token 从哪里取，怎么取（和第 05 章的 all_token_ids / UVA 对应）。

6. 讲 gather_batch_req_state() 和 BatchReqState：
   它在 prepare_inputs 之前先算出了什么，为什么要分成两步。
   顺带说明 uniform_tok_count 是怎么算的、给谁用。

第三部分 · Triton kernel
7. 逐个讲 input_batch.py 里的每个 Triton kernel：输入输出、grid 划分、
   program_id 的维度含义、每个 block 处理的数据范围。
8. 重点讲 combine_sampled_and_draft_tokens：
   - 它拼的是哪两部分数据？
   - 为什么这件事必须在 GPU 上做？如果搬回 CPU，会在哪里引入同步点？
   - 它和 V1 的对应逻辑（在 vllm/v1/worker/gpu_model_runner.py 里找）差在哪？
9. 讲 post_update / post_update_num_computed_tokens 在 step 末尾更新了哪些持久状态，
   以及它们和 RequestState 里 StagedWriteTensor 的关系。

第四部分 · 对照 V1
10. 用一张表对比 V1 和 V2 构造 input_ids / positions / query_start_loc 的方式：
    各自在 CPU 还是 GPU、各自需要几次 H2D、各自有没有同步点。
11. 结合 docs/design/model_runner_v2.md 第 5 节，说明"GPU 能推导出 CPU 不知道的值"
    这句话在投机解码场景下的具体含义，并在代码里指出一个实例。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 讲 Triton kernel 时必须说清楚 program_id 的维度含义。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- 区分 CPU numpy（_np 后缀）和同名 GPU 张量，每次都要写清楚是哪一个。
```

---

## 自测题

1. `InputBuffers` 和 `InputBatch` 的生命周期分别是什么？后者的字段有多少是前者的切片？
2. 写出 batch_idx、req_state_idx、logits 行号三者的关系，并说明 `idx_mapping` 和 `expanded_idx_mapping` 各自的定义域和值域。
3. 无草稿的常见情况下，`expanded_idx_mapping` 等于什么？`cu_num_logits` 等于什么？
4. `num_tokens_after_padding` 和 `num_reqs_after_padding` 分别由谁决定？什么情况下后者是 `None`？
5. `total_num_logits` 在有草稿时怎么算？`num_bonus_tokens` 是什么？
6. `assert (num_scheduled_tokens_np >= num_logits).all()` 防的是什么 bug？去掉它会出现什么现象？
7. `query_start_loc` 的尾部为什么填 `num_tokens`？
8. 自适应验证下，`num_scheduled_tokens` 为什么只是上界？真正的草稿预算在哪里决定？
9. `combine_sampled_and_draft_tokens` 为什么必须跑在 GPU 上？搬到 CPU 会在哪里卡住？
10. 一个 step 结束时，哪几个 kernel 负责把状态写回 `RequestState`？它们写的是哪几列？
