# 持久批与输入准备

V1 把持久状态张量直接当作模型和采样器的输入。输入要求「前 N 行紧凑、顺序符合注意力后端」，所以请求结束必须把后面的行挪下来，改动必须整块拷到 GPU，输入也只能在 CPU 上拼好再传。V2 把持久状态和本步输入拆开：行号在请求活着期间不变，本步要什么顺序就从持久状态里 gather。后面的写入方式和输入准备都是这一刀的结果。

## 行不再搬

| | V1 `gpu_input_batch.py` | V2 `gpu/states.py` |
|--|--------------------------|---------------------|
| 请求状态 | `CachedRequestState`，每请求一份 Python 备份 | 没有。行不会被别的请求覆盖，备份没有存在理由 |
| 加入 / 结束 | `add_request` / `remove_request`，结束留下空洞 | `free_indices` 弹一个槽、还回去。`req_id → 行号` 在活跃期内不变 |
| 腾挪 | `condense()` 把非空行滑到低位；`swap_states()` 换行。block table 跟着 `move_row` / `swap_row` | 没有这些方法 |
| 本步顺序 | 持久表的行序必须就是模型要的顺序（`_may_reorder_batch`） | `sort_batch_req_ids()` 只决定本步 gather 的顺序，不动持久表 |
| 抢占后恢复 | 单独处理 | 当结束。恢复时重新 `add_request`，行号可以换 |

`condense()` 的注释写的就是 "Slide non-empty requests down into lower, empty indices"。挪的不只是一张表：block table、采样参数、token ids 都要跟着走。V2 把这笔成本换成每步一次 GPU gather（`gather_batch_req_state`、`BlockTables.gather_block_tables`）。gather 只碰本步的行，而且在 GPU 上并行。

V2 因此有三层，不要混：`RequestState` 是跨步的持久行；`InputBuffers` 是按最大 batch 预分配、反复填的 GPU 缓冲；`InputBatch` 是每步新建的 dataclass，字段多半是缓冲的切片。

## 改动怎么送到 GPU

V1 只有 `CpuGpuBuffer`（`vllm/v1/utils.py`）：pinned 的 `cpu` 加一份同形状的 `gpu`，`copy_to_gpu(n)` 把前 n 行整块拷过去。一步里只有 3 个请求各追加 1 个 block，拷的仍是 `num_reqs × max_num_blocks`。CPU 真值本身 pinned 且被下一步复用，异步拷贝还没完 CPU 就开始改——这是后面异步屏障要挡的那个竞态。

V2 在 `gpu/buffer_utils.py` 按张量的大小和访问方式分成三种：

| 机制 | 用在哪 | 做法 |
|------|--------|------|
| `StagedWriteTensor` | block table、`total_len`、`num_computed_tokens` 这类大且增量的列 | 基张量常驻 GPU。`stage_write` 把本步的 ragged 写入暂存在 CPU，`apply_write` 打成一段连续缓冲，一次 H2D，`_apply_write_kernel` 散射回去。多次写入收成一次 kernel |
| `UvaBackedTensor` + `UvaBufferPool` | `prompt_len`、`prefill_len` 这类小列 | 真值 `cpu` 显式 `pin_memory=False`。每步拷进轮转池里的下一个 pinned 槽，GPU 只读那个槽 |
| UVA | `all_token_ids`，形状 `(max_num_reqs, max_model_len)`，默认 `uva_instead_of_gpu=True` | `get_accelerator_view_from_cpu_tensor` 让 kernel 直接读 CPU 内存，显存里不放第二份。ngram 要反复扫活跃行，才用 `use_dense_all_token_ids` 改放显存 |

轮转池默认深度 2（`_DEFAULT_MAX_CONCURRENCY`），`set_default_max_concurrency` 按 engine 的在途 step 数加。深度小于在途步数时，会写回 GPU 还在读的槽。

`prompt_len` 是用户 prompt 长度，`prefill_len` 是喂进 runner 的长度，抢占恢复时已生成的 token 会重算，所以 `prefill_len ≥ prompt_len`。prompt logprobs 和 frequency penalty 必须把两者分开。

block table 是上面两节的汇合（V1 `vllm/v1/worker/block_table.py`，V2 `gpu/block_table.py`）：

| | V1 | V2 |
|--|----|----|
| 生效 | `commit_block_table(num_reqs)`，前 n 行全量 H2D | `apply_staged_writes()`，只传本步追加的片段 |
| 交给注意力 | `get_device_tensor(num_reqs)`，持久表前 n 行原样给后端，所以行序不能乱 | `gather_block_tables()` 按本步顺序 gather 出一张新表 |
| 多 KV group | `MultiGroupBlockTable` 持有 N 个 `BlockTable`，每个方法循环 N 次 | 指针张量交给同一个 kernel |

## 输入在 GPU 上填

V1 的 `_prepare_inputs()` 在 numpy 里拼 `input_ids` / `positions` / `query_start_loc` 再 H2D。下一步的 `input_ids` 开头是上一步采出的 token，异步调度下这个值还在 GPU 上，所以要有 `set_async_sampled_token_ids` / `update_async_output_token_ids` 把结果回填给 CPU。

V2 的 `prepare_inputs()` 把填写交给 `gpu/input_batch.py` 里的 kernel：

| kernel | 填什么 |
|--------|--------|
| `prepare_prefill_inputs` | prefill 段的 `input_ids`，直接读 UVA 上的 `all_token_ids` |
| `prepare_pos_seq_lens` | `positions`、`seq_lens` |
| `combine_sampled_and_draft_tokens` | 上一步采样结果和本步草稿都在 GPU 上，拼进 `input_ids`。CPU 不需要知道采出了什么 |
| `expand_idx_mapping` | 每条 logits 对应哪个请求行（投机解码） |
| `post_update` | 步末在 GPU 上改持久状态 |

CPU 仍留着一份镜像做决策：选哪张 CUDA graph、要不要 padding，这些必须在 kernel 之前知道。`num_computed_tokens` 的 numpy 那份是乐观上界，GPU 那份才是真值。自适应验证里 `num_scheduled_tokens` 同样只是均匀预算的上界，`compact_batch` 在 CPU 侧收紧，`reallocate_drafts` 在 GPU 上按接受率重分。
