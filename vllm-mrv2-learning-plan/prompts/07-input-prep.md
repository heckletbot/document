# 07 · 每步的输入张量怎么造出来

> 轴一 · 持久化批处理与输入准备

**对照的那件事**：`input_ids`、`positions`、`query_start_loc`、`seq_lens`、`logits_indices` 这些每步都要重建的张量，在哪里算、用什么算。

对应设计文档第 5 条。这一章也是「GPU 能知道 CPU 不知道的事」这个论点的落点。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 主函数 | `gpu_model_runner.py` · `_prepare_inputs()`（约 330 行） | `gpu/model_runner.py` · `prepare_inputs()`（约 200 行） |
| 辅助 | `_prepare_input_ids()`、`_get_cumsum_and_arange()`、`_compute_prev_positions()`、`_calc_spec_decode_metadata()`、`_calc_mrope_positions()` | `gpu/input_batch.py` 里一组 Triton kernel |
| 计算位置 | 主要是 CPU：numpy + Python 循环，再 H2D | 主要是 GPU：Triton kernel 直接在显存/UVA 上写 |
| 上一步的采样结果 | `set_async_sampled_token_ids()` / `update_async_output_token_ids()`，异步调度下要回填 | `combine_sampled_and_draft_tokens` kernel，**CPU 全程不看** |
| 输出载体 | 写进 `InputBatch` 的持久张量，再切片 | 新建的 `InputBatch` dataclass（视图 + numpy 并行维护） |
| 投机元数据 | `_calc_spec_decode_metadata()` 在 CPU 上算 | `expand_idx_mapping` / `get_num_sampled_and_rejected` 等 kernel 在 GPU 上算 |

V2 的 kernel 清单（都在 `gpu/input_batch.py`）：`_prepare_prefill_inputs_kernel`、`_prepare_pos_seq_lens_kernel`、`_combine_sampled_and_draft_tokens_kernel`、`_get_num_sampled_and_rejected_kernel`、`_post_update_kernel`、`_post_update_num_computed_tokens_kernel`、`_expand_idx_mapping_kernel`。

---

## 差异的因果

### 为什么 V1 必须在 CPU 上算

因为 V1 的输入张量**就是**持久张量的切片（第 04 章），而持久张量的 CPU 侧是真值。要填 `input_ids`，就得先在 numpy 上拼好再拷过去。

这带来两个问题：

1. **Python 开销**。批大、请求多时，numpy 索引和 Python 循环成为 CPU 瓶颈。
2. **CPU 必须知道上一步的采样结果**。下一步的 `input_ids` 开头就是上一步采样出的 token。异步调度下这个结果还在 GPU 上，于是 V1 需要 `set_async_sampled_token_ids()` / `update_async_output_token_ids()` 这类回填机制，并在某些路径上被迫同步。

### V2 的做法

把这些张量的填写全部变成 Triton kernel。`combine_sampled_and_draft_tokens` 是最能说明问题的一个：上一步的采样结果是一个 GPU 张量，草稿 token 也在 GPU 上，kernel 直接把它们拼成本步的 `input_ids`——**CPU 从头到尾不需要知道采样出了什么**。

投机解码下这一点更关键：接受了几个草稿、实际推进了多少 token，这些值要等验证完才知道，CPU 拿不到。V2 让 GPU 自己算 `positions` 和 `seq_lens`（`prepare_pos_seq_lens`），自己统计接受/拒绝（`get_num_sampled_and_rejected`），步末自己更新持久状态（`post_update`）。

### 代价与保留的 CPU 路径

V2 并没有把所有东西搬上 GPU。`prepare_inputs()` 里仍有 numpy 计算（`query_start_loc_np` 的 cumsum、`cu_num_logits_np`、`is_prefilling_np` 等），因为 CPU 需要这些值来做**调度形态决策**（选哪张 CUDA graph、要不要 padding）。

所以 V2 实际是「CPU 维护一份保守的镜像用于决策，GPU 维护真值用于计算」。`RequestState.num_computed_tokens` 同时有 GPU 版和 `_np` 版、注释写明 numpy 那份是**乐观上界**，就是这个双轨制的典型。自适应验证下 `num_scheduled_tokens` 只是上界、真实分配由 GPU 的 `reallocate_drafts()` 决定，也是同一个模式。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 构造每步输入张量的方式。
V1 = vllm/v1/worker/gpu_model_runner.py 的 _prepare_inputs 及其辅助方法
V2 = vllm/v1/worker/gpu/model_runner.py 的 prepare_inputs + gpu/input_batch.py 的 kernel

=== 第一部分：逐个张量对照 ===
1. 做一张主表，每行一个输入张量，列：
   张量名 | V1 在哪算（文件:行号 + CPU/GPU）| V2 在哪算（文件:行号 + CPU/GPU）| 差异一句话。
   至少覆盖：input_ids、positions、query_start_loc、seq_lens、slot_mapping、
   logits_indices / cu_num_logits、spec decode 元数据、mrope positions。
2. 对 input_ids 单独深挖：
   - V1 的 _prepare_input_ids() 怎么拼？prompt 部分和上一步采样出的 token 分别从哪来？
   - V2 的 prepare_prefill_inputs + combine_sampled_and_draft_tokens 怎么拼？
   - 画两张数据流图（CPU 框 / GPU 框），标出每次 H2D、D2H 和同步点。
3. 对 positions / seq_lens 单独深挖：
   V1 的 _compute_prev_positions() 和 _get_cumsum_and_arange() ↔
   V2 的 _prepare_pos_seq_lens_kernel。说明 V2 在 GPU 上算这两个值的前提是什么。

=== 第二部分：异步调度下的关键差异 ===
4. 讲 V1 的 set_async_sampled_token_ids() / update_async_output_token_ids() /
   update_async_spec_token_ids()（gpu_input_batch.py）：
   - 它们解决什么问题？
   - 异步调度下，CPU 要拿到上一步的采样结果需要付出什么代价？
   - 在 V1 里找出因此产生的同步点或回填延迟（给行号）。
5. 讲 V2 的 combine_sampled_and_draft_tokens：
   - kernel 的输入是哪几个 GPU 张量？
   - 明确回答：CPU 在这条路径上需要知道上一步采样出了什么吗？
   - 如果把这个 kernel 改回 CPU 实现，会在哪一行引入同步点？
6. 对比投机解码下的元数据计算：
   V1 的 _calc_spec_decode_metadata() ↔ V2 的 expand_idx_mapping /
   get_num_sampled_and_rejected / post_update。
   回答设计文档第 5 节那句"GPU can derive values that CPU may not know yet"
   在代码里具体指什么值，并举出至少两个这样的值。

=== 第三部分：V2 保留的 CPU 路径 ===
7. 在 V2 的 prepare_inputs() 里找出所有仍然用 numpy 计算的部分，列出来，
   并对每一个回答：为什么这部分不能搬到 GPU？（提示：CPU 需要它做什么决策）
8. 讲 V2 的"双轨制"：
   - RequestState.num_computed_tokens（GPU）和 num_computed_tokens_np（CPU 乐观上界）
     为什么要并存？谁用哪一份？
   - 自适应验证下 num_scheduled_tokens 为什么只是上界？
     compact_batch()（CPU）和 reallocate_drafts()（GPU）的分工是什么？
   - V1 有没有类似的双轨制？如果没有，V1 靠什么拿到精确值？
9. 讲 V2 的 InputBatch 里 _np 后缀字段和同名 GPU 张量的对应关系，
   列出哪些是成对的、哪些只有一边。

=== 第四部分：收敛 ===
10. 回答：把输入准备搬到 GPU，收益分几类？
    （至少讲清楚：CPU 开销、异步性、投机解码的可行性三类，各举代码为证。）
11. 回答：这个搬迁的前提条件是什么？
    （提示：顺着第 04、05 章追——如果持久状态还是直接当输入，kernel 能这么写吗？）
12. 回答：V2 这么做引入了什么新的复杂度或风险？
    （比如 kernel 数量、调试难度、CPU 侧镜像与 GPU 真值不一致的可能。）
    在代码里找出 V2 为了管理这个风险做的事（比如 assert、注释里的约束）。

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 讲 Triton kernel 时必须说清楚 program_id 的维度含义。
- 每次提到一个计算，明确它跑在 CPU 还是 GPU 上。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
