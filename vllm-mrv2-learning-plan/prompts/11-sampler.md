# 11 · 采样参数怎么组织、怎么采

> 轴三 · 采样器与模块化抽象

**对照的那件事**：每请求的采样参数怎么存、怎么对齐到 logits、随机数从哪来、logprobs 怎么算。

对应设计文档第 6 条。这是三条轴里唯一一条「两边代码几乎没有共同祖先」的——V2 的采样器是重写的，不是重构的。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 目录 | `vllm/v1/sample/`（`sampler.py`、`metadata.py`、`ops/`、`logits_processor/`、`rejection_sampler.py`） | `vllm/v1/worker/gpu/sample/` 整个子包 |
| 参数载体 | `SamplingMetadata`（dataclass，**每步从批状态重建**，见 `_make_sampling_metadata()`） | `SamplingStates` 等一组每请求一行的持久状态 |
| 对齐方式 | 张量按 batch 行对齐；投机解码下**展开**成 per-logit 形状 | `idx_mapping` / `expanded_idx_mapping` 在 kernel 内**间接寻址** |
| 快路径标志 | `all_greedy` / `all_random` / `no_top_p` / `no_top_k` / `no_penalties` 等布尔属性 | `any_greedy(idx_mapping_np)` / `needs_logits_processing` 等按本批计算 |
| 随机数 | `generators: dict[int, torch.Generator]`，**有状态**、每请求一个 | `murmur3_*` 系列，**无状态**，从 `(seed, pos, offset, domain)` 现算 |
| 随机采样 | `sample_with_exponential_noise()` / `random_sample()`（`ops/topk_topp_sampler.py`） | `gumbel_sample()` / `gumbel_noised_argmax()`（`sample/gumbel.py`） |
| logprobs | `compute_logprobs()` 全词表 log_softmax，再 `gather_logprobs()` | `compute_topk_scores()` 先选 top-k，再只算选中的 |
| 投机对接 | `_combine_outputs_with_spec_tokens()` | `expanded_idx_mapping` + `expanded_local_pos` |
| 实现语言 | PyTorch 算子为主（部分 Triton：`ops/topk_topp_triton.py`） | Triton 为主 |

---

## 差异的因果

### `SamplingMetadata` 的问题

它是每步（或每次批变动后）重建的 dataclass，字段里混着三类东西：GPU 张量（`temperature`、`top_p`）、Python 结构（`output_token_ids: list[list[int]]`、`bad_words_token_ids: dict`）、以及全局布尔快路径标志（`all_greedy`、`no_penalties`）。

三个连带后果：

1. **重建成本**。`_make_sampling_metadata()` 在 `refresh_metadata()` 里被调用，批一变就要重来。
2. **形状绑死**。张量按 batch 行排列，所以投机解码时每请求有多条 logits，就必须把采样状态**展开**到 logits 的形状。新增一个采样参数就要考虑一遍展开逻辑。
3. **快路径是全局的**。`all_greedy` 是「整批都 greedy」才为真，粒度很粗。

### V2 的两点替换

**每请求一行 + 间接寻址**。采样参数和第 04 章的 `RequestState` 同构：`add_request(req_idx, params)` 写一行，`apply_staged_writes()` 兑现。采样时 kernel 收到 `expanded_idx_mapping`，每条 logits 自己查表找到所属请求的参数行。投机解码的展开问题就此消失——这正是设计文档第 6 节最后一段讲的。

**无状态 RNG**。V1 每请求持有一个 `torch.Generator`（有状态，要维护、要在请求移动时跟着走、对 CUDA graph 不友好）。V2 改成 in-kernel 的 murmur3 哈希，从 `(seed, pos, offset, domain)` 直接算，`spec_step_seed(seed, step)` 给投机的每一步派生 seed。无状态带来可复现、无需维护、图友好三个好处。

### logprobs 的顺序反转

V1：全词表 `log_softmax` → `gather`。峰值显存约 `O(B × V)`（V 是词表，可能十几万）。
V2：先从 logits 选 top-k → 只对选中的算 logprob。峰值降到 `O(B × k)`。

`prompt_logprob.py` 还支持在**单个 prompt 内部**分块，避免长 prompt 的尖峰。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 的采样器实现。
V1 = vllm/v1/sample/（sampler.py、metadata.py、ops/、logits_processor/）
     + vllm/v1/worker/gpu_input_batch.py 的 _make_sampling_metadata / refresh_metadata
V2 = vllm/v1/worker/gpu/sample/ 整个子包

=== 第一部分：参数怎么存 ===
1. 讲 V1 的 SamplingMetadata（vllm/v1/sample/metadata.py）：
   把所有字段按"GPU 张量 / Python 结构 / 全局布尔标志"三类分组列出。
   对每个 Python 结构字段（output_token_ids、bad_words_token_ids、generators、
   logprob_token_ids、logitsprocs 等）回答：它为什么没法做成张量？
2. 讲 _make_sampling_metadata() 和 refresh_metadata()：
   什么时候触发重建？一次重建要做多少工作？
   把它和第 04 章的 condense() 联系起来——两者是不是同一个根因的两个症状？
3. 讲 V2 的对应物：SamplingStates（gpu/sample/states.py）和
   Sampler.__init__ 里装配的那一堆子状态（PenaltiesState / LogitBiasState /
   BadWordsState / LogprobTokenIdsState / ThinkingBudgetState / TraceReplayState）。
   说明 add_request / apply_staged_writes 的两段式，以及它和 RequestState 的同构之处。
4. 一张表对照：每一个采样参数（temperature / top_p / top_k / min_p /
   各种 penalty / bad_words / allowed_token_ids / logprobs 相关 / thinking budget）
   在 V1 存在哪、在 V2 存在哪、存储形态有什么不同。
5. 对比快路径标志：V1 的 all_greedy / all_random / no_top_p / no_top_k /
   no_penalties（整批粒度）↔ V2 的 any_greedy(idx_mapping_np) /
   needs_logits_processing（按本批请求算）。
   说明粒度差异带来的实际影响。

=== 第二部分：投机解码下的对齐（本章重点）===
6. 讲 V1 在投机解码下怎么把采样状态对齐到 logits：
   找到展开（expand / repeat_interleave 之类）的具体代码，
   说明每新增一个采样参数，开发者要在哪里补展开逻辑。
   _combine_outputs_with_spec_tokens() 在这条链上的位置是什么？
7. 讲 V2 的 idx_mapping 间接寻址：
   - expanded_idx_mapping 和 expanded_local_pos 分别是什么（定义域、值域）？
   - 举一个具体例子：3 个请求、其中 2 个带 2 个草稿 token，
     把 idx_mapping / expanded_idx_mapping / expanded_local_pos / cu_num_logits
     的实际数值全写出来。
   - 在 V2 的采样 kernel 里找出至少三处真正用它做间接寻址的地方，
     说明 kernel 内部是怎么从一条 logits 查到参数行的。
8. 对照 docs/design/model_runner_v2.md 第 6 节 "Better Compatibility with
   Speculative Decoding"，回答：新增一个复杂采样参数时，
   V1 要改几处、V2 要改几处？各举一个真实的参数作为例子走一遍。

=== 第三部分：随机数与采样算法 ===
9. 对比随机数来源：
   - V1 的 generators: dict[int, torch.Generator]：谁创建、存在哪、
     请求在批里移动时它怎么办、CUDA graph 捕获时它有什么问题？
   - V2 的 murmur3_hash32 / murmur3_uniform32 / murmur3_uniform64：
     输入是哪几个量？为什么说它无状态？
   - 逐条验证无状态的三个好处（可复现 / 无需维护 / 图友好），每条给代码证据。
   - spec_step_seed(seed, step) 为什么必须存在？所有草稿步共用一个 seed 会怎样？
10. 对比采样算法：
    V1 的 random_sample / sample_with_exponential_noise / empty_exponential_noise_like
    ↔ V2 的 gumbel_sample / gumbel_noised_argmax / gumbel_block_argmax。
    - 写出两者的数学等价关系（指数噪声 vs Gumbel 噪声，各自和 softmax 采样的关系）。
    - 哪一边需要物化 softmax / probs？给代码行号。
    - V2 的 _log1p_neg_stable 在解决什么数值问题？V1 对应的地方怎么处理的？
11. 对比 top-k / top-p 的实现：V1 的 apply_top_k_top_p /
    apply_top_k_top_p_pytorch / apply_top_k_only / topk_topp_triton
    ↔ V2 的 SamplingStates.apply_top_k_top_p / get_top_k_top_p。
12. 对比后端选择：V1 的 flashinfer_sampler_supported / xpu_sampler_supported /
    _aiter_raw_fused_sample / mixed_sample_eligible
    ↔ V2 的 use_flashinfer / use_xpu_sampler / use_fp64_gumbel。
    两边的互斥条件各是什么？

=== 第四部分：logprobs ===
13. 对比 logprobs 路径：
    V1 的 compute_logprobs（全词表 log_softmax）+ gather_logprobs +
    gather_specific_token_logprobs
    ↔ V2 的 compute_topk_scores + _topk_log_softmax_kernel + _ranks_kernel +
      compute_token_logprobs。
    - 用词表大小 V、批大小 B、请求的 k 写出两边的峰值显存量级。
    - 说明 V2 "先选后算" 的正确性：从 logits 选 top-k 和从 logprobs 选 top-k
      为什么结果一致？
14. 对比 prompt logprobs：V1 的 _get_prompt_logprobs_dict ↔ V2 的
    sample/prompt_logprob.py。V2 的"单个 prompt 内部分块"解决什么问题？
15. 对比 logits processor 的接入：
    V1 的 vllm/v1/sample/logits_processor/ + apply_logits_processors
    ↔ V2 的 gpu/sample/logits_processor/ + Sampler.logits_processors 列表。
    V2 注释说"bias adds, penalties scale, so the two do not commute"，
    V1 有没有对应的顺序保证？在哪里？

=== 第五部分：收敛 ===
16. 一张表总结采样器这条线上 V1 → V2 的全部变化，每行写：
    变化点 | V1 做法 | V2 做法 | 收益 | 代价。
17. 回答：V2 的采样器是"重构"还是"重写"？给出判断依据
    （有多少代码/结构是可以一一对应的）。
18. 回答：V2 采样器的改动里，有哪些**不依赖**第 04 章的解耦、
    理论上可以单独移植回 V1？哪些不行？

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 讲 Triton kernel 时必须说清楚 program_id 的维度含义。
- 涉及数值精度的地方（fp32/fp64、log1p 稳定性、softmax 物化）要明确指出。
- V1 和 V2 都有名为 Sampler / RejectionSampler 的类，每次提到都要写清楚是哪个路径下的。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
