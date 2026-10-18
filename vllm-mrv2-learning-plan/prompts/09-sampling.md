# 09 · 采样栈：Triton sampler / logprobs / logits processors

目标：读懂 MRV2 重写的采样路径——为什么用 Gumbel-max、为什么 top-k logprobs 能省显存、`idx_mapping` 怎么让复杂采样参数在投机解码下不爆炸。对应设计文档第 6 条。

---

## 要看的内容

1. `vllm/v1/worker/gpu/sample/sampler.py` · `Sampler`（入口，先读 `__init__` 看它装配了什么）
2. `vllm/v1/worker/gpu/sample/states.py` · `SamplingStates`（每请求一行的采样参数）
3. `vllm/v1/worker/gpu/sample/gumbel.py`（**本章最有意思的文件**：murmur3 哈希 RNG + gumbel argmax）
4. `vllm/v1/worker/gpu/sample/logprob.py` · `compute_token_logprobs` / `compute_topk_scores` / `LogprobTokenIdsState`
5. `vllm/v1/worker/gpu/sample/output.py` · `SamplerOutput` / `SamplingMaskTensors`
6. logits processor 三件套：`logit_bias.py`、`penalties.py`、`bad_words.py` + `logits_processor/interface.py`、`loader.py`
7. 其余：`min_p.py`、`prompt_logprob.py`、`thinking_budget.py`、`watermark.py`、`batch_shard.py`、`trace_replay.py`

---

## 要掌握的知识

### `Sampler.__init__` 就是一张目录

它装配的东西直接对应各个特性文件：`SamplingStates`、`PenaltiesState`、`LogitBiasState`、`BadWordsState`、自定义 logits processors、`LogprobTokenIdsState`、`ThinkingBudgetState`、`TraceReplayState`。

有一条注释必须记住：**logits processor 的列表顺序就是流水线顺序，bias 做加法、penalties 做缩放，两者不可交换**。顺序是 `logit_bias → penalties → bad_words → 自定义`。

还有三个「后端选择」开关需要留意：`use_flashinfer`、`use_xpu_sampler`、`use_fp64_gumbel`。注意 `return_sampling_mask=True` 会同时关掉前两个（因为它们产出不了需要的 mask），而 XPU kernel 只能抽 fp32 噪声所以不兼容 fp64 gumbel。这类「开关互斥」是排查采样数值差异时的第一现场。

### 每请求一行 + `idx_mapping` 间接寻址

和第 05 章一个套路：`SamplingStates` 也是 `add_request(req_idx, sampling_params)` 写某一行、`apply_staged_writes()` 兑现。`Sampler.add_request()` 把这一行同时广播给所有子状态，并记录 `needs_logits_processing[req_idx]`——**没人需要 logits 处理时整条流水线可以跳过**。

采样时传入的是 `input_batch.expanded_idx_mapping`（logits 粒度）和 `idx_mapping`（请求粒度）。设计文档第 6 节说的「不把 per-request 采样状态展开成 per-logit 形状，而是在 kernel 里用 `idx_mapping` 做间接寻址」就是这两个参数的作用。理解它的收益：新增一个采样参数时，**不需要考虑投机解码的展开逻辑**，kernel 自己查表就行。

### Gumbel-max 采样

核心思想：从 softmax 分布采样 ≡ 给 logits 加 Gumbel 噪声后取 argmax。于是**不需要物化 softmax**。

`gumbel.py` 里值得细看的：
- `murmur3_hash32` / `murmur3_uniform32` / `murmur3_uniform64`：**无状态**的 in-kernel RNG，从 `(seed, pos, offset, domain)` 直接算随机数。无状态意味着不需要维护 RNG state、天然可复现、CUDA graph 友好。
- `spec_step_seed(seed, step)`：投机解码的每一步派生一个 seed。
- `_log1p_neg_stable`：数值稳定地算 `log(1-x)`。
- `gumbel_noised_argmax` / `gumbel_block_argmax` / `_gumbel_sample_kernel`。

### logprobs：先选后算

V1 是「物化全词表 logprobs → top-k」。MRV2 反过来：**先从 logits 里挑出 top-k token，再只对选中的 token 算 logprob**。省的是显存峰值（词表可能十几万）。

`compute_topk_scores` / `_topk_log_softmax_kernel` / `_ranks_kernel` 是这条路径；`LogprobTokenIdsState` 维护每请求额外要求 logprob 的 token id；`get_logprobs_dims()` 先算出这一批的 `(num_logprobs, max_per_req_token_ids)`，没人要就返回 `None` 整段跳过。

`prompt_logprob.py` 对应设计文档说的「更细粒度的分块，包括在**单个 prompt 内部**分块」，避免长 prompt 的显存尖峰。

### 几个 V2-only 的采样特性

- `watermark.py`（约 1000 行）：水印，是强制走 V2 的两个特性之一。
- `batch_shard.py`：batch-sharded sampling，把采样按 TP rank 分片（V1 不支持，约束见第 03 章）。
- `thinking_budget.py`：推理预算控制，和 `reasoning_config` 关联。
- `trace_replay.py`：采样轨迹重放，调试用。
- `SamplingMaskTensors`（`output.py`）：`return_sampling_mask` 开启时产出，注意它和 flashinfer / XPU sampler 互斥。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲透 MRV2 的采样栈。

第一部分 · 装配与状态
1. 逐行讲 vllm/v1/worker/gpu/sample/sampler.py 的 Sampler.__init__：
   它装配了哪些子状态和 logits processor，每个对应哪个文件。
2. 重点讲 self.logits_processors 的列表顺序：
   照注释解释"bias adds, penalties scale, so the two do not commute"，
   举一个顺序颠倒会产出不同结果的具体例子。
3. 讲三个后端选择开关：use_flashinfer / use_xpu_sampler / use_fp64_gumbel。
   说明 return_sampling_mask 为什么会同时否决前两个、
   XPU kernel 为什么不能配 fp64 gumbel（照注释讲）。
4. 讲 SamplingStates（sample/states.py）的每个字段：哪些采样参数被存成每请求一行，
   add_request() 返回的 needs_processing 代表什么，谁在用 needs_logits_processing 这个数组。
5. 讲 Sampler.add_request / apply_staged_writes 怎么把一次写入广播给所有子状态，
   以及它在 model_runner.py 的哪一步被调用。

第二部分 · idx_mapping 间接寻址
6. 对照 docs/design/model_runner_v2.md 第 6 节 "Better Compatibility with Speculative
   Decoding"，在代码里找出 expanded_idx_mapping 被真正用于间接寻址的 kernel，
   至少举三处，说明 kernel 内部是怎么从一条 logits 查到它所属请求的状态行的。
7. 说明如果不用这个间接层、改成把采样状态展开成 logits 形状，会多出哪些开销和复杂度。

第三部分 · Gumbel 采样
8. 讲 Gumbel-max 的数学等价性（为什么加 Gumbel 噪声取 argmax 等于按 softmax 采样），
   说明它省掉了什么。
9. 逐个讲 sample/gumbel.py 的函数：tl_rand32 / murmur3_hash32 / murmur3_uniform32 /
   murmur3_uniform64 / _uniform64_from_random53 / _log1p_neg_stable /
   gumbel_noised_argmax / gumbel_block_argmax / _gumbel_sample_kernel / gumbel_sample。
10. 重点讲"无状态 in-kernel RNG"：
    - 随机数是从哪几个输入算出来的（seed / pos / offset / domain 各是什么）？
    - 无状态带来哪三个好处（可复现、无需维护 RNG state、CUDA graph 友好）？
    - spec_step_seed() 为什么要给投机解码的每一步派生 seed？不派生会怎样？
11. 讲 apply_temperature / _temperature_kernel，以及 greedy 请求（temperature=0）
    是怎么被处理的（提示：sampling_states.any_greedy）。
12. 讲 sample/min_p.py 和 SamplingStates 里 top_k / top_p 的应用顺序。

第四部分 · logprobs
13. 对照设计文档的 "Efficient Top-K Logprobs"，讲清楚 MRV2 的"先选后算"流程：
    compute_topk_scores / _topk_log_softmax_kernel / _ranks_kernel 各做什么。
    再去 V1（vllm/v1/worker/gpu_model_runner.py 及其 sampler）找到对应逻辑，
    对比两者的显存峰值差异，用词表大小 V 和 batch 大小 B 写出各自的量级。
14. 讲 LogprobTokenIdsState 和 get_logprobs_dims()：
    "这一批没人要 logprobs"这条快路径是怎么短路的。
15. 讲 sample/prompt_logprob.py 的分块策略，说明"在单个 prompt 内部分块"
    解决的是什么显存问题。

第五部分 · 其余特性文件
16. 每个文件 10 行以内讲清楚：logit_bias.py / penalties.py / bad_words.py /
    thinking_budget.py / watermark.py / batch_shard.py / trace_replay.py。
    每个给：它改变 logits 的方式、它的每请求状态存在哪、它有没有专属 Triton kernel。
17. 讲 logits_processor/interface.py 的 LogitsProcessor 协议和 loader.py 的加载机制：
    用户自定义 logits processor 是怎么接进来的？它要实现哪些方法？
18. 讲 sample/output.py 的 SamplerOutput 和 SamplingMaskTensors：
    后者在什么配置下产出、from_logits / cat / to_cpu_nonblocking / tolists 各自的用途、
    为什么它和 flashinfer sampler 互斥。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 讲 Triton kernel 时必须说清楚 program_id 的维度含义和 block 划分。
- 涉及数值精度的地方（fp32/fp64、log1p 稳定性）要明确指出。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```

---

## 自测题

1. `Sampler` 的 logits processor 流水线顺序是什么？为什么 bias 和 penalties 不可交换？
2. `use_flashinfer` / `use_xpu_sampler` / `use_fp64_gumbel` 三者的互斥关系是什么？
3. `needs_logits_processing` 这个数组存在的意义是什么？它让什么路径被跳过？
4. 为什么 Gumbel-max 采样不需要物化 softmax？写出等价关系。
5. `murmur3_uniform32` 的随机数是从哪几个量算出来的？「无状态」带来哪三个好处？
6. `spec_step_seed(seed, step)` 为什么必须存在？所有草稿步共用一个 seed 会怎样？
7. MRV2 的 top-k logprobs 相比 V1 省的是什么？用词表大小 V 写出两者的量级。
8. 这一批没人要 logprobs 时，代码是在哪里短路的？
9. prompt logprobs 为什么需要「在单个 prompt 内部」分块？
10. `expanded_idx_mapping` 在采样 kernel 里具体怎么用？它让新增一个采样参数省掉了什么工作？
