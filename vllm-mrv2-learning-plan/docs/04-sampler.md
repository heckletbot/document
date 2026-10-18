# 采样器与模块化

采样器是重写。V1 的采样参数绑在「本步 batch 的形状」上，投机解码就把参数展开成 logits 的形状；模型族、连接器的差异则内联在 runner 里。V2 把采样参数做成和请求行一样的持久列，kernel 自己查表；差异逻辑收进单独的类，主循环只调接口。

## 采样

| | V1 `vllm/v1/sample/` | V2 `gpu/sample/` |
|--|----------------------|------------------|
| 参数存在哪 | `SamplingMetadata`，`refresh_metadata()` → `_make_sampling_metadata()` 在批变动后重建 | `SamplingStates` 等一组每请求一行的状态，`add_request` 写一行，`apply_staged_writes` 生效 |
| 和 logits 对齐 | 张量按 batch 行排，投机解码下展开到每条 logits | `expanded_idx_mapping[logits行] = 请求行`，`expanded_local_pos` 是行内位置。kernel 里查，不展开 |
| 随机数 | `generators: dict[int, torch.Generator]`，每请求一个有状态生成器，请求挪行时要跟着走 | `murmur3_uniform32/64`，由 `(seed, pos, offset, domain)` 现算。`spec_step_seed(seed, step)` 给每个草稿步派生 seed |
| 抽样 | `random_sample` / `sample_with_exponential_noise`，先变成概率再抽 | `gumbel_sample`：logits 加 Gumbel 噪声取 argmax，等价于按 softmax 抽样，不物化 softmax |
| logprobs | `compute_logprobs` 对全词表 `log_softmax`，再 `gather`。峰值约一批 × 词表 | `compute_topk_scores` 先从 logits 取 top-k，再只对选中的算 logprob。峰值约一批 × k |
| 快路径 | `all_greedy`、`no_penalties` 等是整批一个布尔 | `any_greedy(idx_mapping_np)`、`needs_logits_processing[req_idx]` 按本步请求算，没人要 logits 处理就整段跳过 |

间接寻址的收益在加参数的时候：temperature、penalty、bad words 都待在请求那一行，新增一个参数不用再写一遍「投机解码下怎么 repeat」。`Sampler.logits_processors` 的顺序是流水线顺序：`logit_bias`（加）→ `penalties`（乘）→ `bad_words` → 自定义，加和乘不可交换。

无状态 RNG 的三个结果：同样的 seed 和位置得到同样的数；没有生成器对象要在 `condense` 时搬运；捕获 CUDA graph 时没有外部 RNG 状态。`return_sampling_mask` 会关掉 flashinfer 和 XPU sampler，因为那两条路径产不出 mask；XPU kernel 只抽 fp32 噪声，配不上 `use_fp64_gumbel`。

`prompt_logprob.py` 把长 prompt 的 logprob 拆到 prompt 内部的块里算，避免一次物化整段。

## 差异放进单独的类

V1 的 `GPUModelRunner` 继承 `LoRAModelRunnerMixin`、`KVConnectorModelRunnerMixin`、`ECConnectorModelRunnerMixin`，连接器方法直接挂在 runner 上，主路径里到处判空。Mamba、M-RoPE、encoder 的分支是 runner 上的私有方法和主循环里的 `if`。投机入口是 `propose_draft_token_ids()`，`vllm/v1/spec_decode/` 下各 proposer 接口不统一。

V2 的 runner 只继承 `LoRAModelRunnerMixin`，其余用组合：

| 差异 | V2 的落点 | 加一个新变体要动哪 |
|------|-----------|-------------------|
| 模型族 | `gpu/model_states/`。`resolve_model_state_cls()` 按「模型自带类 → 有 CrossAttention → encoder-only → hybrid/attention-free → Default」选一个 `ModelState` | 新增一个子类，实现 `preprocess_state` / `prepare_attn` / `prepare_inputs_embeds` 等。主循环不改 |
| 投机方法 | `BaseSpeculator` 三个方法：`init_cudagraph_manager`、`capture`、`propose`。`init_speculator()` 是分派表 | 加一个类，在分派表加一行。`suffix`、`custom_class` 没有这一行，所以缺口是显式的 |
| KV / encoder-cache 连接器 | 空操作基类 + `Active*` 实现，`get_kv_connector()` 二选一。主路径直接调 `pre_forward` / `no_forward`，不判空 | 实现钩子。`no_forward` 保证零 token 的 step 仍做 KV 传输；`reset_capture_state` 保证捕获期状态不留到真实执行 |
| 多模态 / pooling / 图 / 并行 | `gpu/mm/`、`pool/`、`cudagraph_utils.py`、`dp_utils.py`、`pp_utils.py`、`pcp_manager.py`、`ubatch_utils.py` | 改对应文件 |

`MambaHybridModelState.preprocess_state` 的注释要求它跑在 `prepare_attn` 收集 `num_accepted_tokens` 之前，否则跨 block 的状态迁移对注意力元数据不可见。这类顺序在 V1 靠代码位置维持，在 V2 写在接口实现上。

PCP 只在 V2（`pcp_manager.py`，V1 的不支持清单里直接拒绝）。它要把一个长 prefill 按 rank 切开、为局部切片重建注意力元数据、让部分请求在所有 rank 上复制。V1 的持久表行序必须等于模型输入的行序，批一切开就要再 `condense` 一轮，局部批的状态也没有地方放。

## 往 V2 加一个特性

按这个顺序看，每条都对应前面的一处改法：

1. 要不要每请求的持久列？放 `RequestState` 还是这个特性自己的状态类？大且增量用 `StagedWriteTensor`，小且每步全量用 `UvaBackedTensor`，纯 CPU 决策用 numpy，纯 GPU 产出用 GPU 张量。
2. 主路径上有没有 `.item()`、`.tolist()`、非 pinned 的 `.to("cuda")`？有的话，要么搬进 kernel，要么搭上 `AsyncOutput` 那一次同步。
3. 参数要对齐到 logits 吗？走 `expanded_idx_mapping`，不要把状态 repeat 成 logits 的形状。
4. 会不会改变批的形状（token 数、请求数、活跃 LoRA 数）？会的话进 `BatchExecutionDescriptor`，否则图对不上。
5. 捕获用的 dummy 批和真实批行为一样吗？KV sharing fast prefill 就是反例：dummy 批不带 `fast_prefill`，捕获时那些层全量跑。
6. 它是模型族、投机方法，还是连接器？实现对应接口。分派表里没有的方法，运行时是明确的 `NotImplementedError`，不是静默走错分支。
7. 和已有特性互斥吗？影响 runner 选型的写进 `_get_v2_model_runner_unsupported_features()`；特性内部的约束写成单独的 `_validate_*`（自适应验证不兼容 LoRA、要求 full CUDA graph、不兼容 PP>1，就是这种）。
