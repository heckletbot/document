# 07 · Attention 元数据、KV cache 与 `ModelState` 抽象

目标：搞清楚 MRV2 怎么把「一批长短不一的请求」翻译成注意力后端要的元数据，KV cache 怎么分配，以及不同模型族（纯注意力 / Mamba 混合 / encoder-only / encoder-decoder）的差异是怎么被隔离在 `ModelState` 里的。

---

## 要看的内容

1. `vllm/v1/worker/gpu/attn_utils.py` 全文
   - `get_kv_cache_spec()` / `get_shared_kv_cache_layers()` / `init_attn_backend()` / `init_kv_cache()`
   - `build_attn_metadata()` / `build_slot_mappings_by_layer()`
   - `get_attn_cg_support()` / `get_query_lens_mismatch_unsupported_backend()` / `get_varlen_cudagraph_unsupported_backend()`
   - KV sharing fast prefill：`FastPrefillHelper` / `FastPrefillBatchMetadata` / `get_kv_sharing_fast_prefill_eligible_layers()`
2. `vllm/v1/worker/gpu/model_states/__init__.py` · `init_model_state()` / `resolve_model_state_cls()`——分派逻辑很短，先读
3. `vllm/v1/worker/gpu/model_states/interface.py` · `ModelState` 抽象基类（**本章的接口契约**）
4. `vllm/v1/worker/gpu/model_states/default.py`、`mamba_hybrid.py`（最有信息量的两个实现）
5. `vllm/v1/worker/gpu/model_runner.py` · `get_kv_cache_spec()` / `initialize_kv_cache()` / `_init_kv_zero_meta()` / `prepare_attn()` / `prepare_dummy_attn()`

---

## 要掌握的知识

### `ModelState` 是什么

它是「模型族特定行为」的插槽。`resolve_model_state_cls()` 的分派顺序（短路，**顺序有意义**）：

| 顺位 | 条件 | 选中的类 |
|------|------|---------|
| 1 | 模型自己定义了 `get_model_state_cls()` | 模型自带的 |
| 2 | 模型里有 `CrossAttention` 模块 | `EncoderDecoderModelState`（Whisper 等） |
| 3 | 有 `ENCODER_ONLY` 的 attention 层 | `EncoderOnlyModelState`（非因果、不需要 KV cache） |
| 4 | `is_hybrid` 或 `is_attention_free` | `MambaHybridModelState` |
| 5 | 其余 | `DefaultModelState` |

还有一道校验：`enable_prompt_embeds` 但选中的类 `supports_prompt_embeds` 为假时直接报错——注释写明是为了避免**静默忽略**这个开关。

`ModelState` 的接口分几组：
- 生命周期：`add_request` / `remove_request` / `apply_staged_writes`
- KV cache：`initialize_kv_cache`
- 前后处理：`preprocess_state` / `postprocess_state`
- 输入：`prepare_inputs_embeds` / `dummy_inputs_embeds` / `prepare_inputs` / `prepare_dummy_inputs`
- 多模态：`execute_mm_encoder` / `gather_mm_embeddings`
- 注意力：`prepare_attn`、`ModelSpecificAttnMetadata`（`get_extra_common_attn_kwargs` / `get_extra_attn_kwargs`）
- CUDA graph：`capture_inner_cudagraphs` / `get_additional_cg_support`
- 采样：`custom_sampler`

注意 `MambaHybridModelState` 继承自 `DefaultModelState` 而不是直接继承接口——它主要是在 `preprocess_state` / `prepare_attn` / `postprocess_state` 上加东西。第 04 章提到的「Mamba 的 align 前处理：在 forward 前把循环状态跨 block 边界迁移」就在 `preprocess_state` 里，且注释强调它**必须在 `prepare_attn` 收集 `num_accepted_tokens` 之前**跑，否则边界重置对注意力元数据不可见。这类「顺序约束」是本章最该记的东西。

### attention 元数据的构造链

```text
prepare_attn(input_batch)
  ├─ block_tables.gather_block_tables()      # 按本步顺序 gather
  ├─ block_tables.compute_slot_mappings()    # token → KV 槽位
  ├─ build_slot_mappings_by_layer()          # 按层分发（不同层可能属于不同 KV group）
  └─ build_attn_metadata()                   # 调各后端的 AttentionMetadataBuilder
```

要理解的核心是**KV cache group**：不同层可能有不同的 KV cache 规格（滑窗 vs 全注意力、不同 head 数、Mamba 的状态），所以 block table 和 slot mapping 都是「按 group 一份」，再按层映射回去。`get_shared_kv_cache_layers()` 处理层间共享 KV 的情况。

### CUDA graph 相关的三个探测函数

`get_attn_cg_support()`、`get_query_lens_mismatch_unsupported_backend()`、`get_varlen_cudagraph_unsupported_backend()`——它们回答的是「这个注意力后端能不能被塞进 full CUDA graph，以及在什么形状约束下」。第 08 章的 `CudaGraphManager` 会消费这些结论。`AttentionCGSupportInfo.narrow()` 则是把多个后端的能力取交集。

### KV sharing fast prefill

`FastPrefillHelper` / `FastPrefillBatchMetadata`：某些层共享上层的 KV，于是 prefill 时这些层可以只算一部分 token。`InputBatch.fast_prefill` 字段的注释写明：**dummy（CUDA graph 捕获）批不会带这个字段，捕获时 KV sharing 层要全量跑**。这又是一条「捕获路径和真实路径必须对齐」的约束。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲透 MRV2 的注意力元数据构造、KV cache 初始化和 ModelState 抽象。

第一部分 · ModelState
1. 读 vllm/v1/worker/gpu/model_states/__init__.py，把 resolve_model_state_cls() 的
   分派顺序画成决策树，每个分支标注条件来源（哪个 config 字段 / 哪个模块类型）。
   说明为什么顺序不能调换。
2. 说明 supports_prompt_embeds 的那道校验在防什么（照注释讲）。
3. 读 model_states/interface.py，把 ModelState 的所有抽象方法按用途分组列表，
   每个方法给：调用方（在 model_runner.py 的哪一步）、默认实现在不在、
   谁覆盖了它（列出 model_states/ 下所有覆盖该方法的子类）。
4. 讲 ModelSpecificAttnMetadata 的两个钩子（get_extra_common_attn_kwargs /
   get_extra_attn_kwargs）各自在哪里被消费。
5. 对比 DefaultModelState 和 MambaHybridModelState：后者覆盖了哪些方法、各加了什么。
   重点讲 preprocess_state() 里的 Mamba "align" 前处理：
   它做什么、为什么必须在 prepare_attn 收集 num_accepted_tokens 之前跑、
   如果顺序颠倒会出现什么错误。
6. 简单过一遍 encoder_only / encoder_decoder / prompt_embeds / mm_pruning / recoverssm
   这几个 ModelState 各自的适用模型和核心差异（每个 5 行以内）。

第二部分 · KV cache
7. 讲 attn_utils.py 的 get_kv_cache_spec() 怎么从 vllm_config 推导出每层的 KVCacheSpec，
   以及 get_shared_kv_cache_layers() 处理的是什么场景。
8. 讲 model_runner.py 的 initialize_kv_cache() 完整流程，包括它和 attn_utils.init_kv_cache()、
   init_attn_backend() 的分工，以及 _init_kv_zero_meta() 的用途。
9. 讲清楚 "KV cache group" 这个概念：为什么会有多个 group，block table 和 slot mapping
   是怎么按 group 组织再映射回层的。结合第 05 章 BlockTables 的指针张量讲。

第三部分 · attention 元数据
10. 逐段讲 model_runner.py 的 prepare_attn() 和 prepare_dummy_attn()，
    把调用链一路追到 attn_utils.build_attn_metadata() 和 build_slot_mappings_by_layer()。
11. 说明 build_attn_metadata() 怎么和各注意力后端的 AttentionMetadataBuilder 对接，
    InputBatch 的哪些字段被喂进去（至少覆盖 query_start_loc、seq_lens、
    seq_lens_cpu_upper_bound、max_query_len、dcp_local_seq_lens）。
12. 解释 seq_lens 和 seq_lens_cpu_upper_bound 为什么要分开，哪些后端需要 CPU 上界。

第四部分 · CUDA graph 能力探测与 fast prefill
13. 讲 get_attn_cg_support() / get_query_lens_mismatch_unsupported_backend() /
    get_varlen_cudagraph_unsupported_backend() 三个函数各自回答什么问题，
    结论被谁消费（指到 cudagraph_utils.py 的具体位置）。
    说明 AttentionCGSupportInfo.narrow() 的语义。
14. 讲 KV sharing fast prefill：FastPrefillHelper.prepare() 做什么、
    get_kv_sharing_fast_prefill_eligible_layers() 怎么筛层、
    为什么 InputBatch.fast_prefill 在 dummy 批上必须缺席（照字段注释讲）。
15. 讲 compute_mm_prefix_ranges() 的用途。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 凡是代码注释里写了"必须在 X 之前/之后"的顺序约束，全部单独列出来并解释违反的后果。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```

---

## 自测题

1. `resolve_model_state_cls()` 的五个分支是什么？为什么 `CrossAttention` 的判断要排在 `is_hybrid` 之前？
2. `enable_prompt_embeds` 碰上不支持的 `ModelState` 会怎样？为什么不是静默忽略？
3. `ModelState` 的 `preprocess_state` 和 `postprocess_state` 分别在一个 step 的哪两个位置被调用？
4. Mamba 的 align 前处理做什么？为什么它和 `prepare_attn` 之间有严格顺序？
5. 什么是 KV cache group？举两个会产生多个 group 的场景。
6. `prepare_attn()` 一共产出哪几样东西？它们分别被谁消费？
7. `seq_lens` 和 `seq_lens_cpu_upper_bound` 为什么要分开维护？
8. 三个 CUDA graph 能力探测函数各自回答什么问题？`narrow()` 是取交集还是并集？
9. KV sharing fast prefill 省掉了什么计算？为什么 CUDA graph 捕获时不能用它？
10. `MambaHybridModelState` 为什么继承 `DefaultModelState` 而不是直接继承 `ModelState`？
