# 13 · 外围集成：KV connector / 结构化输出 / LoRA / 多模态 / pooling

目标：把 MRV2 和外部子系统的接缝扫一遍。这些模块**单个都不复杂**，但它们是「一个特性要在 MRV2 落地需要改哪些地方」的最好样本——第 03 章要迁移特性时会回头查这一章。

---

## 要看的内容

| 模块 | 文件 |
|------|------|
| KV connector | `vllm/v1/worker/gpu/kv_connector.py`（`KVConnector` / `ActiveKVConnector`） |
| Encoder-cache connector | `vllm/v1/worker/gpu/ec_connector.py` |
| 结构化输出 | `vllm/v1/worker/gpu/structured_outputs.py` |
| LoRA | `vllm/v1/worker/gpu/lora_utils.py` + `mm/lora.py` + `LoRAModelRunnerMixin` |
| 多模态 | `vllm/v1/worker/gpu/mm/encoder_runner.py`、`mm/encoder_cache.py`、`mm/rope.py` |
| Pooling | `vllm/v1/worker/gpu/pool/pooling_runner.py` |
| 指标 | `vllm/v1/worker/gpu/metrics/logits.py` |
| 杂项 | `vllm/v1/worker/gpu/shutdown.py`、`cp_utils.py` |

---

## 要掌握的知识

### KV connector：空实现 + 活实现的对子

`KVConnector` 是**空操作基类**，`ActiveKVConnector` 是真实现，`get_kv_connector()` 二选一。这个模式让主路径上不需要写 `if connector is not None`。

钩子和第 04 章 step 流程的对应：

| 钩子 | 在 step 的哪一步 |
|------|----------------|
| `pre_forward()` | forward 之前，内部触发 `_start_load_kv()` |
| `finish_forward()` | forward 之后 |
| `post_forward()` | 更晚 |
| `finalize_step()` | 步末 |
| `no_forward()` | `total_num_scheduled_tokens == 0` 的短路分支 |
| `reset_capture_state()` | CUDA graph 捕获相关 |
| `set_disabled()` | 临时禁用 |

`no_forward()` 和 `reset_capture_state()` 这两个是最能体现「MRV2 特性集成要想清楚什么」的例子：**零 token 的 step 也要让 KV 传输发生**，**捕获期的连接器状态不能污染真实执行**。

### 结构化输出：一个 Triton kernel + 映射

`StructuredOutputsWorker.apply_grammar_bitmask()` 和 `_apply_grammar_bitmask_kernel`：把语法 bitmask 应用到 logits 上。`_build_grammar_mapping()` 建立请求到语法状态的映射，`grammar_invalid_drafts()` 处理投机草稿里不合语法的部分。

它被调用的位置就是第 04 章说的「两次 worker 调用之间」——`sample_tokens(grammar_output)` 的入参正是为它准备的。

### LoRA

- `LoraState`：每请求的 LoRA 归属（`add_request` / `remove_request` / `make_lora_inputs` / `get_activate_loras`）。
- `get_num_active_loras_for_dispatch()`：活跃 LoRA 数量会进入 `BatchExecutionDescriptor`（第 08 章），因为不同 LoRA 数量要不同的图。
- `get_lora_capture_cases()` / `create_lora_capture_hook()`：捕获期的 LoRA 枚举。
- `mm/lora.py`：多模态 LoRA。
- `GPUModelRunner` 继承 `LoRAModelRunnerMixin`（在 `vllm/v1/worker/lora_model_runner_mixin.py`，V1/V2 共用）。

### 多模态

- `EncoderCache`：按 `mm_hash` 缓存 encoder 输出，支持跨请求复用。
- `EncoderRunner`：`prepare_mm_inputs` / `execute_mm_encoder` / `gather_mm_embeddings` / `get_inputs_embeds`，还带自己的 CUDA graph（`has_cudagraph` / `capture`）和计时（`timed_encoder_operation`）、显存 profiling（`profile_encoder_cache`）。
- `mm/rope.py` · `RopeState`：M-RoPE（多模态位置编码）的持久状态，注意它也有 `apply_staged_writes()`——又一个第 05 章模式的实例。

### Pooling

`PoolingRunner` 走完全不同的输出路径（`pool()` → `AsyncPoolingOutput`，第 11 章）。`release_late_interaction_query_cache()` 对应 late-interaction 检索模型（ColBERT 类）。

### cp_utils

`prepare_dcp_local_seq_lens()` / `_dcp_local_seq_lens_kernel` / `cp_local_slot()`：DCP 下的局部序列长度计算，配合第 12 章。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：把 MRV2 和外围子系统的接缝扫一遍。每个模块都要回答同一组问题：
  (a) 它在 step 流程的哪些位置插入？（在 model_runner.py 里给出调用点行号）
  (b) 它有没有每请求的持久状态？存在哪、怎么写入和兑现？
  (c) 它和 CUDA graph 捕获有没有交互？
  (d) 它有没有专属的 Triton kernel？

第一部分 · KV connector
1. 讲 kv_connector.py 的 KVConnector / ActiveKVConnector 空实现+活实现模式，
   说明 get_kv_connector() 的选择逻辑，以及这个模式让主路径省掉了什么。
2. 按 step 顺序讲每个钩子：pre_forward / _start_load_kv / _build_load_kwargs /
   finish_forward / post_forward / finalize_step / no_forward /
   reset_capture_state / set_disabled，每个给 model_runner.py 里的调用点。
3. 重点讲 no_forward()：为什么零 token 的 step 也必须调用它？什么部署形态下会频繁命中？
4. 重点讲 reset_capture_state()：CUDA graph 捕获期的连接器状态为什么要重置？
   不重置会怎样？
5. 讲 ec_connector.py 的 ECConnector / ActiveECConnector，
   以及 model_runner._merge_ec_connector_no_forward() 在做什么。

第二部分 · 结构化输出
6. 讲 structured_outputs.py 的 StructuredOutputsWorker.apply_grammar_bitmask() 和
   _apply_grammar_bitmask_kernel：bitmask 的布局是什么、kernel 怎么把它应用到 logits。
7. 讲 _build_grammar_mapping() 建立的是什么映射，它怎么用到 idx_mapping。
8. 讲 grammar_invalid_drafts()：投机草稿里不合语法的 token 怎么处理？
9. 回到第 04 章：说明为什么 grammar_output 必须作为 sample_tokens 的入参传进来，
   而不能在 execute_model 里就处理掉。

第三部分 · LoRA
10. 讲 lora_utils.py 的 LoraState：每请求状态是什么、make_lora_inputs() 产出什么。
11. 讲 get_num_active_loras_for_dispatch()：活跃 LoRA 数为什么要进
    BatchExecutionDescriptor？不同 LoRA 数为什么需要不同的图？
12. 讲 get_lora_capture_cases() / create_lora_capture_hook() 在捕获期的作用。
13. 说明 GPUModelRunner 继承的 LoRAModelRunnerMixin 是 V1/V2 共用的，
    列出 V2 实际用到了它的哪些方法。
14. 讲 mm/lora.py 处理的多模态 LoRA 有什么特殊之处。

第四部分 · 多模态
15. 讲 mm/encoder_cache.py 的 EncoderCache：按什么 key 缓存、跨请求怎么复用、
    add_request / remove_request / free_encoder_cache / reset_mm_cache /
    reset_encoder_cache 的区别。
16. 讲 mm/encoder_runner.py 的 EncoderRunner 全流程：
    prepare_mm_inputs → execute_mm_encoder → gather_mm_embeddings → get_inputs_embeds，
    以及它自己的 CUDA graph（has_cudagraph / capture）、
    计时（timed_encoder_operation / get_encoder_timing_stats）、
    显存 profiling（profile_encoder_cache）。
17. 讲 mm/rope.py 的 RopeState：M-RoPE 的持久状态有哪些字段、
    init_prefill_positions / update_prefill_positions / prepare_positions /
    read_prefill_positions 各做什么、_prepare_rope_positions_kernel 的并行划分。
    指出它和第 05 章 StagedWriteTensor 模式的对应（apply_staged_writes）。
18. 讲 attn_utils.compute_mm_prefix_ranges() 和多模态的关系。

第五部分 · pooling 与杂项
19. 讲 pool/pooling_runner.py 的 PoolingRunner：
    它的输出路径和生成模型有什么本质不同、_get_pooling_metadata() 组装了什么、
    dummy_pooler_run 怎么做、release_late_interaction_query_cache() 服务什么模型。
20. 讲 metrics/logits.py 的 get_num_nans()，以及它和第 11 章 AsyncOutput 的 num_nans 的关系。
21. 讲 cp_utils.py 的 prepare_dcp_local_seq_lens() / cp_local_slot()。
22. 讲 shutdown.py 的 free_before_shutdown() 释放了什么。

最后给我一张总表：
23. 把本章所有模块列成表：模块 | step 插入点 | 有无每请求持久状态 | 有无 CUDA graph 交互 |
    有无专属 Triton kernel。这张表我会当作"往 MRV2 迁移新特性"的检查清单用，
    所以请在表后补一段：从这些模块的共性看，一个新特性接入 MRV2 典型要动哪几处。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```

---

## 自测题

1. `KVConnector` 和 `ActiveKVConnector` 的空/活实现模式让主路径省掉了什么？
2. 零 token 的 step 为什么还要调 `no_forward()`？什么部署形态会频繁命中？
3. `reset_capture_state()` 防的是什么？捕获期的连接器状态污染了会怎样？
4. grammar bitmask 为什么必须在 `execute_model` 和 `sample_tokens` 之间注入？
5. `grammar_invalid_drafts()` 处理的是什么场景？
6. 活跃 LoRA 数为什么要进 CUDA graph 描述符？
7. `EncoderCache` 按什么 key 缓存？它怎么实现跨请求复用？
8. `RopeState` 为什么也需要 `apply_staged_writes()`？它属于第 05 章的哪个模式？
9. pooling 模型的输出路径和生成模型有什么本质不同？
10. 根据本章的共性总结：一个新特性要接进 MRV2，典型要动哪几处？（这是第 03 章迁移工作的入口）
