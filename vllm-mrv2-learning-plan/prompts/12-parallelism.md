# 12 · 并行维度：DP / PP / PCP / micro-batch / EPLB

目标：搞清楚 MRV2 在 model runner 这一层为每种并行做了什么。注意边界：**张量并行（TP）基本不在 model runner 层体现**（它在 layer 里），本章讲的是剩下那几种会改变「批形状」或「执行时序」的并行。

---

## 要看的内容

1. `vllm/v1/worker/gpu/dp_utils.py` · `DPSyncState` / `sync_cudagraph_and_dp_padding()` / `dispatch_cg_and_sync_dp()`
2. `vllm/v1/worker/gpu/pp_utils.py` · `PPHandler` / `PendingRecv` / `compute_need_sampled_mask()`
3. `vllm/v1/worker/gpu/pcp_manager.py` · `PCPManager`（900 行，V2-only 的大件）
4. `vllm/v1/worker/gpu/ubatch_utils.py` · `UBatchRunner` / `create_ubatch_slices()` / `merge_ubatch_outputs()`
5. `vllm/v1/worker/gpu/eplb_utils.py` · `EPLBController` / `step_eplb_after()` / `preserve_serving_state()`
6. `vllm/v1/worker/gpu/cp_utils.py`
7. 调用侧：`model_runner.py` 的 `execute_model()` 里这些模块的插入点

---

## 要掌握的知识

### DP：图形状和 padding 必须跨 rank 一致

`dispatch_cg_and_sync_dp()` 把两件事绑在一起做，这不是偶然：**所有 DP rank 必须选同一个 CUDA graph 描述符**，否则集合通信会错配。`sync_cudagraph_and_dp_padding()` 就是那次跨 rank 协商。`DPSyncState` 保存协商结果，一路带到步末。

MoE + DP 的空转 forward（某个 rank 这一步没活干，但仍要参与 all2all）也是这条路径的职责——第 04 章提到的 `batch_desc.num_tokens == 0` 短路（所有 rank 都没活）就是协商的产物。

### PP：跨 step 的状态更新与发送/接收

`PPHandler` 的方法名已经把职责说清楚了：

| 方法 | 职责 |
|------|------|
| `receive()` / `broadcast()` | `IntermediateTensors` 的收发 |
| `get_prev_sampled_outputs()` | 取上一轮的采样结果（PP 下采样只在最后一级 rank 发生，前面的 rank 需要回灌） |
| `broadcast_drafts()` | 草稿 token 的广播 |
| `relay_aux_hidden_states()` / `configure_aux_hidden_state_relay()` | EAGLE 之类需要的辅助 hidden states 透传 |
| `on_req_idx_freed()` | 请求槽位释放时清理 pending 状态 |

`compute_need_sampled_mask()` 回答「这一批里哪些请求本步需要采样结果」。`PendingRecv` 是在途接收。

`model_runner.update_pp_decode_requests()`（第 04 章 step 开头第一件事）就是 PP 专用的——把上一轮还没落地的 decode 请求补齐。

第 03 章的两条缺口在这里：PP + `external_launcher` 在 V2 还没有 `broadcast_pp_output` 等价物；而 PP + 异步调度**只有 V2 能跑**。

### PCP（Prefill Context Parallel）：V2 独有

把**一个长 prompt 的 prefill 切到多个 rank** 上算。`PCPManager` 的核心方法：

- `partition_batch()` / `_build_batch_layout()` / `_get_rank_segments()` / `_reorder_segments()`：怎么切、每个 rank 拿哪段（`RankSegment`）
- `global_batch` / `input_buffers`：本 rank 的局部 `InputBatch` 和全局批的关系
- `replicated_requests()`：哪些请求在所有 rank 上复制而不是切分
- `prepare_attn()` / `prepare_slot_mappings()`：注意力元数据要按局部切片重建
- `validate_config()`：配置约束
- `moe_is_padding()`：MoE 下的 padding 掩码要改

`InputBatch` 里的 `dcp_local_seq_lens` / `dcp_local_seq_lens_cpu_upper_bound` 字段注释提到「PCP+DCP 下才设置」——PCP（prefill 侧）和 DCP（decode 侧）是两种不同的 context parallel，别混。

### micro-batch / DBO

`UBatchRunner` 把一个批切成多个 micro-batch 重叠执行（典型目的是让通信和计算重叠）。

- `create_ubatch_slices()` / `_slice_input_batch()` / `_slice_query_start_loc()` / `_slice_seq_lens()`：怎么切 `InputBatch`
- `_make_forward_contexts()`：每个 micro-batch 一个 forward context
- `run()` / `begin_capturable_run()` / `abort_pending_run()`：执行与捕获
- `merge_ubatch_outputs()`：结果合并
- `maybe_build_ubatch_runner()`：要不要启用

和 CUDA graph 的交互见第 08 章的 `_maybe_ubatch_twin()`。第 03 章提过：`use_ubatching` 开启时 V2 的不支持清单会并进 `_get_dbo_unsupported_features()`。

### EPLB：专家并行负载均衡

`EPLBController` 在 model runner 层的职责是**在合适的时机触发重平衡，并保证重平衡时的状态安全**：

- `step_eplb_after()` 是个装饰器，挂在 `execute_model` / `sample_tokens` 上（第 04 章能看到 `@step_eplb_after()`）。
- `preserve_serving_state()` / `_remove_all_requests()`：重平衡期间要临时清空请求状态再恢复。
- `maybe_register_model()` / `maybe_register_speculator()`：草稿模型也可能参与 EPLB（`draft_model_supports_eplb()`、`eplb_draft_model_name()`）。
- `suppress()` 上下文 + `eep_eplb_suppressed` 属性：某些阶段要抑制重平衡。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲清楚 MRV2 在 model runner 层为各种并行做了什么。

先回答一个前置问题：
0. 张量并行（TP）为什么基本不在 model runner 层出现？它体现在哪一层？
   （只要 5 行，用来划清本章边界。）

第一部分 · DP
1. 讲 dp_utils.py 的 dispatch_cg_and_sync_dp()：
   为什么 CUDA graph 描述符的选择必须和 DP 同步绑在一起做？
   如果各 rank 各选各的，会在哪里炸？
2. 讲 sync_cudagraph_and_dp_padding() 的协商过程：通信了什么、怎么取共识。
3. 讲 DPSyncState 保存了什么、在 step 的后续阶段被谁消费。
4. 讲 MoE + DP 下"某个 rank 这一步没有请求"时的空转 forward：
   代码走的是哪条路径？和 model_runner.execute_model 里
   batch_desc.num_tokens == 0 的短路是什么关系？

第二部分 · PP
5. 讲 PPHandler 的每个方法职责和调用时机（在 model_runner.py 里指出调用点）：
   receive / broadcast / get_prev_sampled_outputs / broadcast_drafts /
   relay_aux_hidden_states / configure_aux_hidden_state_relay / on_req_idx_freed。
6. 讲 compute_need_sampled_mask() 回答的问题，以及 PendingRecv 的作用。
7. 讲 model_runner.update_pp_decode_requests()：它解决 PP 下的什么时序问题？
   为什么它是 execute_model 的第一件事？
8. 结合 vllm/config/vllm.py 的两份缺口清单，说明：
   - 为什么 PP + external_launcher 在 V2 还不行（去 V1 找 broadcast_pp_output 对照）；
   - 为什么 PP + 异步调度只有 V2 能跑。
9. 讲 InputBatch.max_seq_len_np 字段的注释（"only populated for non-dummy
   pipeline-parallel batches"），说明 PP 为什么需要它。

第三部分 · PCP
10. 先用 10 行讲清楚 PCP 和 DCP 的区别（分别切什么、在 prefill 还是 decode）。
11. 讲 PCPManager 的切分逻辑：partition_batch / _build_batch_layout /
    _get_rank_segments / _reorder_segments / _iter_rank_chunks，
    用一个具体例子（比如 1 个 8192 token 的 prefill 请求 + 2 个 decode 请求，PCP=2）
    把每个 rank 拿到的 RankSegment 写出来。
12. 讲 replicated_requests()：哪些请求不切分、为什么。
13. 讲 PCP 下注意力元数据和 slot mapping 的重建（prepare_attn / prepare_slot_mappings /
    get_dummy_block_tables / get_dummy_slot_mappings）。
14. 讲 moe_is_padding() 为什么要专门处理。
15. 讲 validate_config() 的所有约束。
16. 讲 InputBatch 里 dcp_local_seq_lens 和 dcp_local_seq_lens_cpu_upper_bound
    两个字段在 PCP+DCP 组合下的含义。

第四部分 · micro-batch（DBO）
17. 讲 UBatchRunner 的完整流程：maybe_build_ubatch_runner 的启用条件、
    create_ubatch_slices 的切分规则、_slice_input_batch 怎么切
    （query_start_loc 和 seq_lens 的切分为什么要单独处理）、
    _make_forward_contexts、run / begin_capturable_run / abort_pending_run、
    merge_ubatch_outputs。
18. 讲 ubatch 和 CUDA graph 的交互（cudagraph_utils._maybe_ubatch_twin），
    说明"孪生图"是什么。
19. 结合 vllm/config/vllm.py 的 _get_dbo_unsupported_features()，
    列出开启 ubatching 后哪些特性会被禁用。

第五部分 · EPLB
20. 讲 EPLBController 的生命周期：prepare_load / maybe_register_model /
    maybe_register_speculator / maybe_start_async_loop / step / prepare_forward /
    setup_from_mapping。
21. 讲 step_eplb_after() 装饰器：它挂在哪些方法上、is_dummy / is_profile 怎么影响它。
22. 讲 preserve_serving_state() 和 _remove_all_requests()：
    重平衡期间为什么要清空请求状态？不清会出什么问题？
23. 讲 suppress() 和 eep_eplb_suppressed 属性的用途。
24. 讲 draft_model_supports_eplb() / eplb_draft_model_name()：草稿模型参与 EPLB 的条件。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 每次提到一种并行，先说清楚它切分的是"批"、"序列"还是"模型"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```

---

## 自测题

1. 为什么 CUDA graph 描述符的选择必须跨 DP rank 协商？不协商会在哪里炸？
2. MoE + DP 下某个 rank 没有请求时会发生什么？为什么不能直接跳过这一步？
3. PP 下采样发生在哪一级 rank？其他 rank 怎么拿到采样结果？
4. `update_pp_decode_requests()` 解决的是什么时序问题？
5. PP + `external_launcher` 在 V2 缺的是什么机制？PP + 异步调度为什么只有 V2 能跑？
6. PCP 和 DCP 分别切什么？各自在 prefill 还是 decode 阶段？
7. PCP 下哪些请求会被复制到所有 rank 而不是切分？为什么？
8. micro-batch 的目的是什么？它和 CUDA graph 的「孪生图」是什么关系？
9. EPLB 重平衡时为什么要先清空请求状态？`preserve_serving_state()` 保证了什么？
10. `step_eplb_after()` 挂在哪些方法上？`is_dummy=True` 时它的行为有什么不同？
