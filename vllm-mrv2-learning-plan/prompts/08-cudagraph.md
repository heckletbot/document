# 08 · CUDA Graph 管理、warmup 与显存 profiling

目标：读懂 MRV2 的显式图管理——什么形状会被捕获、运行时怎么 dispatch、full graph 和 piecewise graph 怎么分工、以及投机解码的多步草稿怎么被融进一张图。对应设计文档第 8、9 条。

---

## 要看的内容

1. `vllm/v1/worker/gpu/cudagraph_utils.py` 全文（约 1100 行，本章主体）
   - `BatchExecutionDescriptor`、`AttentionState`、`_is_compatible()`
   - `CudaGraphManager`：`_init_candidates` / `capture` / `dispatch` / `run_fullgraph` / `run_pw_graph` / `release_graphs` / `needs_capture`
   - `ModelCudaGraphManager`（子类，覆盖 `capture` / `run_fullgraph`）
   - `prepare_inputs_to_capture()`、`profile_cudagraph_memory()`、`set_dcp_capture_context()`
2. `vllm/v1/worker/gpu/model_runner.py` · `capture_model()` / `needs_cudagraph_capture()` / `profile_cudagraph_memory()` / `_dummy_run()` / `profile_run()`
3. `vllm/v1/worker/gpu/warmup.py` · `run_mixed_prefill_decode_warmup()` / `warmup_kernels()`
4. `docs/design/model_runner_v2.md` 第 8、9 节（含 Fused Multi-Step Draft Decoding）
5. `vllm/v1/worker/gpu/dp_utils.py` · `dispatch_cg_and_sync_dp()`（捕获形状和 DP 同步是耦合的）

---

## 要掌握的知识

### `BatchExecutionDescriptor` 是整章的中心

它描述「这一步要用哪种执行形态」：token 数、request 数、cg_mode（`NONE` / `PIECEWISE` / `FULL` 系列）、是否 uniform decode、活跃 LoRA 数、是否 ubatch 等。
- **捕获时**：按候选列表枚举描述符，一个描述符捕一张图。
- **运行时**：`dispatch()` 把实际批形状匹配到某个已捕获的描述符，`_is_compatible()` 判断兼容性。

理解这里的关键是「一张图服务一个描述符，而实际批会被 padding 到描述符的形状」。`InputBatch.max_query_len` 字段的注释正好解释了一个微妙点：当描述符承诺的 query 长度超过本批实际的最大值时，要把这个上界写进去，**否则注意力元数据对这张图的其他 replay 就不成立了**。

### full graph 和 piecewise graph 的分工

- **PIECEWISE**：`run_pw_graph()`，跑 torch.compile 产出的分段图或 breakable cudagraph。
- **FULL**：`run_fullgraph()`，整个 forward 进一张图，由 `ModelCudaGraphManager` 用标准 PyTorch API 捕获。

设计文档强调 MRV2 的图管理是**显式**的：捕获用独立路径（`capture_model()`），不再寄生在 `dummy_run` 里。`needs_cudagraph_capture()` 决定要不要捕。

### Fused Multi-Step Draft Decoding（第 9 节的重点）

自回归投机解码一个 scheduler step 里要跑若干**互相依赖**的草稿步。融合路径的做法是：

- 把 prefill 之后的所有草稿步**捕进一张 full graph**，而不是每个草稿 token replay 一次。
- 注意力元数据在循环**之前**构建一次；步间变化的张量保持**地址稳定**，原地更新。

代价是后端必须实现 `AttentionMetadataBuilder.update_draft_decode_metadata()`，用来在草稿输入推进之后更新/失效那些派生状态（scheduler metadata、稀疏索引之类）。这个钩子有三条极其容易踩坑的性质，设计文档写得很明确：

1. 钩子**在捕获期间运行**，所以只有它发出的 GPU 操作被录进图；replay 时它的 Python 代码**不会再执行**。
2. 因此实现必须只用 capture-safe 的操作，且所有会被 replay 的张量状态必须放在持久存储里。
3. 只有当**每一个** draft attention group 都声明 `supports_draft_decode_metadata_update` 时才启用融合路径，否则回退到「每步重建元数据」。位置固定的草稿模型不需要这个更新。

另外设计文档特别提醒：启用某后端前要审计**所有**派生元数据，包括从父 builder 继承的、以及辅助注意力后端持有的。

### 显存 profiling 的两条线

- `profile_run()`：走 `execute_model(dummy_run=True, is_profile=True)`，量出模型本身的峰值显存。
- `profile_cudagraph_memory()`：单独量图捕获要吃多少显存；`capture_model(profile_only=True)` 支持只量不捕。

这是设计文档第 8 条的落地：`dummy_run` 不再是万能抽屉。

### warmup

- `run_mixed_prefill_decode_warmup()`：构造混合 prefill/decode 批跑一遍，顺便暴露只在混合批出现的问题；里面有自己的 block 分配逻辑（`_alloc_blocks` / `_warmup_block_counter` / `_reserved_block_count`）。
- `warmup_kernels()`：触发 kernel 的 autotune / JIT 编译，避免首个真实请求付编译开销。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲透 MRV2 的 CUDA graph 管理、warmup 和显存 profiling。

第一部分 · 描述符与候选集
1. 讲 BatchExecutionDescriptor（vllm/v1/worker/gpu/cudagraph_utils.py）的每个字段含义，
   以及它同时服务"捕获"和"dispatch"两件事的方式。
2. 讲 CudaGraphManager._init_candidates()：候选描述符是怎么枚举出来的？
   受哪些配置影响（cudagraph_mode、cudagraph_capture_sizes、LoRA、ubatch、DP）？
3. 讲 _is_compatible() 和 dispatch()：运行时一个真实批是怎么被匹配到某张图的？
   匹配失败会怎样？
4. 讲 AttentionState 这个 NamedTuple 在兼容性判断里的角色。
5. 解释 InputBatch.max_query_len 这个字段的注释（"Set when a cudagraph descriptor
   promises a query length this batch's own split does not reach"），
   说清楚不设它会导致什么错误。

第二部分 · 捕获与执行
6. 讲 capture() 的完整流程（CudaGraphManager 和 ModelCudaGraphManager 两个版本都要讲，
   说明子类覆盖了什么）。包括：输入是怎么准备的（prepare_inputs_to_capture）、
   捕获用哪个 stream（_capture_stream）、捕获顺序有没有要求。
7. 讲 run_fullgraph() 和 run_pw_graph() 的区别，以及 init_breakable_cg_runner() 的用途。
   在 model_runner.py 的 execute_model 里指出三种 forward 分派的分叉点。
8. 讲 LoRA 相关的 _build_lora_dispatch_map() / _resolve_effective_loras() 和
   lora_utils.get_lora_capture_cases() / create_lora_capture_hook()：
   LoRA 数量是怎么进入描述符的？
9. 讲 _maybe_ubatch_twin()：ubatch 开启时为什么要捕"孪生"图？
10. 讲 release_graphs() 和 needs_capture() 的调用时机。
11. 讲 set_dcp_capture_context() 解决什么问题。

第三部分 · Fused Multi-Step Draft Decoding
12. 对照 docs/design/model_runner_v2.md 第 9 节，在代码里找到融合草稿解码的实现位置
    （提示：vllm/v1/worker/gpu/spec_decode/ 下的 cudagraph 相关文件，
     以及 cudagraph_utils.py 里的草稿相关分支）。
13. 讲清楚 AttentionMetadataBuilder.update_draft_decode_metadata() 这个钩子：
    - 它在什么时机被调用（捕获期 vs replay 期）？
    - 为什么"replay 时它的 Python 代码不再执行"这件事会让实现变得危险？
      举一个会出错的错误实现例子。
    - supports_draft_decode_metadata_update 这个标志在哪里被检查、
      不满足时回退到什么路径。
    - 在仓库里找出已经实现了这个钩子的后端，列出来。
14. 说明"位置固定的草稿模型不需要这个更新"是什么意思，哪些 speculator 属于这一类。

第四部分 · profiling 与 warmup
15. 讲 model_runner.py 的 profile_run() / profile_cudagraph_memory() /
    capture_model(profile_only=True) 三者的分工，对照设计文档第 8 节，
    说明 MRV2 把 V1 的 dummy_run 四类职责拆到了哪里。
16. 讲 cudagraph_utils.profile_cudagraph_memory() 的测量方法。
17. 讲 warmup.py 的 run_mixed_prefill_decode_warmup()：
    它构造的是什么形状的批、为什么要混合、_alloc_blocks / _warmup_block_counter /
    _reserved_block_count 各自在算什么。
18. 讲 warmup_kernels() 触发了哪些 kernel 的预热。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 凡是涉及"捕获期执行 vs replay 期执行"的区别，每次都要明确写出来。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```

---

## 自测题

1. `BatchExecutionDescriptor` 有哪些字段？它在捕获和运行时分别扮演什么角色？
2. 候选描述符集是怎么枚举出来的？哪些配置会让它变大？
3. 一个真实批怎么被匹配到已捕获的图？匹配不上会怎样？
4. `InputBatch.max_query_len` 什么时候需要被设置？不设会坏在哪？
5. PIECEWISE 和 FULL 两种图分别由哪个方法执行？各自适合什么场景？
6. MRV2 的图捕获为什么不走 `dummy_run`？V1 的 `dummy_run` 四类职责在 MRV2 分别落到哪里？
7. 融合多步草稿解码省掉了什么？代价是什么？
8. `update_draft_decode_metadata()` 在 replay 时会被执行吗？这对实现者意味着什么约束？
9. `supports_draft_decode_metadata_update` 不满足时，MRV2 回退到什么路径？
10. `run_mixed_prefill_decode_warmup()` 为什么要专门构造混合批？纯 decode warmup 漏掉了什么？
