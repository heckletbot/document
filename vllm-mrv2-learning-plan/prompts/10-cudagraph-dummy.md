# 10 · CUDA graph 与 `dummy_run`

> 轴二 · 异步调度

**对照的那件事**：图的捕获与分派，以及 `dummy_run` 这个在 V1 里被当成万能抽屉的东西。

对应设计文档第 8、9 条。放在轴二，是因为「哪种执行形态」这个决策必须在每步开头就无同步地做完。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 图管理 | 散落在 runner 上：`_capture_cudagraphs()`、`_warmup_and_capture()`、`_check_and_update_cudagraph_mode()` | `gpu/cudagraph_utils.py` · `CudaGraphManager` / `ModelCudaGraphManager`（专门的类） |
| 形态决策 | `_determine_batch_execution_and_padding()`、`_is_uniform_decode()`、`_allow_microbatching()` | `BatchExecutionDescriptor` + `dispatch()` + `dp_utils.dispatch_cg_and_sync_dp()` |
| 捕获入口 | `capture_model()` → `_warmup_and_capture()` → `_capture_cudagraphs()`，内部调 `_dummy_run()` | `capture_model()` 独立路径，不走 `execute_model` |
| `dummy_run` | `_dummy_run()`（约 420 行）：profiling + compile + 捕获 + warmup + DP 空转 | `_dummy_run()` 委托给 `execute_model(dummy_run=True, ...)` |
| profiling | `profile_run()`、`profile_cudagraph_memory()`、`_init_minimal_kv_cache_for_profiling()`、`_cleanup_profiling_kv_cache()` | `profile_run()` 走 `execute_model`；`profile_cudagraph_memory()` / `capture_model(profile_only=True)` 独立 |
| 多步草稿 | 每个草稿 token 单独 replay | 可把全部草稿步融进一张 full graph（需后端支持） |

---

## 差异的因果

### V1 的 `dummy_run` 为什么会膨胀

因为「跑一次假的 forward」这个能力同时被四件事需要：显存 profiling、`torch.compile` 触发、CUDA graph 捕获、warmup（再加 EP+DP 的空转 forward）。每件事都往同一个函数里加分支和参数，于是它和真实的 `execute_model` 逐渐长成两套不同的逻辑——**行为漂移导致的 bug 就从这里来**。

### V2 的拆法

1. `execute_model()` 自己支持 dummy 模式（`dummy_run` / `skip_attn_for_dummy_run` / `is_profile` / `randomize_inputs` / `valid_dummy_state_slots` 这几个参数），且**不影响状态**。
2. `_dummy_run()` / `profile_run()` 变成对 `execute_model()` 的薄封装。
3. CUDA graph 捕获走**独立**路径（`capture_model()`），不再寄生。

收益是 dummy 路径和真实路径共享同一份代码，不会漂移。

### 显式图管理

V1 的图生命周期是隐式的：什么形状被捕了、运行时怎么匹配，要在 runner 的多个方法里拼出来。V2 把它收进 `CudaGraphManager`：

- `BatchExecutionDescriptor` 显式描述「一种执行形态」（token 数、request 数、cg_mode、是否 uniform decode、活跃 LoRA 数、是否 ubatch）。
- `_init_candidates()` 枚举候选，`capture()` 逐个捕，`dispatch()` + `_is_compatible()` 在运行时匹配。
- 用标准 PyTorch API 捕获完整图。

一个直接受益：V2 能把**多个草稿前向**捕进一张图（设计文档第 9 节的 Fused Multi-Step Draft Decoding）。代价是后端要实现 `AttentionMetadataBuilder.update_draft_decode_metadata()`，且这个钩子只在**捕获期**执行 Python 体、replay 时不再执行——这条约束是整节最容易踩坑的地方。

### 和轴二的关系

DP 下所有 rank 必须选同一个描述符，否则集合通信错配。V2 把图选择和 DP padding 同步绑在 `dispatch_cg_and_sync_dp()` 一次做完，对比 V1 的 `_determine_batch_execution_and_padding()` 看这件事是怎么组织的。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 的 CUDA graph 管理与 dummy_run 职责划分。
V1 = vllm/v1/worker/gpu_model_runner.py（_dummy_run、capture_model、
     _warmup_and_capture、_capture_cudagraphs、_check_and_update_cudagraph_mode、
     _determine_batch_execution_and_padding、profile_run、profile_cudagraph_memory）
V2 = vllm/v1/worker/gpu/cudagraph_utils.py + gpu/model_runner.py + gpu/warmup.py

=== 第一部分：dummy_run 的职责 ===
1. 读 V1 的 _dummy_run()：它有多少行、多少个参数？
   把每个参数列出来，标注它是为哪一类调用方服务的
   （profiling / compile / 捕获 / warmup / DP 空转 / 其他）。
2. 找出 V1 里所有 _dummy_run() 的调用点，按上面五类归类，
   说明每类调用走的是函数里的哪些分支。
3. 对照 docs/design/model_runner_v2.md 第 8 节列的四类职责，
   逐条在 V1 代码里定位，确认文档说的和代码一致。
4. 讲 V2 的拆法：
   - execute_model() 的 dummy_run / skip_attn_for_dummy_run / is_profile /
     randomize_inputs / valid_dummy_state_slots 各自改变了哪些分支？
   - dummy_run=True 时，明确列出哪些持久状态不会被修改，以及代码靠什么保证这一点。
   - _dummy_run() / profile_run() / _dummy_sampler_run() / _dummy_pooler_run()
     各自怎么委托给 execute_model。
   - capture_model() 为什么要独立，不走 execute_model？
5. 回答：V1 的"dummy 路径和真实路径会漂移"这个风险，具体会以什么形式暴露成 bug？
   在 V1 代码里找一个两条路径处理方式不同的地方作为例子
   （比如某个状态在 dummy 下被跳过、在真实路径下被更新）。

=== 第二部分：形态决策 ===
6. 对比 V1 的 _determine_batch_execution_and_padding() 和
   V2 的 BatchExecutionDescriptor + dispatch_cg_and_sync_dp()：
   - 各自的输入是什么、输出是什么？
   - V1 把决策结果存在哪（runner 属性？局部变量？）V2 存在哪？
   - V2 把"决策结果"显式化成一个对象，带来了什么（可比较、可枚举、可作为图的 key）？
7. 讲 V2 的 _is_compatible() 和 AttentionState：运行时匹配的规则是什么？
   V1 的对应判断散在哪几个方法里（_is_uniform_decode / _allow_microbatching / ...）？
8. 讲 DP 的约束：为什么所有 rank 必须选同一个描述符？
   V1 在哪里做这个协商？V2 的 sync_cudagraph_and_dp_padding() 又是怎么做的？
   对比两者的通信时机和内容。

=== 第三部分：捕获与执行 ===
9. 对比捕获流程：
   V1 的 capture_model → _warmup_and_capture → _capture_cudagraphs → _dummy_run
   ↔ V2 的 capture_model → CudaGraphManager.capture / ModelCudaGraphManager.capture
      → prepare_inputs_to_capture。
   说明 V2 的 ModelCudaGraphManager 覆盖了父类的什么。
10. 对比运行时执行：V1 怎么决定跑 eager / piecewise / full？
    V2 的三分支（ubatch_runner.run / run_pw_graph / 直接 self.model(**inputs)）
    在 execute_model 的哪一段？
11. 对比 LoRA 进入图 key 的方式：
    V1 怎么处理不同 LoRA 数量的图？V2 的 _build_lora_dispatch_map /
    _resolve_effective_loras / get_lora_capture_cases 又是怎么做的？
12. 对比 warmup：V1 的 _warmup_and_capture ↔ V2 的 gpu/warmup.py
    （run_mixed_prefill_decode_warmup / warmup_kernels）。
    V2 为什么要专门构造混合 prefill/decode 批？

=== 第四部分：V2 独有的融合草稿解码 ===
13. 对照设计文档第 9 节，在代码里定位融合多步草稿解码的实现。
    说明 V1 是怎么跑多步草稿的（每步 replay 一张图？重建元数据？），
    V2 的融合路径省掉了什么。
14. 讲 AttentionMetadataBuilder.update_draft_decode_metadata() 这个钩子：
    - 捕获期执行、replay 期不执行 Python 体，这意味着什么约束？
    - 举一个会出错的错误实现例子。
    - supports_draft_decode_metadata_update 在哪里被检查、不满足时回退到什么。
    - 仓库里哪些后端已经实现了它？
15. 回答：这个能力为什么在 V1 的隐式图管理下很难做？

=== 第五部分：收敛 ===
16. 一张表总结：V1 的图管理代码分散在哪几个方法里（给行号），
    V2 收进了哪几个类，分散 → 收敛带来的具体好处各是什么。
17. 回答：显式图管理的代价是什么？（新增一种执行形态时两边各要改几处？）

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 凡是涉及"捕获期执行 vs replay 期执行"的区别，每次都要明确写出来。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
