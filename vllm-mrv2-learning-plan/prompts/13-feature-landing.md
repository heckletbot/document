# 13 · 一个特性在两边各要动多少地方

> 轴三 · 采样器与模块化抽象

**对照的那件事**：前面十二章讲的都是「结构差异」。这一章用**具体特性**把三条轴串起来，得到一份可操作的结论：往 V2 加东西和往 V1 加东西，分别是什么体验。

这是全套文档的收口，也是你真要动手改 MRV2 时唯一需要带在手边的一章。

---

## 怎么挑案例

三个案例，各打一条轴：

| 案例 | 打中的轴 | 为什么选它 |
|------|---------|-----------|
| **LoRA** | 轴一 + 轴二 | 两边都支持，且都用同一个 `LoRAModelRunnerMixin`，差异纯粹来自持久状态和图 key 的组织方式 |
| **结构化输出（grammar bitmask）** | 轴二 | 它卡在 `execute_model` 和 `sample_tokens` 之间，最能体现 step 拆分的动机 |
| **投机解码的一个新方法** | 轴三 | V1 要写 proposer + 改巨型方法，V2 要实现 `BaseSpeculator` + 加一个分派分支 |

再加一个反向案例：

| 案例 | 为什么选它 |
|------|-----------|
| **PCP（prefill context parallel）** | 它**只在 V2 存在**（第 03 章）。问「为什么没人往 V1 加」，比问「V2 怎么实现的」更有信息量 |

---

## 要得到的结论形态

最终产出应该是一张**接入检查清单**，形如：

```text
往 MRV2 加一个特性，依次回答：
1. 要不要每请求持久状态？→ 加在 RequestState 还是子系统自己的状态类？
                           → 选 StagedWriteTensor / UvaBackedTensor / 裸 numpy / 裸 GPU？
2. 它的 CPU 侧写入会不会引入隐式同步？→ 主路径上有没有 .item() / .tolist() / 非 pinned .to()
3. 它的参数要不要对齐到 logits？→ 用 idx_mapping 间接寻址，不要展开
4. 它会不会改变批形状？→ 要不要进 BatchExecutionDescriptor（图 key）
5. 它要不要在捕获期和 replay 期有不同行为？→ 有没有 dummy 批上的缺席约定
6. 它属于哪个模型族/投机方法/连接器维度？→ 实现对应接口，不要在主循环插 if
7. 它有没有和别的特性互斥？→ 写进 _get_v2_model_runner_unsupported_features 还是
                              单独的 _validate_xxx
```

这张清单的每一条都应该能追溯到前面某一章的结论。让 AI 把追溯关系写出来，别只给清单。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：用具体特性把 V1/V2 的结构差异落到"改代码要动哪里"上，最后产出一份接入检查清单。
V1 = vllm/v1/worker/gpu_model_runner.py 及其相关模块
V2 = vllm/v1/worker/gpu/ 这个包

=== 案例一：LoRA（两边都支持）===
1. 两边都继承 vllm/v1/worker/lora_model_runner_mixin.py 的 LoRAModelRunnerMixin。
   列出 V1 用到了它的哪些方法、V2 用到了哪些，差集说明什么。
2. 对比每请求 LoRA 归属的存储：
   V1 存在哪（InputBatch 的哪些字段 / CachedRequestState 的哪个字段）？
   V2 的 gpu/lora_utils.py LoraState 存在哪？
   请求在批里移动时，两边各要做什么？（这条直接打轴一）
3. 对比 LoRA 进入 CUDA graph key 的方式：
   V1 怎么处理不同活跃 LoRA 数量的图？
   V2 的 get_num_active_loras_for_dispatch → BatchExecutionDescriptor →
   _build_lora_dispatch_map / _resolve_effective_loras / get_lora_capture_cases
   这条链是怎么走的？（这条打轴二）
4. 结论：同样一个 LoRA 特性，两边的实现复杂度差在哪？差的部分可以归因到哪一章？

=== 案例二：结构化输出 ===
5. 对比两边 grammar bitmask 的注入位置：
   都是 sample_tokens(grammar_output) 的入参吗？在 V1 里找到对应位置。
6. 对比 bitmask 的应用实现：
   V1 用什么算子？V2 的 gpu/structured_outputs.py 的
   _apply_grammar_bitmask_kernel 是 Triton 的吗？两者的开销差异。
7. 讲 V2 的 _build_grammar_mapping()：它怎么用 idx_mapping？
   V1 对应的映射怎么建？（这条打轴三）
8. 讲 grammar_invalid_drafts()：投机草稿里不合语法的 token 怎么处理？
   V1 有等价逻辑吗？
9. 结论：step 被拆成两次调用这件事，对结构化输出意味着什么？
   如果不拆，这个特性要怎么实现？

=== 案例三：加一个新的投机方法 ===
10. 假设要加一个新的投机方法（你可以挑仓库里已有的一个作为样板，
    比如 ngram 或 mtp，把它当成"刚被加进来"来讲）。
    分别列出在 V1 和 V2 要动的文件和函数清单：
    - V1：要在 vllm/v1/spec_decode/ 下加什么？要改 propose_draft_token_ids() 的哪几段？
          要改 gpu_model_runner.py 的哪些地方（状态、拷贝路径、cudagraph、配置校验）？
    - V2：要在 gpu/spec_decode/ 下加什么？要改 init_speculator() 的哪里？
          BaseSpeculator 的三个方法各要实现什么？
          要不要碰 model_runner.py？要不要碰 cudagraph_utils.py？
11. 对比两边的"改动点数量"和"是否需要触碰主循环"。
12. 结论：V2 的分派表 + 契约，相比 V1 的巨型方法，在加新方法这件事上省掉了什么、
    又要求实现者额外想清楚什么（提示：融合草稿解码的
    supports_draft_decode_metadata_update）。

=== 案例四：PCP —— 一个只存在于 V2 的特性 ===
13. 确认 PCP 在 V1 不被支持（见 vllm/config/vllm.py 的
    _get_v1_model_runner_unsupported_features），引用代码注释。
14. 读 gpu/pcp_manager.py，说明 PCP 需要 model runner 提供哪些能力：
    - 要能按 rank 重切批（partition_batch / _get_rank_segments）；
    - 要能为局部切片重建注意力元数据；
    - 要能让某些请求在所有 rank 上复制（replicated_requests）。
15. 关键问题：这些能力在 V1 的架构下为什么难以提供？
    请逐条对应到前面各章的结论
    （提示：持久张量即模型输入 → 批顺序被绑死；condense → 重切批后行号怎么办；
     属性耦合 → 局部批状态存哪）。
    如果你认为其中某条在 V1 其实可行，明说并给理由。
16. 结论：PCP 这个案例说明 V2 的重构带来的不只是性能和可维护性，还有什么？

=== 最终产出：接入检查清单 ===
17. 综合上面四个案例，产出一份"往 MRV2 接入一个新特性"的检查清单。
    要求：
    - 每一条都是一个可回答的问题，不是口号；
    - 每一条都标注它追溯到本套文档的哪一章结论
      （04 批状态 / 05 状态写入 / 06 block table / 07 输入准备 /
       08 异步竞态 / 09 同步点 / 10 cudagraph / 11 采样器 / 12 模块化）；
    - 每一条都给一个"做错了会怎样"的具体后果。
18. 再产出一份对照的"如果是往 MRV1 接入"清单，用来凸显差异。
    对于 V1 已废弃这件事（见 vllm/config/vllm.py 里的注释），
    说明它对"还要不要往 V1 加东西"这个问题的答案是什么。
19. 最后用 200 字以内回答：如果让你向一个没读过这套文档的人解释
    "MRV2 到底改了什么"，你会说哪三句话？

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- "要动哪些地方"这类清单必须基于实际读过的代码，不要凭接口名推测。
  推测的部分请明确标注"这是推测"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
