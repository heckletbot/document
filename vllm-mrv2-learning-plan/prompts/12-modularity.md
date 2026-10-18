# 12 · 模型族、投机、连接器的差异放在哪

> 轴三 · 采样器与模块化抽象

**对照的那件事**：「不同模型 / 不同投机方法 / 不同连接器要走不同逻辑」这件事，在两边分别用什么手段表达。

对应设计文档第 7 条。核心问题不是「文件多少」，而是**可替换性**：加一个新模型族，V1 要在主循环里插几个 `if`，V2 要实现几个接口方法。

---

## 两边的锚点

| 差异维度 | V1 的表达方式 | V2 的表达方式 |
|---------|-------------|-------------|
| 整体 | `GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin)` 单文件约 7200 行 | `gpu/` 包约 70 文件，runner 本体约 2500 行 |
| 模型族 | 主循环里的内联分支 + 专用方法（`_get_mamba_bufs`、`_init_mrope_positions`、`_calc_mrope_positions`、`_get_encoder_seq_lens`、encoder-decoder 处理等） | `gpu/model_states/` · `ModelState` 抽象 + `resolve_model_state_cls()` 分派 |
| 投机 | `propose_draft_token_ids()`（巨型方法）+ `vllm/v1/spec_decode/` 下各 proposer | `gpu/spec_decode/` · `BaseSpeculator` + `init_speculator()` 分派 |
| KV 连接器 | `KVConnectorModelRunnerMixin`（继承） | `gpu/kv_connector.py` · `KVConnector`（空实现）/ `ActiveKVConnector`（活实现）+ `get_kv_connector()` |
| EC 连接器 | `ECConnectorModelRunnerMixin` | `gpu/ec_connector.py` 同样的空/活对子 |
| LoRA | `LoRAModelRunnerMixin`（V1/V2 共用） | 仍继承该 mixin + `gpu/lora_utils.py` 的 `LoraState` |
| 多模态 | runner 上的 `_execute_mm_encoder` / `_gather_mm_embeddings` / `_batch_mm_inputs_from_scheduler` / `_cache_encoder_output` 等 | `gpu/mm/` · `EncoderRunner` + `EncoderCache` + `RopeState` |
| pooling | runner 上的 `_pool` / `_dummy_pooler_run*` | `gpu/pool/pooling_runner.py` · `PoolingRunner` |
| 图管理 | runner 上的多个方法 | `gpu/cudagraph_utils.py` · `CudaGraphManager` |
| 并行 | runner 上的 `_pp_broadcast_prev_sampled_token_ids` 等 + `gpu_ubatch_wrapper.py` | `gpu/pp_utils.py`、`dp_utils.py`、`pcp_manager.py`、`ubatch_utils.py`、`eplb_utils.py` |

---

## 差异的因果

### 继承 vs 组合

V1 用 **mixin 继承**接连接器和 LoRA。mixin 的问题是它把方法直接挂到 runner 的命名空间上，调用时看不出边界，也无法在运行时替换；没有连接器时靠 `if self.kv_connector is not None` 之类的判断散落各处。

V2 用 **空对象模式**：`KVConnector` 是空操作基类，`ActiveKVConnector` 是真实现，`get_kv_connector()` 二选一。主路径上直接调 `self.kv_connector.pre_forward(...)`，不需要判空。边界清晰，且替换成本是零。

### 内联分支 vs 接口分派

模型族是最能说明问题的维度。V1 把 Mamba、M-RoPE、encoder-only、encoder-decoder 的特殊逻辑**内联**在主循环和一堆 `_xxx` 私有方法里。加一个新模型族，要在主循环的多个点插判断。

V2 定义 `ModelState` 接口（`add_request` / `preprocess_state` / `prepare_attn` / `prepare_inputs_embeds` / `gather_mm_embeddings` / `custom_sampler` 等），`resolve_model_state_cls()` 按模型特征选一个实现类。加新模型族 = 新增一个子类。

这也让**顺序约束**变得可声明。例如 `MambaHybridModelState.preprocess_state()` 的注释写明它必须在 `prepare_attn` 收集 `num_accepted_tokens` 之前跑——在 V1 里这种约束只能靠代码位置隐式维持。

### 投机解码

V1：`propose_draft_token_ids()` 是个巨型方法，`vllm/v1/spec_decode/` 下的 proposer 各自有不同的接口形状。
V2：`BaseSpeculator` 契约只有三个方法（`init_cudagraph_manager` / `capture` / `propose`），`init_speculator()` 是一张清晰的分派表，`DraftModelSpeculator` 作为「有独立草稿模型」那类的公共基类。

第 03 章的缺口清单在这里有了解释：V2 不支持 `suffix` / `custom_class`，是因为分派表里没有对应分支——缺口是**显式**的，而不是「某个 if 没覆盖到」。

### 「属性耦合」这个指标

设计文档提到 V2「把模型输入收进 `InputBatch`，减少直接的 model runner 属性耦合」。这是一个可量化的对比：数一数 V1 的 `_build_attention_metadata()` 从 `self.` 上读了多少个属性，V2 的 `build_attn_metadata()` 从 `InputBatch` 这一个参数里取了多少个字段。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 表达"差异逻辑"的手段，评估可替换性。
V1 = vllm/v1/worker/gpu_model_runner.py + vllm/v1/spec_decode/ + 各 mixin
V2 = vllm/v1/worker/gpu/ 这个包

=== 第一部分：量化规模与耦合 ===
1. 给出两边的规模对比：V1 单文件行数 / 方法数；
   V2 的 runner 本体行数 + 包内文件数与总行数。
   再给一个更有意义的指标：V1 的 GPUModelRunner 有多少个实例属性（数 self.xxx = 的赋值），
   V2 的有多少个。
2. 量化"属性耦合"：挑三个同功能的函数对
   （比如 _build_attention_metadata ↔ build_attn_metadata、
    _prepare_inputs ↔ prepare_inputs、_sample ↔ sample），
   分别数出它们从 self.* 读了多少个属性、从参数里取了多少个字段。
   用这组数字说明设计文档第 7 节那句"reduces direct model-runner attribute coupling"
   到底指什么。
3. 列出 V1 GPUModelRunner 继承的三个 mixin，说明各自往 runner 上挂了哪些方法，
   以及 V2 对应地用了什么（继承 / 组合 / 空对象）。

=== 第二部分：逐个差异维度对照 ===
对下面每个维度，都回答同一组问题：
  (a) V1 把差异逻辑放在哪（给文件:行号）？
  (b) V2 放在哪？用的是什么抽象？
  (c) 加一个新的 variant（新模型族 / 新投机方法 / 新连接器），两边各要改几处、改哪里？
  (d) 这个抽象有没有把某种隐式约束变成显式的？

4. 模型族：V1 的 Mamba / M-RoPE / encoder-only / encoder-decoder 内联处理
   ↔ V2 的 ModelState + resolve_model_state_cls()。
   额外要求：
   - 把 resolve_model_state_cls() 的分派顺序画成决策树，说明顺序为什么不能换。
   - 列出 ModelState 的所有接口方法，标注每个方法在 V1 对应的是哪个 _xxx 方法
     或哪段内联代码；找不到对应的，说明是 V2 新增的抽象点。
   - 重点讲 MambaHybridModelState.preprocess_state() 的顺序约束注释，
     说明同样的约束在 V1 里靠什么维持。
   - 讲 supports_prompt_embeds 那道校验：它把什么隐患显式化了？
5. 投机解码：V1 的 propose_draft_token_ids() + vllm/v1/spec_decode/ 下各 proposer
   ↔ V2 的 BaseSpeculator + init_speculator()。
   额外要求：
   - 对比 V1 各 proposer 的接口形状是否统一，V2 的 BaseSpeculator 契约有多大。
   - 把 init_speculator() 的分派表和 vllm/v1/spec_decode/ 下的文件做一一对照，
     指出哪些方法两边都有、哪些只有一边（对应第 03 章的缺口清单）。
   - 回答：V2 不支持 suffix / custom_class，这个"不支持"在代码里是显式的还是隐式的？
     和 V1 那种"某个 if 没覆盖到"的缺口有什么本质区别？
6. KV / EC 连接器：V1 的 KVConnectorModelRunnerMixin / ECConnectorModelRunnerMixin
   ↔ V2 的空实现 + 活实现对子。
   额外要求：
   - 在 V1 主路径上数出所有 "if connector is not None" 一类的判空分支。
   - 讲 V2 的 no_forward() 和 reset_capture_state()：
     这两个钩子在 V1 对应什么？它们把什么边界情况变成了接口的一部分？
7. 多模态：V1 的 _execute_mm_encoder / _gather_mm_embeddings /
   _batch_mm_inputs_from_scheduler / _cache_encoder_output / _init_mrope_positions /
   _calc_mrope_positions ↔ V2 的 gpu/mm/（EncoderRunner / EncoderCache / RopeState）。
8. pooling：V1 的 _pool / _dummy_pooler_run* ↔ V2 的 PoolingRunner。
9. 并行：V1 散在 runner 上的 PP / DP / ubatch 处理
   ↔ V2 的 pp_utils / dp_utils / pcp_manager / ubatch_utils / eplb_utils。
   额外要求：指出 PCP 在 V1 里**不存在**（第 03 章），
   说明这件事本身是不是模块化的一个证据。

=== 第三部分：收敛 ===
10. 一张总表：差异维度 | V1 手段（继承 / 内联 if / 私有方法 / 独立模块）|
    V2 手段 | 新增一个 variant 的改动点数量（V1 vs V2）。
11. 回答：V2 的模块化有没有代价？具体列举
    （跨文件跳转、抽象泄漏、接口方法过多、间接层带来的性能损耗等），
    每条要在代码里找一个实例，而不是泛泛而谈。
12. 对照 docs/design/model_runner_v2.md 的 "Development Philosophy"，
    回答：它为什么强调"不要照搬 V1 行为，要在 MRV2 语境下重新设计"？
    结合你上面找到的例子说明照搬会破坏哪个抽象边界。

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- "改动点数量"这类结论必须基于实际数过的代码，不要估计；数不动就说明你只数了哪部分。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
