# 10 · 投机解码与拒绝采样

目标：读懂 MRV2 的 speculator 抽象、各家草稿方法怎么插进来、拒绝采样怎么做，以及 V2 独有的自适应验证。这是 `gpu/` 下文件最多的子包（20+ 文件），**不要试图一次读完**，先把骨架和一条具体路径打通。

---

## 要看的内容

**第一轮（骨架）：**
1. `vllm/v1/worker/gpu/spec_decode/__init__.py` · `init_speculator()`——分派表，先读
2. `vllm/v1/worker/gpu/spec_decode/speculator.py` · `BaseSpeculator` / `DraftModelSpeculator`
3. `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` · `RejectionSampler`
4. `vllm/v1/worker/gpu/model_runner.py` 里 speculator 的调用点 + `take_draft_token_ids()`

**第二轮（挑一条具体路径走通，建议 ngram 或 eagle）：**
5. `spec_decode/ngram/speculator.py`（最简单，不需要 draft 模型）或 `spec_decode/eagle/`

**第三轮（V2-only 能力）：**
6. `spec_decode/adaptive_verification.py` + `spec_decode/acceptance_estimator.py`
7. `spec_decode/rejection_sampler_utils.py`（1300+ 行，按需查）

**其余按需：** `mtp/`、`multi_module_mtp/`、`gemma4/`、`dflash/`、`dflash2/`、`dspark/`、`lilicorr/`、`standalone_ar/`、`target_dependent_ar/`、`extract_hidden_states.py`

---

## 要掌握的知识

### 分派表就是能力清单

`init_speculator()` 的分支顺序（短路）：

```text
method == "extract_hidden_states"  → ExtractHiddenStatesSpeculator
method == "dflash"
    ├─ draft 架构含 LiLiCorrDraftModel  → LiLiCorrSpeculator
    ├─ draft 架构含 DFlash2DraftModel   → DFlash2Speculator
    └─ 否则                              → DFlashSpeculator
method == "dspark"                 → DSparkSpeculator
use_gemma4_mtp()                   → Gemma4Speculator
use_multi_module_mtp()             → MultiModuleMTPSpeculator
method == "mtp"                    → MTPSpeculator
use_eagle()                        → EagleSpeculator
uses_draft_model()                 → StandaloneARSpeculator
use_ngram()                        → NgramGPUSpeculator   ← 唯一额外收 req_states
其余                                → NotImplementedError
```

注意 **只有 ngram 要传 `req_states`**：它要在 GPU 上反复扫活跃请求的历史 token 做匹配。这也解释了第 05 章 `all_token_ids` 的 `use_dense_all_token_ids` 开关为什么是为 ngram 开的——UVA 上反复扫太慢。

第 03 章提到的缺口在这里得到印证：`suffix` 和 `custom_class` 在这张表里根本没有分支。

### `BaseSpeculator` 的契约

很小：`init_cudagraph_manager()` / `capture()` / `propose()`。`propose()` 是核心——给定本步的 hidden states 和 input batch，产出草稿 token。

`DraftModelSpeculator` 是「有独立草稿模型」那一类的公共基类，内容多得多：`load_draft_model` / `load_model` / `set_attn` / `_build_attn_metadata` / `compute_draft_logits` / `get_draft_top_tokens` / `sample_draft` / `set_eplb_state` / `_prepare_eplb_forward` / `observe_verification` / `_maybe_predict_acceptance`。

两个值得停下来的点：
- `_configure_decode_metadata_update()`：这正是第 08 章「融合多步草稿解码」的开关检查点，决定能不能把所有草稿步捕进一张图。
- `_validate_local_argmax_reduction()` 和 `draft_logits_spec()`：草稿 logits 的 dtype / 缩放约定，涉及 TP 下的局部 argmax 归约正确性。

### 拒绝采样

`RejectionSampler.__call__` 的主要分支是 `_verify()` 和 `_verify_in_chunks()`——后者按 `get_max_chunk_logits(vocab_size)` 切块，避免一次物化过大的 logits。`gather_draft_sampled()` / `_gather_draft_sampled_kernel` / `_flatten_sampled_kernel` 负责把草稿和目标模型的采样结果对齐。`_watermarking_kwargs()` 说明水印要穿透到拒绝采样里。

### 自适应验证（V2-only）

第 06 章已经碰到了它的两段式：

- `compact_batch()`（CPU 侧）：在 `prepare_inputs` 早期压缩批，此时 `num_scheduled_tokens` 还只是**均匀分配的草稿预算上界**。
- `reallocate_drafts()`（GPU 侧）：真正按每请求的接受率预测**不均匀**地分配草稿预算，重算 `cu_num_logits` / `query_start_loc` / `total_num_draft_tokens`。

配套的 `acceptance_estimator.py` 维护接受率估计，`observe_verification()` 回填观测值。

它的约束（第 03 章提过）：不兼容 LoRA（per-token LoRA 映射从 CPU 占位边界构建，而真实边界在 GPU 上才定）、要求 full CUDA graph、不兼容 PP>1（代价曲线和置信度只在最后一级 PP rank 上）。**这三条约束各自的原因都是很好的理解检验题**。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲透 MRV2 的投机解码子系统。分三轮讲，我说"继续"你再进下一轮。

=== 第一轮：骨架 ===
1. 读 vllm/v1/worker/gpu/spec_decode/__init__.py，把 init_speculator() 的分派顺序
   画成决策树，每个分支标注：判定条件来自哪个 speculative_config 字段 / 哪个
   draft 架构名、选中的类在哪个文件。
2. 说明为什么只有 NgramGPUSpeculator 额外接收 req_states，
   并把它和第 05 章 RequestState.all_token_ids 的 use_dense_all_token_ids 开关联系起来。
3. 讲 BaseSpeculator（spec_decode/speculator.py）的抽象契约：
   init_cudagraph_manager / capture / propose 各自的职责和调用时机
   （在 model_runner.py 里指出调用点）。
4. 讲 DraftModelSpeculator 的完整方法清单，按"模型加载 / 注意力 / logits / 采样 /
   EPLB / 接受率观测"分组，每个方法一句话。
5. 讲 model_runner.py 里草稿 token 的产出与取用：
   sample_tokens 里 speculator 在哪一步被调用、draft_tokens 写回到 RequestState 的哪一列、
   take_draft_token_ids() 为什么是独立一次调用。

=== 第二轮：一条具体路径（选 ngram 和 eagle 各讲一遍）===
6. 讲 spec_decode/ngram/speculator.py 的完整 propose 流程：
   它在 GPU 上怎么做 n-gram 匹配、扫的是哪块数据、为什么不需要草稿模型。
7. 讲 spec_decode/eagle/ 下的 speculator、utils.py、eagle3_utils.py：
   EAGLE 的 hidden states 是怎么从目标模型拿到的（和 aux_hidden_states 的关系）、
   多步草稿怎么循环、EAGLE3 多了什么。
8. 讲 spec_decode/extract_hidden_states.py 的用途（它不是真的投机，对吧？说明它干什么）。
9. 讲 _configure_decode_metadata_update() 和 supports_draft_decode_metadata_update：
   它和第 08 章的融合多步草稿解码是什么关系，哪些 speculator 走融合路径。

=== 第三轮：拒绝采样与自适应验证 ===
10. 讲 RejectionSampler.__call__ 的完整流程，说明 _verify() 和 _verify_in_chunks()
    的分叉条件、get_max_chunk_logits() 怎么定块大小、分块解决的是什么问题。
11. 讲 gather_draft_sampled / _gather_draft_sampled_kernel / _flatten_sampled_kernel
    各自的输入输出形状和对齐逻辑。
12. 讲 _watermarking_kwargs()：水印为什么必须穿透到拒绝采样里？
13. 讲 adaptive_verification.py 的两段式：
    - compact_batch() 在 CPU 侧做什么？此时 num_scheduled_tokens 的语义是什么？
    - reallocate_drafts() 在 GPU 侧做什么？它重算了哪些张量？
    - 为什么这件事必须在 GPU 上做（CPU 缺什么信息）？
14. 讲 acceptance_estimator.py 的估计方法，以及 observe_verification() /
    _maybe_predict_acceptance() 的反馈闭环。
15. 回到 vllm/config/vllm.py 的 _validate_adaptive_verification()，
    逐条解释它的三个约束（LoRA / full cudagraph / PP>1）各自的原因，照注释讲。
16. 最后给一张表：spec_decode/ 下每个子目录 → 对应的投机方法 → 需不需要独立草稿模型 →
    是不是 V2-only（依据第 03 章的两份缺口清单）。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 投机方法的名字一律以 init_speculator() 的分派为准，不要用社区俗称。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```

---

## 自测题

1. `init_speculator()` 一共分派多少种 speculator？哪两种在第 03 章的缺口清单里根本没有分支？
2. 为什么只有 ngram 需要 `req_states`？这和 `all_token_ids` 的存储位置有什么关系？
3. `BaseSpeculator` 的契约只有三个方法，`DraftModelSpeculator` 为什么膨胀这么多？
4. 草稿 token 产出后存在哪里？`take_draft_token_ids()` 为什么要单独取？
5. `_configure_decode_metadata_update()` 决定了什么？和 CUDA graph 有什么关系？
6. `RejectionSampler` 什么时候走分块验证？分块解决的是显存还是算力问题？
7. 自适应验证的两段式分别在 CPU 和 GPU 上做什么？为什么不能都放 CPU？
8. 自适应验证为什么不兼容 LoRA？原因和「CPU 占位边界 vs GPU 真实边界」有什么关系？
9. 自适应验证为什么要求 full CUDA graph、为什么不兼容 PP>1？
10. 水印为什么要穿透到拒绝采样里？不穿透会出什么问题？
