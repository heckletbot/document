# 03 · 双向特性缺口：V2 待补 / V1 已不支持

目标：拿到一份**代码级可信**的迁移清单，知道哪些特性还卡在 V1、为什么卡住、迁移时要动哪些模块。

---

## 要看的内容

1. `vllm/config/vllm.py` · `_get_v2_model_runner_unsupported_features()`——V2 还缺什么
2. `vllm/config/vllm.py` · `_get_v1_model_runner_unsupported_features()`——V1 已经跟不上了
3. 两者的代码注释（这里的注释信息量极大，**注释本身就是迁移说明**）
4. `_get_dbo_unsupported_features()`（被 V2 清单在 `use_ubatching` 时复用）
5. `docs/design/model_runner_v2.md` 的 "Development Philosophy" 一节

---

## 要掌握的知识

### 方向一：MRV2 还不支持的（命中就回退 V1）

| 缺口 | 触发条件 | 代码注释给出的原因 |
|------|---------|-------------------|
| PP + `external_launcher` | `distributed_executor_backend == "external_launcher"` 且 `pipeline_parallel_size > 1` | V2 没实现 V1 用来保持各 rank 同步的 PP 输出广播（`broadcast_pp_output`） |
| 投机方法 `suffix` / `custom_class` | `speculative_config.method` 命中 | V2 的 `init_speculator()` 没有对应分派 |
| EAGLE 的并行草稿 | `parallel_drafting` 为真且 method 不是 `dflash`/`dspark` | V2 的 `EagleSpeculator` 不支持 parallel drafting；DFlash / DSpark 在 V2 里用自己的 speculator 原生做并行草稿 |
| 异构词表 draft 模型 | `use_heterogeneous_vocab` | V2 的 draft-model speculator 没有 draft↔target 的词表映射（TLI 待定） |
| DBO（micro-batch）相关若干项 | `parallel_config.use_ubatching` 为真时，把 `_get_dbo_unsupported_features()` 整个并进来 | 见该函数本身 |

**读法提醒**：这张表会变，而且变得很快。别背，要记住的是「清单在哪个函数里」。每次升级 vLLM 重跑一次：

```bash
rg -n "_get_v2_model_runner_unsupported_features" -A 40 vllm/config/vllm.py
```

### 方向二：V1 已经不支持的（说明重心已经转移）

| 特性 | 代码注释给出的原因 |
|------|-------------------|
| Prefill context parallel（PCP） | PCP 的运行时只在 V2 实现（`gpu/pcp_manager.py`） |
| PP + 异步调度 | V1 + PP>1 + async sched + 结构化输出 有已知 bug（issue #45014）；因为 V1 已废弃，社区决定不修而是直接禁用这个组合，V2 本来就正常 |
| DSpark 投机 | 只在 V2 的 GPU model runner 实现 |
| 自适应草稿验证（adaptive verification） | 只在 V2（`gpu/spec_decode/adaptive_verification.py`） |
| draft lm_head 量化 | 同上 |
| 混合 sliding/full 的 DFlash draft | 需要多个 KV group，V2-only |
| DFlash candidate-head draft | candidate head 只存在于 V2 的 speculator；在 V1 上同一份权重会走 `DFlashProposer`，**静默退化成 DFlash1**——这是最危险的一类缺口：不报错，只是效果变差 |
| diffusion 模型 | V1 不支持 |
| batch-sharded sampling | 只在 V2（`gpu/sample/batch_shard.py`） |

注释里还有一句关键信息：**MRV1 已被标记为 deprecated**。所以"待迁移"的真实含义是「这些特性在 V2 补齐之前，用户会被自动降级到一个已废弃的 runner」，而不是「V2 是可选的实验分支」。

### 第三类缺口：不写在清单里的

上面两个函数只覆盖「会影响 runner 选型」的特性。还有一类缺口藏在别处，排查时别漏：

- `_validate_adaptive_verification()` 这类**特性内部的互斥校验**（比如自适应验证与 LoRA、与 PP>1 不兼容，且要求 full CUDA graph）。
- `_validate_batch_sharded_sampling()` 里的 blockers（TP=1、`max_num_seqs < tp_size`、`max_logprobs == -1`、`return_sampling_mask` 等）。
- 散落在 MRV2 各模块里的 `NotImplementedError` / `assert` / `TODO`。

### 迁移一个特性过去，要想清楚什么

设计文档的 "Development Philosophy" 定了调子：**不要照搬 V1 行为，要在 MRV2 的设计语境下重新想一遍**。落到具体，至少要回答：

1. 这个特性要不要新增 per-request 持久状态？如果要，加在 `states.py` 的 `RequestState` 还是子系统自己的状态类里？
2. 它的 CPU 侧写入会不会破坏 async-first（有没有引入隐式同步）？
3. 它要不要参与 CUDA graph 捕获？形状是否随 batch 变化？
4. 投机解码下它的 per-request 参数怎么通过 `idx_mapping` 映射到 per-logit？

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。

本轮任务：给我一份代码级可信的「MRV1 ↔ MRV2 特性缺口」清单和迁移分析。

第一部分 · 两份清单
1. 精读 vllm/config/vllm.py 的 _get_v2_model_runner_unsupported_features()，
   输出表格：缺口 | 触发条件（配置字段） | 代码注释给出的原因 | 文件:行号。
   注释里提到的 issue / PR 编号要原样保留。
2. 同样处理 _get_v1_model_runner_unsupported_features()。
3. 说明这两个函数各自被谁调用、调用结果是"自动回退"还是"抛异常"。
4. 展开 _get_dbo_unsupported_features()，说明它为什么会被 V2 的清单在 use_ubatching 时复用。

第二部分 · 逐项深挖（挑下面这几项做，每项 10 行以内）
5. "PP + external_launcher"：在 V1 里找到 broadcast_pp_output 的实现，说明它解决什么问题，
   以及 V2 的 pp_utils.py 为什么没有等价物。
6. "EAGLE 并行草稿"：在 vllm/v1/worker/gpu/spec_decode/eagle/ 里确认 V2 的 EagleSpeculator
   确实没有 parallel drafting 路径；再说明 dflash / dspark 的 speculator 是怎么原生支持的。
7. "DFlash candidate-head draft"：解释为什么这一项在 V1 上是"静默退化"而不是报错，
   给出 V1 走的 DFlashProposer 路径和 V2 的 candidate head 路径的分叉点。
8. "PP + 异步调度"：找到注释里提到的 issue 编号，说明社区为什么选择禁用而不是修复。

第三部分 · 清单之外的缺口
9. 在 vllm/v1/worker/gpu/ 全目录下搜 NotImplementedError、"not supported"、"TODO"、
   "FIXME"，筛掉噪声后列出真正代表功能缺口的那些，给 文件:行号 和一句话说明。
10. 读 vllm/config/vllm.py 里的 _validate_adaptive_verification() 和
    _validate_batch_sharded_sampling()，列出这两个 V2-only 特性各自的互斥约束。

第四部分 · 迁移指引
11. 假设我要把其中一个 V2 缺口（你从第 1 部分里挑一个工作量适中的）迁移过去，
    给我一份改动清单：需要新增/修改 vllm/v1/worker/gpu/ 下的哪些文件、
    需要在 RequestState 里加什么持久状态、需要注意 async-first 的哪条约束、
    要不要改 cudagraph 捕获路径、现有测试在哪（tests/v1/ 下搜 mrv2 / gpu_model_runner_v2）。
12. 结合 docs/design/model_runner_v2.md 的 "Development Philosophy"，
    指出哪些 V1 的实现方式是"不应该照搬"的反面教材。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 特性清单必须完整覆盖那两个函数的每一个分支，不要省略。
- 代码注释里的因果解释要原样转述，不要改写成你的猜测。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码。
```

---

这一章得到的缺口清单是「还没搬过去的」。第 13 章会反过来问：已经搬过去的那些特性，在 V1 和 V2 里各自要动多少地方。
