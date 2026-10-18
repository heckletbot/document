# vLLM Model Runner V2 学习计划

目标：读懂 vLLM 的 **Model Runner V2（下称 MRV2）**——它的设计动机、当前能力边界、以及代码怎么跑起来。

路线：**设计转变（1 章）→ 能力边界（2 章）→ 代码细节（10 章）**。

每一章都带一段可以直接粘给 AI 的 prompt，让 AI 对着源码给你讲，你只负责提问和验收。

---

## 先把两份代码的位置钉死

MRV2 不是一个 `_v2.py` 文件，而是一整个目录。对照关系：

| | V1（MRV1） | V2（MRV2） |
|--|-----------|-----------|
| 主文件 | `vllm/v1/worker/gpu_model_runner.py`（单文件约 7200 行） | `vllm/v1/worker/gpu/` 整个包（约 70 个文件、26000 行） |
| 入口类 | `GPUModelRunner` | `vllm/v1/worker/gpu/model_runner.py` 里的 `GPUModelRunner`（约 2500 行） |
| 批状态 | `vllm/v1/worker/gpu_input_batch.py` 的 `InputBatch` + `CachedRequestState` | `gpu/states.py` 的 `RequestState` + `gpu/input_batch.py` 的 `InputBatch` |
| 官方设计文档 | 无 | `docs/design/model_runner_v2.md` |

注意命名陷阱：**「V1 / V2」指的是 model runner 的版本，不是 engine 的 V0/V1**。两者都跑在 engine V1 之上，目录都在 `vllm/v1/` 底下。`vllm/v1/worker/gpu_model_runner.py`（带下划线、平铺在 `worker/` 下）是 V1；`vllm/v1/worker/gpu/model_runner.py`（在 `gpu/` 子包里）是 V2。

选哪个由 `VllmConfig.use_v2_model_runner`（`vllm/config/vllm.py`）决定，worker 侧在 `vllm/v1/worker/gpu_worker.py` 的 `_make_model_runner()` 里实例化。

---

## 章节大纲

### 第一部分 · 设计思路的转变

| 章 | 主题 | 文档 |
|----|------|------|
| 01 | V1 的债务与 MRV2 的九条重构主张 | [prompts/01-design-shift.md](prompts/01-design-shift.md) |

### 第二部分 · 能力边界

| 章 | 主题 | 文档 |
|----|------|------|
| 02 | MRV2 当前支持的特性全景与启用开关 | [prompts/02-v2-features.md](prompts/02-v2-features.md) |
| 03 | 双向特性缺口：V2 待补 / V1 已不支持 | [prompts/03-feature-gaps.md](prompts/03-feature-gaps.md) |

### 第三部分 · 代码细节

| 章 | 主题 | 文档 |
|----|------|------|
| 04 | 入口与一个 step 的生命周期 | [prompts/04-entry-and-step.md](prompts/04-entry-and-step.md) |
| 05 | 持久化状态与无同步写入（`states` / `block_table` / `buffer_utils`） | [prompts/05-persistent-state.md](prompts/05-persistent-state.md) |
| 06 | `InputBatch` 与 GPU 原生输入准备 | [prompts/06-input-batch.md](prompts/06-input-batch.md) |
| 07 | Attention 元数据、KV cache 与 `ModelState` 抽象 | [prompts/07-attention-and-kv.md](prompts/07-attention-and-kv.md) |
| 08 | CUDA Graph 管理、warmup 与显存 profiling | [prompts/08-cudagraph.md](prompts/08-cudagraph.md) |
| 09 | 采样栈：Triton sampler / logprobs / logits processors | [prompts/09-sampling.md](prompts/09-sampling.md) |
| 10 | 投机解码与拒绝采样 | [prompts/10-spec-decode.md](prompts/10-spec-decode.md) |
| 11 | 异步输出与跨 step 流水 | [prompts/11-async-output.md](prompts/11-async-output.md) |
| 12 | 并行维度：DP / PP / PCP / micro-batch / EPLB | [prompts/12-parallelism.md](prompts/12-parallelism.md) |
| 13 | 外围集成：KV connector / 结构化输出 / LoRA / 多模态 / pooling | [prompts/13-integrations.md](prompts/13-integrations.md) |

---

## 怎么用这些 prompt

1. 本地准备一份 vLLM 源码，建议浅克隆：

```bash
git clone --depth 50 --filter=blob:none https://github.com/vllm-project/vllm.git
```

2. 打开每章文档，先自己读「要掌握的知识」，再把「Prompt」整段粘给 AI（Cursor / Claude Code 之类能读本地代码的 agent 最好）。
3. 每个 prompt 里的 `<VLLM_REPO>` 替换成你的本地路径。
4. AI 讲完后，用章节底部的「自测题」口头验收。答不上来的点，继续追问同一个 AI 会话，不要急着进下一章。

### 给 prompt 的通用约束

下面这段建议追加在每个 prompt 末尾（各章 prompt 里已经内置了，这里单列出来方便你改）：

```text
约束：
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 的历史版本行为。
- 每个结论都要给出 `文件路径:行号` 和函数名；拿不准就直接说"代码里没找到"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- 遇到 V1/V2 都有的同名实体（比如两个 GPUModelRunner），每次都写清楚是哪个路径。
```

### 版本说明

本计划的文件路径、函数名、特性清单对照的是 vLLM 主干某次提交（`vllm/v1/worker/gpu/` 包已存在，`docs/design/model_runner_v2.md` 已存在）。MRV2 仍在高速迭代，**文件会增删、函数会改名**。如果某个路径对不上，先用下面的方式重新定位，再继续：

```bash
# 确认 MRV2 包还在不在
ls vllm/v1/worker/gpu/
# 确认开关逻辑还在不在
rg -n "use_v2_model_runner|_get_v2_model_runner_unsupported_features" vllm/config/vllm.py
# 列出当前 MRV2 各文件规模，作为阅读优先级
find vllm/v1/worker/gpu -name '*.py' | xargs wc -l | sort -rn
```

---

## 阅读顺序建议

- **只想知道「为什么要重写」**：读 01、02、03 就够。
- **要改 MRV2 代码 / 迁移一个特性过去**：01 → 03 → 04 → 05 → 06，再按你要碰的子系统跳到 07–13 对应章。
- **要排查一个线上问题**：04（看一个 step 怎么走）→ 11（异步输出）→ 08（CUDA graph）通常覆盖大部分现场。

05 和 06 是全书的地基：MRV2 的所有"反直觉"写法（不 pin 的 CPU 状态、UVA、StagedWriteTensor、`idx_mapping` 间接寻址）都在这两章解释。跳过它们直接看 `model_runner.py` 会非常痛苦。
