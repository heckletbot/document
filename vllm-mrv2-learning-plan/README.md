# vLLM Model Runner：V1 → V2 的差异

**这份计划的目的不是学 Model Runner V1，而是搞清楚 V2 相对 V1 变了什么、为什么变。**

所有章节都是**对照着读**：每章都给出 V1 的实现锚点和 V2 的实现锚点，问的问题永远是「同一件事，两边分别怎么做，差异的因果是什么」。

---

## 先把两份代码的位置钉死

| | V1（MRV1） | V2（MRV2） |
|--|-----------|-----------|
| 主文件 | `vllm/v1/worker/gpu_model_runner.py`（单文件约 7200 行） | `vllm/v1/worker/gpu/` 整个包（约 70 个文件、26000 行） |
| 入口类 | `GPUModelRunner` | `vllm/v1/worker/gpu/model_runner.py` 的 `GPUModelRunner`（约 2500 行） |
| 批状态 | `vllm/v1/worker/gpu_input_batch.py`：`InputBatch` + `CachedRequestState` | `gpu/states.py` 的 `RequestState` + `gpu/input_batch.py` 的 `InputBatch` |
| block table | `vllm/v1/worker/block_table.py`：`BlockTable` / `MultiGroupBlockTable` | `gpu/block_table.py` 的 `BlockTables` |
| 缓冲抽象 | `vllm/v1/utils.py` 的 `CpuGpuBuffer` | `gpu/buffer_utils.py` 的 `StagedWriteTensor` / `UvaBackedTensor` / `UvaBufferPool` |
| 采样器 | `vllm/v1/sample/`：`sampler.py`、`metadata.py`、`ops/` | `gpu/sample/` 整个子包 |
| 设计文档 | 无 | `docs/design/model_runner_v2.md` |

命名陷阱：**「V1 / V2」指的是 model runner 的版本，不是 engine 的 V0/V1**。两者都跑在 engine V1 之上，目录都在 `vllm/v1/` 底下。带下划线平铺在 `worker/` 下的是 V1，在 `gpu/` 子包里的是 V2，**两个类都叫 `GPUModelRunner`**。

选哪个由 `VllmConfig.use_v2_model_runner`（`vllm/config/vllm.py`）决定，worker 侧在 `vllm/v1/worker/gpu_worker.py` 的 `_make_model_runner()` 里实例化。

---

## 三条差异主轴

官方设计文档列了九条重构主张，但它们不是并列的。收敛成三条主轴之后，剩下的都是推论：

| 轴 | 一句话 | 推论出的设计文档条目 |
|----|--------|---------------------|
| **一 · 持久化批处理与输入准备** | 持久状态不再**就是**模型输入，而是「每请求固定一行 + 每步 gather」；输入构造从 Python 搬到 GPU | 第 1、4、5 条 |
| **二 · 异步调度** | 核心循环里不允许有 CPU 同步点；竞态靠「不共享内存」消除，而不是靠屏障 | 第 2、3、8、9 条 |
| **三 · 采样器与模块化抽象** | 采样重写为 Triton + 每请求行 + `idx_mapping` 间接寻址；巨型单文件拆成可替换的插槽 | 第 6、7 条 |

章节就按这三条轴组织。

---

## 章节大纲

### 第一部分 · 总览（3 章）

| 章 | 主题 | 文档 |
|----|------|------|
| 01 | 九条主张与 V1 的对应痛点 | [prompts/01-design-shift.md](prompts/01-design-shift.md) |
| 02 | V2 当前支持的特性全景与启用开关 | [prompts/02-v2-features.md](prompts/02-v2-features.md) |
| 03 | 双向特性缺口：V2 待补 / V1 已不支持 | [prompts/03-feature-gaps.md](prompts/03-feature-gaps.md) |

### 第二部分 · 轴一：持久化批处理与输入准备（4 章）

| 章 | 对照的那件事 | V1 锚点 | V2 锚点 | 文档 |
|----|-------------|---------|---------|------|
| 04 | 请求进出批时，状态怎么挪 | `CachedRequestState`、`condense()`、`swap_states()` | `RequestState` 固定行、`free_indices` | [prompts/04-batch-state.md](prompts/04-batch-state.md) |
| 05 | 状态怎么写、怎么拷到 GPU | `CpuGpuBuffer.copy_to_gpu()` 全量拷 | `StagedWriteTensor` 增量 + UVA + 轮转池 | [prompts/05-state-write.md](prompts/05-state-write.md) |
| 06 | block table 与注意力元数据 | `move_row` / `swap_row` / `commit_block_table` | staged write + `gather_block_tables` kernel | [prompts/06-block-table.md](prompts/06-block-table.md) |
| 07 | 每步的输入张量怎么造出来 | `_prepare_inputs()` 的 Python/numpy 路径 | `prepare_inputs()` + 一组 Triton kernel | [prompts/07-input-prep.md](prompts/07-input-prep.md) |

### 第三部分 · 轴二：异步调度（3 章）

| 章 | 对照的那件事 | V1 锚点 | V2 锚点 | 文档 |
|----|-------------|---------|---------|------|
| 08 | 异步下的竞态怎么处理 | `synchronize_input_prep()` 屏障 | 持久 buffer 不 pin + 临时副本 | [prompts/08-async-barrier.md](prompts/08-async-barrier.md) |
| 09 | 同步点在哪、结果怎么回传 | `_bookkeeping_sync()`、`_to_list()` | `AsyncOutput` 双流 + 延后 `get_output()` | [prompts/09-sync-points.md](prompts/09-sync-points.md) |
| 10 | CUDA graph 与 dummy_run | `_dummy_run()` 一肩挑、隐式图管理 | `CudaGraphManager` 显式、dummy 走主路径 | [prompts/10-cudagraph-dummy.md](prompts/10-cudagraph-dummy.md) |

### 第四部分 · 轴三：采样器与模块化抽象（3 章）

| 章 | 对照的那件事 | V1 锚点 | V2 锚点 | 文档 |
|----|-------------|---------|---------|------|
| 11 | 采样参数怎么组织、怎么采 | `SamplingMetadata` 展开成 per-logit | 每请求行 + `idx_mapping` + Gumbel | [prompts/11-sampler.md](prompts/11-sampler.md) |
| 12 | 模型族/投机/连接器差异放在哪 | 单文件里的 `if` 分支与 mixin | `ModelState` / `Speculator` / `Connector` 插槽 | [prompts/12-modularity.md](prompts/12-modularity.md) |
| 13 | 一个特性在两边各要动多少地方 | 案例对照 + 迁移检查清单 | 同左 | [prompts/13-feature-landing.md](prompts/13-feature-landing.md) |

---

## 怎么用这些 prompt

1. 本地准备一份 vLLM 源码，建议浅克隆：

```bash
git clone --depth 50 --filter=blob:none https://github.com/vllm-project/vllm.git
```

2. 打开每章文档，先读「两边的锚点」和「差异的因果」，再把「Prompt」整段粘给 AI（Cursor / Claude Code 之类能读本地代码的 agent 最好）。
3. 每个 prompt 里的 `<VLLM_REPO>` 替换成你的本地路径。

### 给 prompt 的通用约束

各章 prompt 末尾都内置了这段，单列出来方便你改：

```text
约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- V1 和 V2 都有名为 GPUModelRunner / InputBatch / Sampler / BlockTable 的实体，
  每次提到都要写清楚是哪个路径下的。
```

### 版本说明

路径、函数名、特性清单对照的是 vLLM 主干某次提交（`vllm/v1/worker/gpu/` 包与 `docs/design/model_runner_v2.md` 均已存在）。MRV2 仍在高速迭代，**文件会增删、函数会改名**。路径对不上时先重新定位：

```bash
ls vllm/v1/worker/gpu/
rg -n "use_v2_model_runner|_get_v2_model_runner_unsupported_features" vllm/config/vllm.py
find vllm/v1/worker/gpu -name '*.py' | xargs wc -l | sort -rn
```

---

## 阅读顺序建议

- **只想要一张差异速查表**：01 → 02 → 03，然后看每章开头的「两边的锚点」表，不用读 prompt。
- **要完整理解重写动机**：01 → 04 → 05 → 08 → 11，这五章覆盖了三条轴各自最核心的那一刀。
- **要往 V2 迁移一个特性**：03 → 12 → 13。
- **要排查线上问题**：09（同步点在哪）→ 10（CUDA graph）→ 07（输入准备）。

04 和 05 是其余各章的地基：V2 所有「反直觉」的写法（不 pin 的 CPU 状态、UVA、staged write、kernel 内间接寻址）都源自这两章讲的那一个决定——**持久状态和模型输入解耦**。
