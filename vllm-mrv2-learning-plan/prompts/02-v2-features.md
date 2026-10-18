# 02 · MRV2 当前支持的特性全景与启用开关

目标：搞清楚「我的这个启动命令，到底跑的是 V1 还是 V2」，以及 V2 这个包里到底已经长出了哪些特性。

---

## 要看的内容

1. `vllm/config/vllm.py` · `VllmConfig.use_v2_model_runner` 属性——**本章的核心函数**
2. `vllm/config/vllm.py` · `_get_v2_model_runner_unsupported_features()` / `_validate_v2_model_runner()`
3. `vllm/envs.py` · `VLLM_USE_V2_MODEL_RUNNER`（类型是 `bool | None`，三态）
4. `vllm/v1/worker/gpu_worker.py` · `_make_model_runner()` 以及所有 `self.use_v2_model_runner` 分支
5. `ls vllm/v1/worker/gpu/` 和各子包（`sample/`、`spec_decode/`、`model_states/`、`mm/`、`pool/`、`metrics/`）——用目录结构反推特性清单

---

## 要掌握的知识

### 开关是三态的，不是布尔

`VLLM_USE_V2_MODEL_RUNNER` 默认 `None`（未设置），语义是「交给配置自己判断」。判定顺序（读 `use_v2_model_runner` 属性，自上而下短路）：

| 顺位 | 条件 | 结果 |
|------|------|------|
| 1 | 配了 HiSparse（`attention_config.hisparse_config`） | **强制 V2**；显式设 `=0` 直接抛 `ValueError` |
| 2 | 配了 watermark（`watermark_config`） | **强制 V2**；显式设 `=0` 只打日志提示被覆盖 |
| 3 | 用户显式设了环境变量 | 听用户的 |
| 4 | 没装 Triton | 回退 V1 并告警（MRV2 依赖 Triton） |
| 5 | 命中 `_get_v2_model_runner_unsupported_features()` | 回退 V1 并告警 |
| 6 | 其余 | **默认 V2** |

注意第 1、2 条和第 3 条的顺序：**强制项排在用户开关之前**，所以存在「我设了 0 但还是跑 V2」的情况。这是排查环境问题时最容易踩的坑。

还要区分两条路径：
- `use_v2_model_runner`（属性）是**自动回退**，命中不支持项就悄悄降级到 V1 + 一条 `warning_once`。
- `_validate_v2_model_runner()` 是**硬校验**，命中就 `raise ValueError`。即用户显式要 V2 时不再允许静默降级。

### MRV2 的能力版图（按子包读）

| 子包 / 文件 | 承担的特性 |
|-------------|-----------|
| `model_runner.py` | 主循环：请求增删、输入准备、forward、采样、异步输出 |
| `states.py` / `block_table.py` / `buffer_utils.py` | 持久批状态、block table、`StagedWriteTensor` / UVA 基础设施 |
| `input_batch.py` | `InputBatch` 数据结构 + 一堆 Triton 输入准备 kernel |
| `attn_utils.py` | KV cache spec、注意力后端初始化、attn metadata 构造、KV sharing fast prefill |
| `cudagraph_utils.py` | `CudaGraphManager` / `ModelCudaGraphManager`、捕获与 dispatch、显存 profiling |
| `warmup.py` | 混合 prefill/decode warmup、kernel warmup |
| `sample/` | Triton sampler、gumbel 采样、top-k/top-p/min-p、penalties、bad_words、logit_bias、logprobs、prompt_logprobs、thinking budget、watermark、batch-sharded sampling、logits processor 接口 |
| `spec_decode/` | ngram / eagle / mtp / multi_module_mtp / gemma4 / dflash / dflash2 / dspark / lilicorr / standalone_ar / target_dependent_ar、拒绝采样、自适应验证、接受率估计 |
| `model_states/` | 按模型族分叉的状态：default / encoder_only / encoder_decoder / mamba_hybrid / prompt_embeds / mm_pruning / recoverssm |
| `mm/` | 多模态 encoder runner、encoder cache、M-RoPE、多模态 LoRA |
| `pool/` | pooling / embedding 模型路径 |
| `dp_utils.py` / `pp_utils.py` / `pcp_manager.py` / `ubatch_utils.py` / `eplb_utils.py` | DP、PP、prefill context parallel、micro-batch（DBO）、专家并行负载均衡 |
| `kv_connector.py` / `ec_connector.py` | KV / encoder-cache connector（PD 分离、KV offload） |
| `structured_outputs.py` | 语法 bitmask 应用（Triton kernel） |
| `lora_utils.py` | LoRA 状态与 CUDA graph 捕获用例 |
| `async_utils.py` | `AsyncOutput` / `AsyncPoolingOutput`、step timing 采集 |
| `metrics/` | logits 相关指标 |

这张表的用法：**先按目录判断「这个特性在 V2 里有没有落地」，再去读实现**。目录里没有对应文件，基本就等于还没迁移。

### 有几个特性是「只有 V2 才有」

读 `_get_v1_model_runner_unsupported_features()` 可以反向确认这些：PCP（prefill context parallel）、PP + 异步调度、DSpark 投机、自适应验证（adaptive verification）、draft lm_head 量化、混合 sliding/full 的 DFlash draft、DFlash candidate-head draft、diffusion 模型、batch-sharded sampling。再加上开关里硬编码的 HiSparse 和 watermark。

这意味着 MRV2 已经不只是「追平 V1」，它是新特性的默认落地点。细节放到第 03 章。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。

本轮任务：给我一份「MRV2 当前支持什么 + 怎么被启用」的全景。

第一部分 · 开关逻辑
1. 精读 vllm/config/vllm.py 里的 use_v2_model_runner 属性，把判定顺序画成一棵决策树，
   每个分支标注 文件:行号。
2. 说明 envs.VLLM_USE_V2_MODEL_RUNNER 的三态（None / True / False）各自的含义，
   并指出哪些条件会覆盖用户显式设置的 False、分别是报错还是只打日志。
3. 区分 use_v2_model_runner（自动回退）和 _validate_v2_model_runner()（硬校验）的调用场景，
   给出各自的调用点。
4. 在 vllm/v1/worker/gpu_worker.py 里找出 _make_model_runner()，说明 V2 分支下
   什么情况会实例化 MMEncoderModelRunner 而不是 gpu/model_runner.py 的 GPUModelRunner。
5. 列出 gpu_worker.py 里所有 if self.use_v2_model_runner 的分支，逐个说明 V1/V2 在
   worker 层面的行为差异（例如 workspace lanes、PP 最后一级 rank、pooling 模型）。

第二部分 · 特性版图
6. 遍历 vllm/v1/worker/gpu/ 的所有子目录和文件，输出一张表：
   文件 → 它负责的特性 → 行数 → 阅读优先级（高/中/低）。
   优先级判断标准：是否在每个 step 的主路径上。
7. 针对 sample/ 和 spec_decode/ 两个子包，单独展开成二级表，列出每个文件对应的
   采样功能 / 投机方法名（投机方法名请以 vllm/v1/worker/gpu/spec_decode/__init__.py 的
   init_speculator() 分派逻辑为准）。
8. 指出哪些特性是「只有 MRV2 才有，V1 根本不支持」的，依据是
   vllm/config/vllm.py 的 _get_v1_model_runner_unsupported_features() 和
   use_v2_model_runner 里的强制 V2 分支。

第三部分 · 实操
9. 给我 3 条 vllm serve 命令示例：一条必然走 V2、一条必然回退 V1、一条显式设了
   VLLM_USE_V2_MODEL_RUNNER=0 但仍然被强制走 V2。每条都说明命中了决策树的哪个分支。
10. 告诉我在日志里怎么确认当前跑的是哪个 runner（给出具体的日志字符串和它的产生位置）。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- 特性清单必须以代码为准，不要替我"推测"某个特性应该被支持。
```

---

## 自测题

1. `VLLM_USE_V2_MODEL_RUNNER` 不设置时，默认跑哪个 runner？依据是哪一行？
2. 哪两个特性会**无视**用户设的 `=0` 强行走 V2？两者在「用户设了 0」时的反应有什么不同？
3. 为什么没装 Triton 就必须回退 V1？MRV2 的哪些子系统是 Triton 写的？
4. `use_v2_model_runner` 和 `_validate_v2_model_runner()` 的区别是什么？什么时候用户会看到报错而不是静默降级？
5. `_make_model_runner()` 在 V2 分支下有两个可能的类，分别在什么条件下被选中？
6. 不看实现，只看 `ls vllm/v1/worker/gpu/sample/`，你能列出 MRV2 支持哪些采样特性吗？
7. `spec_decode/__init__.py` 的 `init_speculator()` 一共分派出多少种 speculator？哪一种需要额外传 `req_states`，为什么？
8. `model_states/` 为什么要按模型族分文件？列出其中三个文件各自对应的模型类型。
9. 说出三个「V1 不支持、只有 V2 有」的特性。
10. 线上只给你一段启动日志，你怎么判断它跑的是 V1 还是 V2？
