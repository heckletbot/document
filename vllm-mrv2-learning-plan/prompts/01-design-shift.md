# 01 · V1 的债务与 MRV2 的九条重构主张

目标：说清楚 MRV2 **不是性能优化补丁，而是一次从第一性原理出发的重写**。学完能复述九条主张各自解决 V1 的哪个具体痛点。

---

## 要看的内容

1. `docs/design/model_runner_v2.md`——官方设计文档，整篇精读，这是本章唯一的「权威文本」
2. `docs/assets/design/model_runner_v2/` 下的几张图（persistent batch v1 / mrv2、async_sched、async_race_condition）
3. `vllm/v1/worker/gpu/README.md`——一句话：实验性、Woosuk Kwon 主导
4. 对照着翻 `vllm/v1/worker/gpu_model_runner.py`（V1，约 7200 行单文件）和 `ls vllm/v1/worker/gpu/`（V2，约 70 个文件），先用体感确认「模块化」这条

---

## 要掌握的知识

### 九条主张速查

| # | 主张 | 针对 V1 的什么痛点 |
|---|------|-------------------|
| 1 | Persistent Batch 重做 | V1 把持久状态张量**直接当作**模型/采样器输入，布局和顺序被绑死；请求进出要做全张量重排，还得额外维护 `CachedRequestState` 备份 |
| 2 | Async-First | V1 不是按异步调度设计的，异步支持靠后补的 hack |
| 3 | 干掉 async barrier | V1 用「异步屏障」保护临界区，容易漏保护、组织僵硬、牺牲重叠度 |
| 4 | `StagedWriteTensor` | block table 这类大张量每步全量 H2D 拷贝太贵 |
| 5 | GPU 原生的输入准备与输出处理 | Python 构造 `input_ids`/`positions`/`query_start_loc` 是 CPU 瓶颈，且投机解码下 CPU 根本不知道某些值 |
| 6 | Triton 原生 sampler | V1 要先把全词表 logprobs 物化出来再做 top-k，显存峰值高；投机解码下还要把 per-request 采样状态展开成 per-logit 形状 |
| 7 | 模块化 | V1 的 `gpu_model_runner.py` 巨大且纠缠 |
| 8 | 不再滥用 `dummy_run` | V1 的 `dummy_run` 同时干 profiling、torch.compile、CUDA graph capture、warmup、DP 空转，容易和 `execute_model` 行为漂移 |
| 9 | 显式 CUDA Graph 管理 | V1 的图管理是隐式的，难推理、难扩展 |

### 必须钉死的三组因果

**第一组：持久状态 ≠ 模型输入。**
MRV2 预分配 `max_num_reqs` 行（默认 1024），每个请求在活跃期独占一行，**抢占按「结束」处理**，恢复时当新请求重新加入。每步按注意力后端要求的顺序从持久状态 **gather** 出输入张量。于是行的插入/删除不再引发全张量重排，`CachedRequestState` 这层冗余备份也就没有存在理由了。大状态放 GPU，gather 由 GPU 并行做。

**第二组：消除竞态靠「不共享」，不靠「加锁」。**
V1 的错误写法是 CPU 持久 buffer 本身是 pinned 的，然后 `.to("cuda", non_blocking=True)`——GPU 还在异步读，CPU 已经在改下一步的值。MRV2 的解法是持久 buffer **不 pin**，每步 `self.states.pin_memory()` 复制出一份临时 pinned 张量再发起异步拷贝。CPU 写 `self.states`，GPU 读 `tmp_states`，两者物理分离，不需要任何同步点。

**第三组：核心执行循环里没有 CPU 同步点。**
CPU 端只负责往 stream 上排队。所以显式同步（`torch.accelerator.synchronize`）和**隐式**同步（非 pinned 内存的 `.to("cuda")`）都是违规的。这条是理解后面所有章节奇怪写法的总钥匙。

### 容易读漏的两个细节

- **UVA**：让 GPU kernel 直接访问 CPU 上的大张量（例如 `prefill_token_ids`），避免往显存里复制一份。
- **`idx_mapping` 间接寻址**：投机解码下不再把每个请求的采样参数展开成 logits 的形状，而是在 kernel 内用一层映射把每条 logits 指回它所属请求的状态行。复杂采样参数和 logits processor 的支持成本因此大幅下降。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。

本轮任务：讲清楚 Model Runner V2（MRV2）相对 Model Runner V1 的设计思路转变。

必读材料（按顺序）：
1. docs/design/model_runner_v2.md 全文
2. vllm/v1/worker/gpu/README.md
3. vllm/v1/worker/gpu_model_runner.py（这是 V1）
4. vllm/v1/worker/gpu/ 目录（这是 V2）

请按下面结构输出：

A. 先用一张表列出设计文档里的九条主张，每条写：主张一句话 + V1 对应的具体痛点一句话。

B. 对「Persistent Batch」这条做深挖：
   - 在 V1 代码里指出「持久状态张量被直接当成模型/采样器输入」的具体位置（给文件:行号）。
   - 在 V1 代码里找到 CachedRequestState，说明它为什么必须存在。
   - 在 V2 代码里找到对应的替代物（提示：vllm/v1/worker/gpu/states.py 和 input_batch.py），
     说明「每请求固定一行 + 每步 gather」是怎么落地的。
   - 说明「抢占按完成处理」这条在 V2 代码里体现在哪。

C. 对「消除 async barrier」这条做深挖：
   - 在 V1 代码里找到 async barrier / 临界区保护的实现，说清楚它保护的是什么。
   - 在 V2 代码里找到「持久 CPU buffer 不 pin，每步临时 pin_memory() 再异步拷贝」的实际写法，
     给出至少两处真实调用点。
   - 解释为什么这样就不需要同步了，用「谁写谁读哪块内存」的方式讲。

D. 对「不滥用 dummy_run」和「显式 CUDA graph 管理」这两条，分别给出 V1 和 V2 的入口函数对比
   （V2 提示：vllm/v1/worker/gpu/model_runner.py 的 _dummy_run / profile_run / capture_model，
     以及 vllm/v1/worker/gpu/cudagraph_utils.py 的 CudaGraphManager）。

E. 最后用 150 字以内回答：如果让你用一句话概括 MRV2 的设计哲学，是什么？并指出设计文档里
   "Development Philosophy" 一节对后续迁移特性提出了什么要求。

约束：
- 只依据这个仓库里的真实代码和文档回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- V1 和 V2 都有名为 GPUModelRunner 的类，每次提到都要写清楚是
  vllm/v1/worker/gpu_model_runner.py 还是 vllm/v1/worker/gpu/model_runner.py。
```

---

读完这章，后面三条轴（04–07 / 08–10 / 11–13）分别是对九条主张里哪几条的深挖，见 [README](../README.md) 的「三条差异主轴」。
