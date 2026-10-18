# 04 · 入口与一个 step 的生命周期

目标：能把一次 `scheduler_output` 从 worker 入口一路讲到 `ModelRunnerOutput`，中间每一步在哪个函数、改了哪块状态、有没有 GPU 同步。这是全书的主干，后面每一章都是在给这条主干补细节。

---

## 要看的内容

1. `vllm/v1/worker/gpu_worker.py` · `_make_model_runner()`、`execute_model()`、`sample_tokens()`、`take_draft_token_ids()`
2. `vllm/v1/worker/gpu/model_runner.py` · `GPUModelRunner.__init__`（看清楚构造了哪些子系统）
3. 同文件 · `execute_model()`（主路径）、`sample_tokens()`、`postprocess_sampled()`、`pool()`
4. 同文件 · 请求增删四件套：`finish_requests()` / `free_states()` / `add_requests()` / `update_requests()`
5. 同文件 · `gather_batch_req_state()` / `prepare_inputs()` / `prepare_attn()` / `sample()`
6. 同文件末尾 · `ExecuteModelState`、`BatchReqState`、`sort_batch_req_ids()` 三个小而关键的定义

---

## 要掌握的知识

### 一个 step 被切成两次调用

MRV2 把一步拆成 `execute_model()` 和 `sample_tokens()` 两个 worker 入口。中间靠 `self.execute_model_state`（一个 `ExecuteModelState` NamedTuple）传递。

这个拆分不是为了好看——中间要插入**结构化输出的 grammar bitmask**（`sample_tokens(grammar_output)` 的入参），同时让 CPU 在 GPU 跑 forward 的时候去干别的。`sample_tokens()` 开头有一句防御：`execute_model_state is None` 说明上一次 `execute_model` 失败了，直接返回 `None`。

### `execute_model()` 的骨架

非 dummy 路径下，开头是一组**状态更新**，顺序有意义：

```text
update_pp_decode_requests()   # PP 下补齐上一轮的 decode 请求
finish_requests(scheduler_output)
free_states(scheduler_output)
add_requests(scheduler_output)
update_requests(scheduler_output)
block_tables.apply_staged_writes()   # 把这一轮暂存的 block table 写入刷到 GPU
```

然后是一个短路：`total_num_scheduled_tokens == 0` 时不跑模型，直接走 `kv_connector.no_forward()`。这条路径在 PD 分离场景下很常见（这一步只有 KV 传输没有计算）。

接着是**批形状决策**：

```text
gather_batch_req_state()      # 收集 CPU 侧的 BatchReqState，并给出 uniform_tok_count
  └─ sort_batch_req_ids()     # 排序：verification/decode → short_extend → prefill
dispatch_cg_and_sync_dp()     # 选 CUDA graph 描述符 + 跨 DP rank 同步 padding
```

`sort_batch_req_ids()` 这个小函数值得单独记：它保证 **decode 类请求排在前面**，因为下游 `split_decodes_and_prefills` 依赖这个顺序。MRV2 的「批内顺序由注意力后端决定」这条设计主张，落地点就在这里。

再往后是**输入准备**：

```text
prepare_inputs()              # 产出 InputBatch
prepare_attn()                # block_tables + slot_mappings
model_state.preprocess_state()  # 模型族特定的前处理（例如 Mamba 的跨 block 状态迁移）
```

最后是 **forward 分派**，三选一：ubatch runner / `cudagraph_manager.run_pw_graph()`（PIECEWISE）/ 直接 `self.model(**model_inputs)`（eager）。外面包着 `kv_connector.pre_forward()` 和 `finish_forward()`。

收尾：最后一个 PP rank 取 `hidden_states`（可能还有 `aux_hidden_states`），非最后一级 rank 返回 `IntermediateTensors` 给 PP 发送；两种情况都会把现场塞进 `ExecuteModelState`。

### `sample_tokens()` 做什么

从 `execute_model_state` 把现场取回来，然后：算 logits → 应用 grammar bitmask → `sample()` → 投机解码的草稿生成 → `postprocess_sampled()` → 包装成 `AsyncOutput`。草稿 token 通过 `take_draft_token_ids()` 单独取走。

### dummy run 的新定位

`execute_model()` 带 `dummy_run` / `skip_attn_for_dummy_run` / `is_profile` / `randomize_inputs` 等参数，**dummy 是主路径的一种模式，不是另一条代码路径**。`_dummy_run()` / `profile_run()` 都委托给它。这正是第 01 章第 8 条主张的落地。CUDA graph 捕获则是独立路径（`capture_model()`），不走这里。

### 读这一章时要随手记的三件事

1. 每个函数**改了哪些持久状态**（`req_states`、`block_tables`、`sampler` 的状态行）。
2. 每个函数**有没有 CPU-GPU 同步**（按 async-first 原则，主路径上应该一个都没有）。
3. 哪些是 CPU 侧的 numpy（带 `_np` 后缀），哪些是 GPU 张量。MRV2 里这两套是并行维护的，`InputBatch` 的字段命名就能看出来。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：把一个 step 的完整生命周期讲清楚。

第一部分 · 入口
1. 从 vllm/v1/worker/gpu_worker.py 出发，画出 worker → model runner 的调用关系：
   execute_model() / sample_tokens() / take_draft_token_ids() 分别转发到哪里。
2. 解释为什么一步要拆成 execute_model 和 sample_tokens 两次调用。
   重点说明 ExecuteModelState 传递了哪些现场、grammar_output 为什么必须在这个缝隙里注入。
3. 读 GPUModelRunner.__init__（vllm/v1/worker/gpu/model_runner.py），
   列出它构造的所有子系统（req_states / block_tables / sampler / speculator /
   cudagraph_manager / kv_connector / model_state / pp_handler / pcp_manager / ubatch_runner 等），
   每个一句话职责 + 它在哪一章的文件里。

第二部分 · execute_model 主路径
4. 按执行顺序逐段讲 execute_model()，每段给出 起止行号 + 这段干了什么 +
   它读/写了哪些持久状态。至少覆盖这些段落：
   a. 请求状态四件套：update_pp_decode_requests / finish_requests / free_states /
      add_requests / update_requests，说明这五个为什么是这个顺序。
   b. block_tables.apply_staged_writes() 为什么必须在这里调用。
   c. total_num_scheduled_tokens == 0 的短路分支，什么场景会走到。
   d. gather_batch_req_state() 产出的 BatchReqState 每个字段的含义。
   e. sort_batch_req_ids() 的排序键，以及为什么 decode 类请求必须排前面
      （提示：split_decodes_and_prefills）。
   f. dispatch_cg_and_sync_dp() 做的两件事。
   g. prepare_inputs / prepare_attn / model_state.preprocess_state 三者的分工。
   h. forward 的三种分派方式（ubatch / PIECEWISE 图 / eager）各自的触发条件。
   i. PP 最后一级 rank 和非最后一级 rank 的返回值差异。

第三部分 · sample_tokens 主路径
5. 同样逐段讲 sample_tokens()：取回现场 → 算 logits → grammar bitmask →
   sample() → 草稿生成 → postprocess_sampled() → 包装 AsyncOutput。
6. 说明 postprocess_sampled() 和 postprocess_num_computed_tokens() 各自更新了什么状态，
   以及为什么要在采样之后才更新。
7. take_draft_token_ids() 为什么是独立的一次取用，而不是塞在输出里？

第四部分 · dummy / profile
8. 说明 execute_model() 的 dummy_run / skip_attn_for_dummy_run / is_profile /
   randomize_inputs / valid_dummy_state_slots 这几个参数各自改变了哪些分支。
9. 说明 _dummy_run() / profile_run() / _dummy_sampler_run() / _dummy_pooler_run()
   如何委托给 execute_model，以及 capture_model() 为什么是独立路径。
   把这和 docs/design/model_runner_v2.md 第 8 节对应起来。

第五部分 · 核对 async-first
10. 在 execute_model + sample_tokens 的主路径上，逐行找出所有可能的 CPU-GPU 同步点
    （.item() / .cpu() / .tolist() / torch.cuda.synchronize / 非 pinned 的 .to("cuda")）。
    对每一个说明它是真同步还是被规避了（例如走 AsyncOutput 延后取回）。
    如果一个都没有，明确说"主路径无同步点"并给出你的检查范围。

最后输出一张 ASCII 时序图：一个 step 里 CPU 做什么、GPU 做什么、两者怎么重叠。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- 遇到 _np 后缀的 numpy 数组和同名 GPU 张量，每次都要写清楚是哪一个。
```

---

## 自测题

1. 一个 step 为什么要拆成 `execute_model` 和 `sample_tokens` 两次 worker 调用？中间塞进来的是什么？
2. `ExecuteModelState` 里装了哪些字段？如果 `execute_model` 抛异常了，`sample_tokens` 怎么表现？
3. `finish_requests` / `free_states` / `add_requests` / `update_requests` 为什么是这个顺序？调换会出什么问题？
4. `block_tables.apply_staged_writes()` 为什么要在请求增删之后、prepare 之前调用？
5. `total_num_scheduled_tokens == 0` 时这一步还做什么？什么部署形态下会频繁命中？
6. `sort_batch_req_ids()` 的排序键是什么？三类请求的先后顺序是什么？谁依赖这个顺序？
7. `dispatch_cg_and_sync_dp()` 为什么要把 CUDA graph 选择和 DP padding 同步放在一起做？
8. forward 的三种分派分别在什么条件下被选中？
9. PP 的非最后一级 rank，`execute_model()` 返回的是什么？`sample_tokens()` 在它上面还做事吗？
10. `dummy_run=True` 时，哪些状态**不会**被修改？为什么 MRV2 要让 dummy 走主路径而不是另写一份？
