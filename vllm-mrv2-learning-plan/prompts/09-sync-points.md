# 09 · 同步点在哪、结果怎么回传

> 轴二 · 异步调度

**对照的那件事**：采样结果从 GPU 回到 CPU 的路径上，一步里一共停顿几次、停在哪。

第 08 章讲的是「输入侧」的竞态，这一章讲「输出侧」的同步。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 异步输出类 | `gpu_model_runner.py` · `AsyncGPUModelRunnerOutput` / `AsyncGPUPoolingModelRunnerOutput` | `gpu/async_utils.py` · `AsyncOutput` / `AsyncPoolingOutput` |
| 簿记 | `_bookkeeping_sync()`（名字里就写着 sync） | `postprocess_sampled()` + `post_update` kernel（GPU 上做） |
| token 回传 | `_to_list()`：pinned 拷贝 + `transfer_event.synchronize()` + `tolist()` | `async_copy_to_np()` 在 copy_stream 上，延后到 `get_output()` |
| 草稿回传 | `_copy_draft_token_ids_to_cpu()` / `_get_draft_token_ids_cpu()` | `take_draft_token_ids()`，草稿留在 GPU |
| 有效计数 | `_copy_valid_sampled_token_count()` / `_get_valid_sampled_token_count()` | `num_sampled_tokens` 随异步拷贝一起走 |
| step 拆分 | `execute_model()` + `sample_tokens()`（已拆） | `execute_model()` + `sample_tokens()`，现场装在 `ExecuteModelState` |
| event 类型 | `transfer_event` | `copy_event = torch.cuda.Event(blocking=True)` |

---

## 差异的因果

### V1 里这条路被打过补丁

`_to_list()` 的注释是最好的入口：

```
# This is a short term mitigation for issue mentioned in
# https://github.com/vllm-project/vllm/issues/22754.
# `tolist` would trigger a cuda wise stream sync, which
# would block other copy ops from other cuda streams.
# A cuda event sync would avoid such a situation. Since
# this is in the critical path of every single model
# forward loop, this has caused perf issue for a disagg setup.
```

三层信息：(1) 原本直接 `tolist()`；(2) 它触发的是**设备级**流同步，会卡住别的 stream 上的拷贝；(3) 这在 PD 分离部署里造成了真实性能问题。补丁是换成 pinned 拷贝 + event 同步。

这是典型的「V1 不是按异步设计、靠补丁逼近」的例子。设计文档说的 "required retrofitted behavior and hacks" 就是指这类。

另一个入口是 `_bookkeeping_sync()` 这个名字：一步里要做的簿记（哪些请求结束、采了几个 token、NaN 统计、logprobs……）被集中在一个带 sync 的函数里。读它的返回值类型（一个八元组）能直观感受到耦合度。

### V2 的做法

`AsyncOutput` 构造时在独立的 **copy_stream** 上发一批 D2H 异步拷贝，`record` 一个 event 就返回；`get_output()` 才 `synchronize`。中间 CPU 可以去准备第 N+1 步。

几个细节值得对照着看：

- `copy_event` 用 `blocking=True` 创建，注释说明是为了**避免 busy-poll CUDA driver lock**——忙等会抢锁，拖慢正在排队的其他 CPU 工作。V1 的 `transfer_event` 对照着看是什么策略。
- 必须**持有 GPU 张量引用**，因为拷贝发生在和创建张量不同的 stream 上。
- 簿记被搬到 GPU：`post_update` / `post_update_num_computed_tokens` kernel 在步末更新持久状态，不需要 CPU 知道结果。
- 顺带搭车：EP 故障检查（`check_ep_fault`）和 NaN 统计也走同一个异步拷贝，在已有的同步点上一起取回，不额外开同步。

### step 拆成两次调用

两边都有 `execute_model()` + `sample_tokens()` 的拆分，但要对比**拆分的动机和中间状态的载体**：V1 和 V2 都有 `ExecuteModelState`，字段不同。拆分主要是为了在中间注入结构化输出的 grammar bitmask，同时让 CPU 在 GPU 跑 forward 时去干别的。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 在"采样结果回传 CPU"这条路径上的同步点分布。
V1 = vllm/v1/worker/gpu_model_runner.py（AsyncGPUModelRunnerOutput、
     _bookkeeping_sync、_to_list、_copy_* 系列）
V2 = vllm/v1/worker/gpu/async_utils.py + gpu/model_runner.py

=== 第一部分：数一数同步点 ===
1. 这是本章最重要的一问，请做得彻底：
   分别在 V1 和 V2 的"一个完整 step"路径上
   （从 worker.execute_model 进入，到调度器拿到 ModelRunnerOutput 为止），
   找出所有 CPU-GPU 同步点：
   .item() / .tolist() / .cpu() / .numpy() / Event.synchronize() /
   torch.cuda.synchronize() / Stream.synchronize() / 非 pinned 的 .to("cuda")。
   输出两张表：同步点 | 文件:行号 | 触发它的原因 | 它阻塞了什么。
   最后给出两边的同步点总数对比。
2. 对每个 V1 独有的同步点，回答：V2 是怎么消掉它的？
   （搬到 GPU？延后？搭车到别的同步点？还是根本不需要这个值？）

=== 第二部分：V1 的补丁痕迹 ===
3. 精读 _to_list() 的注释，把它引用出来并逐句解释：
   - 原本的实现是什么？
   - "cuda wise stream sync" 具体阻塞了谁？为什么在 PD 分离部署里特别痛？
   - 现在的 pinned + event 方案解决了多少、还剩什么问题？
4. 讲 _bookkeeping_sync()：它的返回值是一个几元组？列出每一项。
   回答：为什么这些东西必须在一个函数里一起算？
   这个函数里有哪些操作其实不需要同步、只是被顺带关进来了？
5. 讲 V1 的 AsyncGPUModelRunnerOutput.__init__ 和 get_output()：
   它已经做了多少异步化？还有哪些东西没能异步？
6. 讲 _copy_draft_token_ids_to_cpu / _get_draft_token_ids_cpu /
   _copy_valid_sampled_token_count / _get_valid_sampled_token_count：
   为什么草稿和有效计数要单独一套拷贝路径？V2 对应的是什么？

=== 第三部分：V2 的做法 ===
7. 逐行讲 AsyncOutput.__init__：
   - stream(copy_stream, main_stream) 上下文管理器做了什么？
   - copy_stream.wait_stream(main_stream) 在等什么？不加会怎样？
   - 列出所有被异步拷回的字段。
   - 构造返回的那一刻，CPU 能读这些数据吗？
8. 讲两条注释并对比 V1：
   - "We must retain references to the GPU tensors"：为什么跨 stream 会引入这个问题？
     V1 有没有同样的问题、怎么处理的？
   - copy_event 用 blocking=True 创建：busy-poll CUDA driver lock 会伤害谁？
     V1 的 transfer_event 是怎么创建的？对比一下。
9. 讲 V2 怎么把簿记搬到 GPU：post_update / post_update_num_computed_tokens /
   postprocess_sampled / postprocess_num_computed_tokens 各做什么，
   对比 V1 的 _update_states_after_model_execute() 和 _bookkeeping_sync()。
10. 讲"搭车"：check_ep_fault 和 num_nans 为什么能复用已有同步点而不新开一个？
    V1 的 _get_nans_in_logits() 是怎么做的？对比一下。

=== 第四部分：step 拆分 ===
11. 对比两边的 execute_model / sample_tokens 拆分：
    - 各自的 ExecuteModelState 装了哪些字段？差异在哪？
    - 拆分的动机是否相同（grammar bitmask 注入 + CPU/GPU 重叠）？
    - V2 在 sample_tokens 开头对 execute_model_state is None 的防御，
      V1 有没有等价物？
12. 画两张时序图（V1 一张、V2 一张）：连续三个 step 里，
    CPU 线、GPU main stream、copy stream 三条泳道上各发生什么，
    把同步点标成红线。目标是让人一眼看出两边的重叠度差异。

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 每次提到一个操作，明确它跑在哪条 stream 上。
- 第一部分的同步点清单要求穷尽，不要只举例；如果某条路径你没检查，明说没检查。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
