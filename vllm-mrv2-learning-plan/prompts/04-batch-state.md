# 04 · 请求进出批时，状态怎么挪

> 轴一 · 持久化批处理与输入准备

**对照的那件事**：一个请求加入批、结束、被抢占时，持久状态张量里发生了什么。

这是 V1 → V2 最根本的一刀。设计文档第 1 条，其余大半改动都是它的推论。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 文件 | `vllm/v1/worker/gpu_input_batch.py` | `vllm/v1/worker/gpu/states.py` + `gpu/input_batch.py` |
| 请求状态 | `CachedRequestState`（dataclass，**每请求一份 Python 对象备份**） | 无等价物（被消掉了） |
| 批状态 | `InputBatch`（持久张量**就是**模型/采样器输入） | `RequestState`（持久）与 `InputBatch`（每步 dataclass 视图）**分离** |
| 加入 | `_register_add_request()` / `add_request()` | `RequestState.add_request()`，从 `free_indices` 弹一个槽 |
| 移除 | `remove_request()` | `RequestState.remove_request()`，槽位还回 `free_indices` |
| 腾挪 | `swap_states(i1, i2)`、`condense()` | **没有**。行号在请求活跃期内不变 |
| 抢占 | 单独处理 | 当作「结束」，恢复时按新请求重新加入 |
| 元数据刷新 | `refresh_metadata()` / `_make_sampling_metadata()` | 每步 gather，无需刷新 |
| runner 侧 | `_update_states()`（约 370 行）、`_may_reorder_batch()` | `add_requests` / `update_requests` / `finish_requests` / `free_states` / `_remove_request` |

---

## 差异的因果

### V1 为什么必须 `condense()`

V1 把持久张量**直接当作模型和采样器的输入**。模型输入要求前 `num_reqs` 行是紧凑的、顺序符合注意力后端的要求。于是一旦中间某行的请求结束，就必须把后面的行往下挪——`condense()` 的 docstring 写得很直白：「Slide non-empty requests down into lower, empty indices」。

挪行意味着**每一个**持久张量（block table、采样参数、token ids……）都要跟着挪。`BlockTable.move_row()` / `swap_row()` 就是为这个存在的。

### `CachedRequestState` 为什么必须存在

因为持久张量的行会被覆盖，而请求可能还活着。V1 需要一份不会被覆盖的权威副本，这就是 `CachedRequestState`。设计文档把它称作 "a redundant backup copy of request state"。

### V2 的解法

解耦。持久状态（`RequestState`）**不再是**模型输入；每步按本步需要的顺序从持久状态 gather 出 `InputBatch`。于是：

- 行号在请求活跃期内**永不变化** → 不需要 `condense` / `swap_states` / `move_row`。
- 行不会被覆盖 → 不需要 `CachedRequestState`。
- 批内顺序由谁决定？由本步的 gather 顺序决定，和持久布局无关（`sort_batch_req_ids()`）。
- 大状态放 GPU，gather 由 GPU 并行做，代价很低。

抢占当成「结束」是这个设计的配套简化：恢复时重新 `add_request`，拿到的可能是另一行，无所谓。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 在"请求进出批"这件事上的状态管理差异。
V1 = vllm/v1/worker/gpu_model_runner.py + vllm/v1/worker/gpu_input_batch.py
V2 = vllm/v1/worker/gpu/ 这个包

=== 第一部分：V1 做了什么（只讲到足够对比为止）===
1. 讲 CachedRequestState（gpu_input_batch.py）：它存了哪些字段，和 InputBatch 里的
   持久张量有哪些是重复的。再回答：为什么它必须作为独立副本存在？
2. 讲 InputBatch 的持久张量清单：列出 __init__ 里分配的所有张量，
   标注它们是否被直接当作模型输入或采样器输入使用（给使用点行号）。
   这一条是整章的关键证据，请找全。
3. 讲 add_request / _register_add_request / remove_request / swap_states / condense
   五个方法的协作：一次"中间某个请求结束"会触发哪些张量操作？
   把 condense() 的 docstring 原样引用，并说明它要搬多少份数据
   （把需要跟着搬的张量都列出来）。
4. 讲 refresh_metadata() 和 _make_sampling_metadata()：
   为什么每次批变动后都要重建元数据？
5. 讲 gpu_model_runner.py 的 _update_states()：它有多少行，处理了哪几类事件
   （新增 / 恢复 / 结束 / 抢占 / 草稿更新 ...），以及 _may_reorder_batch() 在做什么。

=== 第二部分：V2 改成了什么 ===
6. 讲 vllm/v1/worker/gpu/states.py 的 RequestState：
   req_id_to_index / index_to_req_id / free_indices 三件套怎么构成槽位分配器，
   add_request / remove_request 的完整动作。明确回答：有没有任何张量被搬动？
7. 讲 V2 的 InputBatch（gpu/input_batch.py）为什么是每步新建的 dataclass 而不是持久对象，
   它和 InputBuffers、和 RequestState 分别是什么关系（这是三层，请讲清楚）。
8. 在 V2 里找出"每步 gather"的实际代码：
   gather_batch_req_state() / prepare_inputs() / BlockTables.gather_block_tables()，
   说明本步的请求顺序由谁决定（提示：sort_batch_req_ids），
   以及这个顺序和持久布局为什么可以完全无关。
9. 讲 V2 的请求生命周期方法：model_runner.py 的 finish_requests / free_states /
   add_requests / update_requests / _remove_request，说明调用顺序及其理由。
10. 在 V2 里搜索 CachedRequestState 的等价物，确认它确实不存在，
    并说明 V2 靠什么保证"行不会被覆盖而请求还活着"这个问题不再发生。

=== 第三部分：差异收敛 ===
11. 用一张表对比三个事件在两边的操作成本：
    事件（请求加入 / 中间请求结束 / 请求被抢占后恢复）
    × V1 的动作与数据搬运量 × V2 的动作与数据搬运量。
    搬运量请用"O(什么)"的形式表达，并说明常数项里有什么（比如要搬几张表）。
12. 回答：V2 把搬运成本转移到哪里去了？（提示：每步 gather）
    为什么这个转移是划算的？在什么情况下可能不划算？
13. 对照 docs/design/model_runner_v2.md 第 1 节，
    指出文档里的哪几句话分别对应你在代码里找到的哪个事实。
14. 最后一句话总结：这一刀砍下去之后，后面哪些改动变成了"必然推论"？

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- V1 和 V2 都有名为 GPUModelRunner / InputBatch 的类，每次提到都要写清楚是哪个路径下的。
```
