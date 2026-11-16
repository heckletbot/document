# 08 · 异步下的竞态怎么处理

> 轴二 · 异步调度

**对照的那件事**：异步调度下 CPU 正在改的 buffer，GPU 可能还在读。V1 加屏障，V2 拆共享。

对应设计文档第 2、3 条。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 机制 | 异步屏障 | 内存不共享 |
| 核心代码 | `gpu_model_runner.py` · `synchronize_input_prep()`（contextmanager） | `gpu/buffer_utils.py` · `UvaBackedTensor` + `UvaBufferPool` |
| 实现 | `prepare_inputs_event.synchronize()` → `yield` → `prepare_inputs_event.record()` | 真值 `self.cpu` 不 pin；每步 `copy_to_uva()` 轮转到另一个 pinned 槽 |
| 调用点 | 两处：`execute_model` 内、`_dummy_run` 内 | 无屏障，无调用点 |
| 保护范围 | 整个「输入准备」临界区，所有复用 CPU 张量 | 逐张量，由存储类型本身保证 |
| 漏保护的后果 | 静默数据竞争 | 结构上不可能发生（除非轮转深度配错） |

---

## 差异的因果

### V1 的屏障长什么样

```python
def synchronize_input_prep(self):
    if self.prepare_inputs_event is None:
        yield
        return
    # Ensure prior step has finished with reused CPU tensors.
    # This is required in the async scheduling case because
    # the CPU->GPU transfer happens async.
    self.prepare_inputs_event.synchronize()
    try:
        yield
    finally:
        self.prepare_inputs_event.record()
```

注释已经把问题说清楚了：复用的 CPU 张量 + 异步 H2D = 上一步的拷贝可能还没完成。屏障等上一步的 event，然后在临界区结束时记录新 event。

设计文档列了它的三个缺点，读的时候要在代码里逐条找证据：

1. **容易漏**。哪些 buffer 需要被保护没有任何机制强制，全靠写代码的人记得把操作放进 `with` 块。
2. **组织僵硬**。所有 CPU 工作必须待在屏障里，不能按逻辑自由拆分。
3. **重叠度下降**。`synchronize()` 是真等待，这段时间 CPU 什么都做不了。

### V2 为什么不需要屏障

竞态的前提是「CPU 写的内存」和「GPU 读的内存」是同一块。V1 的 `CpuGpuBuffer.cpu` 默认 pinned 且被反复复用，所以成立。

V2 把这个前提拆掉：`UvaBackedTensor.cpu` 显式 `pin_memory=False`，CPU 只写它；每步 `copy_to_uva()` 做一次 CPU→CPU 拷贝进轮转池里的某个 pinned 槽，GPU 只读那个槽。两块内存不同，竞态消失。

**代价**是多了一次 CPU→CPU 拷贝，以及轮转池的常驻内存。**正确性条件**只剩一条：轮转深度 ≥ 在途 step 数（`set_default_max_concurrency()` 从 engine 的 `batch_queue_size` 传下来）。这条件是集中的、可验证的，而不是散落在几十个 `with` 块里。

### 读的时候要注意的边界

V2 不是「没有任何同步」，而是「**核心执行循环**里没有」。初始化、profiling、捕获、shutdown 这些路径上仍然可以同步。判断一段代码是否违反 async-first，先问它在不在每步主路径上。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 的异步屏障和 V2 的"消除竞态"，说清楚两种思路的取舍。
V1 = vllm/v1/worker/gpu_model_runner.py 的 synchronize_input_prep 及其调用点
V2 = vllm/v1/worker/gpu/buffer_utils.py + gpu/model_runner.py

=== 第一部分：竞态本身 ===
1. 先把竞态讲清楚：在 V1 里指出一个具体的 CpuGpuBuffer，
   写出"CPU 在第 N+1 步写它、GPU 在第 N 步还在异步读它"的完整时间线
   （标出每个操作的文件:行号）。
2. 说明这个竞态为什么只在异步调度下出现，同步调度下为什么不存在。
3. 对照 docs/design/model_runner_v2.md 第 3 节的 unsafe 示例，
   指出 V1 代码里哪一行对应示例里的 self.states[req_idx] = new_req.data，
   哪一行对应 states = self.states.to("cuda", non_blocking=True)。

=== 第二部分：V1 的屏障 ===
4. 逐行讲 synchronize_input_prep()：prepare_inputs_event 是什么时候创建的、
   什么条件下为 None、synchronize() 和 record() 各自的语义。
5. 找出它的所有调用点（应该有两处），说明各自保护的临界区有多大、
   里面做了哪些工作。
6. 逐条验证设计文档列的三个缺点，每条都要给代码证据：
   a. "容易漏保护"：有没有任何机制强制某个 buffer 必须在屏障内被写？
      在 V1 里找出一个"看起来应该被保护、但不在任何 with 块里"的写入点，
      或者明确说明你没找到。
   b. "组织僵硬"：临界区里有多少 CPU 工作其实和竞态无关、只是被顺带关进去了？
   c. "重叠度下降"：synchronize() 这次等待在时间线上卡住了什么？
7. 在 V1 里找出其他同步原语：_sync_device()、transfer_event、
   _get_or_create_async_output_copy_stream() 等，
   说明它们和 synchronize_input_prep 是什么关系（互补还是重复）。

=== 第三部分：V2 的拆解 ===
8. 讲 V2 怎么把竞态的前提拆掉：
   - UvaBackedTensor.cpu 为什么显式 pin_memory=False？
     如果改成 True 会怎样（回到 V1 的问题吗）？
   - copy_to_uva() 的 CPU→CPU 拷贝发生在哪、拷多大、代价多少？
   - 轮转池怎么保证 GPU 正在读的槽不会被 CPU 覆盖？
9. 在 V2 的整个 gpu/ 包里搜索所有同步调用
   （.item() / .cpu() / .tolist() / synchronize / .numpy() / 非 pinned 的 .to("cuda")），
   逐个分类：
   - 在每步主路径上（execute_model / sample_tokens / prepare_* 的调用链里）；
   - 只在初始化、profiling、捕获、shutdown 路径上。
   第一类如果有，逐个解释为什么可以接受。
10. 明确回答：V2 的 async-first 约束的边界在哪？
    "核心执行循环无同步点"这句话里的"核心执行循环"具体包括哪些函数？

=== 第四部分：取舍对比 ===
11. 一张表对比两种方案：
    正确性保证方式 | 额外内存开销 | 额外拷贝开销 | 出错时的表现 |
    新增一个状态张量时开发者要记得做什么。
12. 回答：V2 的方案把"正确性条件"从"几十处 with 块"收敛成了几条？分别是什么？
    这些条件在代码里有没有被 assert 或注释固化？给位置。
13. 回答：V1 能不能在不改架构的前提下采用 V2 的方案？
    （提示：V1 的持久张量同时是模型输入，复制一份临时 pinned 副本意味着什么？
     顺着第 04 章的解耦去想。）

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 每次描述竞态时，明确写出"谁写哪块内存、谁读哪块内存"。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
