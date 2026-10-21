# 05 · 状态怎么写、怎么拷到 GPU

> 轴一 · 持久化批处理与输入准备

**对照的那件事**：CPU 侧改了某个请求的状态之后，这个改动怎么到达 GPU。

第 04 章解决了「行不用搬」，这一章解决「行怎么更新」。对应设计文档第 3、4 条。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 缓冲抽象 | `vllm/v1/utils.py` · `CpuGpuBuffer` | `vllm/v1/worker/gpu/buffer_utils.py` · `StagedWriteTensor` / `UvaBackedTensor` / `UvaBufferPool` / `UvaBuffer` |
| 构造处 | `gpu_model_runner.py` · `_make_buffer()` | `gpu/states.py` 各字段按用途各选一种 |
| CPU 侧真值 | `self.cpu`，**默认 pinned**（`pin_memory=PIN_MEMORY`） | `UvaBackedTensor.cpu`，**显式 `pin_memory=False`** |
| 拷贝方式 | `copy_to_gpu(n)`：前 n 行**全量**拷 | `StagedWriteTensor`：只拷这一步的 ragged diff；`UvaBufferPool`：拷进轮转的临时 pinned 槽 |
| 大张量 | 显存里存一份同形状的 `self.gpu` | UVA：GPU kernel 直接读 CPU 内存，显存里不存副本 |
| 竞态处理 | 靠 `synchronize_input_prep()` 屏障（见第 08 章） | 靠「CPU 写的和 GPU 读的不是同一块内存」 |

---

## 差异的因果

### V1 的 `CpuGpuBuffer` 长什么样

一对同形状的张量：pinned 的 `self.cpu` + 显存上的 `self.gpu`，外加一个 numpy 视图 `self.np`。`copy_to_gpu(n)` 把前 n 行整体拷过去。

两个代价叠在一起：

1. **全量拷贝**。block table 这种 `(max_num_reqs, max_num_blocks)` 的大张量，哪怕这一步只有一个请求追加了一个 block，也要拷整块。
2. **CPU 真值本身是 pinned 的**。这意味着 CPU 下一步往 `self.cpu` 写新值时，GPU 可能还在异步读它——这正是设计文档第 3 节画的那个竞态。V1 用屏障挡住，V2 直接拆掉共享。

### V2 的两种替代

**`StagedWriteTensor`** 针对「大 + 增量」：基张量常驻 GPU，这一步的写入先在 CPU 暂存，`apply_write()` 把所有 ragged 写入打包成连续缓冲，一次 H2D，再用一个 Triton kernel（`_apply_write_kernel`）散射回基张量。kernel launch 压到 1 次，拷贝量从「整张表」降到「这一步真改了的那些片段」。`FusedStagedWriter` 再把多个张量的写入合并。

**`UvaBackedTensor` + `UvaBufferPool`** 针对「小 + 每步全量」：真值 `self.cpu` **不 pin**；`UvaBufferPool` 预分配 `max_concurrency` 个 pinned `UvaBuffer`，每次 `copy_to_uva()` 轮转到下一个。CPU 写真值，GPU 读临时槽，物理分离，不需要同步。

设计文档示意的是每步 `self.states.pin_memory()` 现分配，实现换成了轮转池以省掉分配开销。**轮转深度必须 ≥ 在途 step 数**（`_DEFAULT_MAX_CONCURRENCY = 2`，`set_default_max_concurrency()` 从 engine 的 `batch_queue_size` 传下来），否则会回绕到一个 GPU 还在读的 buffer。

**UVA** 则是第三条路：`get_accelerator_view_from_cpu_tensor()` 把 pinned CPU 张量包成 GPU kernel 能直接寻址的 view。`RequestState.all_token_ids` 形状是 `(max_num_reqs, max_model_len)`，可能几个 GB，默认走 UVA 就省掉了整份显存副本。

### 结果

同一件事（「把 CPU 的状态改动送到 GPU」）在 V1 只有一种做法，在 V2 按「大小 × 访问模式」分成了三种。这是模块化的副产品，也是显存和带宽收益的主要来源。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 把"CPU 侧状态改动送到 GPU"这件事的实现差异。
V1 = vllm/v1/utils.py 的 CpuGpuBuffer + vllm/v1/worker/gpu_model_runner.py
V2 = vllm/v1/worker/gpu/buffer_utils.py + gpu/states.py

=== 第一部分：V1 的单一机制 ===
1. 逐行讲 CpuGpuBuffer（vllm/v1/utils.py）：self.cpu / self.gpu / self.np 的关系、
   pin_memory 默认值、copy_to_gpu(n) 和 copy_to_cpu(n) 各做什么。
   注意 copy_to_gpu 里 cpu.pin_memory() 这一行：当 self.cpu 已经是 pinned 时它等于什么？
2. 在 gpu_model_runner.py 里找出所有 _make_buffer() 的调用点，
   列出 V1 一共有多少个 CpuGpuBuffer、各自的形状量级。
   挑出其中最大的两三个，算出它们每步全量拷贝的字节数（用 max_num_reqs、
   max_model_len、max_num_blocks 等符号表达）。
3. 回答：这些 buffer 里有多少是"每步只改几行、却要整块拷"的？给具体例子。
4. 指出 V1 的 CPU 真值是 pinned 的这一事实，并说明它和 docs/design/model_runner_v2.md
   第 3 节描述的竞态是怎么对应的。（屏障本身放到第 08 章讲，这里只确认竞态存在。）

=== 第二部分：V2 拆成三种机制 ===
5. 讲 StagedWriteTensor 的完整生命周期：stage_write / stage_write_elem /
   apply_write / clear_staged_writes。
   重点讲 apply_write()：暂存的 ragged 写入是怎么打包成连续缓冲的？
   _apply_write_kernel 这个 Triton kernel 的输入布局是什么、怎么散射回基张量、
   program_id 的维度含义是什么？
   再讲 FusedStagedWriter 在它之上又合并了什么。
6. 讲 UvaBackedTensor + UvaBufferPool + UvaBuffer 三者的分工：
   - UvaBackedTensor.cpu 为什么显式 pin_memory=False？
   - UvaBufferPool 的轮转机制：_DEFAULT_MAX_CONCURRENCY 默认值、
     set_default_max_concurrency() 被谁调用传什么值、
     轮转深度小于在途 step 数会发生什么具体错误？
   - copy_to_uva() 里的动态扩容（1 << (n-1).bit_length()）在什么条件下触发，
     注释说的 "callers must retire a slot's GPU readers before reuse" 由谁保证？
7. 讲 UVA 这条路：get_accelerator_view_from_cpu_tensor() 做了什么，
   NonUvaBuffer 这条兜底在什么平台上被选中、丢掉了什么。
8. 讲 gpu/states.py 里每个字段的存储策略选择：
   做一张表，字段 | 选了三种机制里的哪一种（或裸 numpy / 裸 GPU 张量）| 选它的理由。
   重点解释 all_token_ids 的 uva_instead_of_gpu 开关：
   算出 max_num_reqs × max_model_len × 4 字节的量级，说明默认走 UVA 的理由，
   以及 use_dense_all_token_ids 为谁打开。
9. 讲 RequestState.apply_staged_writes() 把哪几列一次性兑现，
   以及它在 step 流程里的调用时机。

=== 第三部分：差异收敛 ===
10. 一张表对比：同一个状态列（挑 block table、all_token_ids、num_computed_tokens
    这三个代表）在 V1 和 V2 各自用什么机制、每步的 H2D 字节数、kernel launch 次数、
    显存占用。
11. 回答：V2 为什么要把一种机制拆成三种？分类的依据是什么（大小？访问模式？读写方？）
12. 回答：V1 的 CpuGpuBuffer 有没有办法在不改整体架构的前提下获得 StagedWriteTensor
    的收益？如果不能，卡在哪？（这条用来确认第 04 章的解耦是前提条件。）
13. 对照 docs/design/model_runner_v2.md 第 3、4 节，
    指出文档里的示意代码和实际实现有哪些出入，以及出入的原因。

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 讲 Triton kernel 时必须说清楚 program_id 的维度含义。
- 区分 CPU numpy 视图（.np）、CPU 张量（.cpu）、UVA view（.uva()）、纯 GPU 张量，
  每次提到都要写清楚是哪一个。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```
