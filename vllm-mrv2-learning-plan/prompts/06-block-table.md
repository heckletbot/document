# 06 · block table 与注意力元数据

> 轴一 · 持久化批处理与输入准备

**对照的那件事**：block table 这张最大的持久表怎么维护，以及每步的注意力元数据怎么从它派生出来。

这一章是第 04、05 两章的合流点：block table 既是「挪行成本最高的表」，又是「全量拷贝最贵的表」。

---

## 两边的锚点

| | V1 | V2 |
|--|----|----|
| 文件 | `vllm/v1/worker/block_table.py` | `vllm/v1/worker/gpu/block_table.py` |
| 类 | `BlockTable` + `MultiGroupBlockTable` | `BlockTables`（单类直接管多 group） |
| 追加 | `append_row()` / `add_row()` | `append_block_ids()`（staged） |
| 腾挪 | `move_row()` / `swap_row()` / `clear_row()` | **没有** |
| 生效 | `commit_block_table(num_reqs)`（前 n 行全量 H2D） | `apply_staged_writes()`（只拷 diff + 一个 kernel） |
| 取用 | `get_device_tensor(num_reqs)`，直接把前 n 行喂给后端 | `gather_block_tables()`，按本步顺序 gather 出新张量 |
| slot mapping | `compute_slot_mapping()` + `ComputeSlotMappingKernel` | `compute_slot_mappings()` + `_compute_slot_mappings_kernel` |
| 多 group | `MultiGroupBlockTable` 持有多个 `BlockTable`，逐个转发 | 指针张量（`_make_ptr_tensor` / `init_block_table_layout_tensors`）交给同一个 kernel |
| 注意力元数据 | `gpu_model_runner.py` · `_build_attention_metadata()`（约 330 行） | `gpu/attn_utils.py` · `build_attn_metadata()` + `build_slot_mappings_by_layer()` |

---

## 差异的因果

### `move_row` / `swap_row` 是第 04 章的尾巴

V1 的 `condense()` 要挪行，block table 作为持久张量之一必须跟着挪——于是 `BlockTable` 不得不暴露 `move_row` / `swap_row`。表越宽，挪一行越贵。`MultiGroupBlockTable` 把这个成本乘以 group 数。

V2 的行号不变，这组方法整个消失。

### `commit_block_table` vs `apply_staged_writes`

V1 的 `commit_block_table(num_reqs)` 把 CPU 侧前 `num_reqs` 行整体拷到 GPU。一步里可能只有几个请求各追加了一个 block，但拷的是 `num_reqs × max_num_blocks` 个 int32。

V2 用第 05 章的 `StagedWriteTensor`：只打包真正改动的片段。block table 是设计文档点名的 `StagedWriteTensor` 头号场景。

### `get_device_tensor` vs `gather_block_tables`

这是第 04 章解耦的直接体现。V1 把持久表的前 n 行**原样**交给注意力后端，所以持久表的行顺序必须**就是**后端要的顺序——这是 `condense` 存在的根本原因。

V2 多了一次 gather：持久表按行号存，每步按 `idx_mapping` 的顺序 gather 出一张本步用的表。多出来的是一次 GPU gather kernel，换掉的是 CPU 上的整表重排。

### 多 KV cache group

两边都要处理「不同层有不同 KV 规格」（滑窗 vs 全注意力、Mamba 状态等）。V1 用组合：`MultiGroupBlockTable` 持有 N 个 `BlockTable`，每个方法循环转发。V2 用指针张量：把 N 张表的地址打包成一个指针张量，一个 kernel 处理全部 group。

对比这两种做法时注意看 kernel launch 次数的差异。

---

## Prompt

```text
你是 vLLM 源码对比助手。仓库在 <VLLM_REPO>。

本轮任务：对比 V1 和 V2 的 block table 维护方式与注意力元数据构造。
V1 = vllm/v1/worker/block_table.py + gpu_model_runner.py 的 _build_attention_metadata
V2 = vllm/v1/worker/gpu/block_table.py + gpu/attn_utils.py

=== 第一部分：block table 本身 ===
1. 一张表逐方法对照：
   V1 的 append_row / add_row / clear_row / move_row / swap_row /
   compute_slot_mapping / commit_block_table / get_device_tensor / get_cpu_tensor
   ↔ V2 的 append_block_ids / apply_staged_writes / gather_block_tables /
   compute_slot_mappings / get_dummy_block_tables / get_dummy_slot_mappings。
   对每一行标注："行为相同 / 行为不同 / V1 独有 / V2 独有"，独有的说明为什么另一边不需要。
2. 重点讲 move_row / swap_row：
   - 它们被谁调用（顺着 condense / swap_states 追上去）？
   - 一次调用要搬多少字节（用 max_num_blocks、group 数表达）？
   - V2 为什么完全不需要它们？把因果一路追回第 04 章的解耦。
3. 重点对比 commit_block_table(num_reqs) 和 apply_staged_writes()：
   - 各自的 H2D 字节数（给公式）；
   - 各自的 kernel launch 次数；
   - 在"100 个请求、这一步只有 3 个请求各追加 1 个 block"这个具体场景下，
     两者的实际拷贝量差多少倍？
4. 重点对比 get_device_tensor(num_reqs) 和 gather_block_tables()：
   - V1 为什么可以直接把持久表的前 n 行交给注意力后端？这对持久表的行顺序提了什么要求？
   - V2 多出来的这次 gather 代价是什么、换掉了什么？
   - 讲 _gather_block_tables_kernel 的 grid 划分和 program_id 含义。

=== 第二部分：多 KV cache group ===
5. 讲 V1 的 MultiGroupBlockTable：它怎么组织多个 BlockTable，
   每个方法是不是都要循环 N 次，slot mapping 是怎么分组计算的。
6. 讲 V2 的指针张量方案：_make_ptr_tensor() 和 init_block_table_layout_tensors()
   打包了什么，_compute_slot_mappings_kernel 怎么在一个 kernel 里处理所有 group，
   _load_ptr 这类辅助函数在 Triton 里怎么解引用。
7. 对比两者在 group 数为 N 时的 kernel launch 次数和 Python 开销。

=== 第三部分：注意力元数据 ===
8. 对比 V1 的 _build_attention_metadata()（gpu_model_runner.py）和
   V2 的 attn_utils.build_attn_metadata()：
   - 各自有多少行、各自从哪里取输入；
   - V1 从 self.* 上直接读了多少个属性？V2 从 InputBatch 这一个对象里取了多少个字段？
     （这条用来量化设计文档第 7 节说的"减少 model runner 属性耦合"。）
   - 两边分别怎么把 slot mapping 按层分发（V2 看 build_slot_mappings_by_layer）。
9. 对比两边怎么处理注意力后端对批顺序的要求：
   V1 的 _may_reorder_batch() / calculate_reorder_batch_threshold() ↔
   V2 的 sort_batch_req_ids()。
   说清楚一个关键差异：V1 的重排作用在持久状态上还是只作用在本步视图上？V2 呢？
10. 对比 seq_lens 相关：V2 的 InputBatch 为什么要同时有 seq_lens 和
    seq_lens_cpu_upper_bound？V1 对应的是什么？

=== 第四部分：收敛 ===
11. 一张表总结 block table 这条线上 V1 → V2 的全部变化，每行写：
    变化点 | V1 做法 | V2 做法 | 收益 | 这个收益依赖第 04 章的哪个前提。
12. 回答：如果只把 V1 的 CpuGpuBuffer 换成 StagedWriteTensor，
    但保留 condense()，能拿到多少收益？卡在哪？

约束：
- 这是一次 V1 与 V2 的对比，不是 V1 代码导读。任何只讲 V1 内部细节而不落到
  "V2 把它改成了什么、为什么" 的段落，请直接砍掉。
- 只依据我给的仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要同时给出 V1 和 V2 的 文件路径:行号；只找到一边就明说另一边没找到。
- 讲 Triton kernel 时必须说清楚 program_id 的维度含义。
- 涉及数据量对比的，一律给出公式和一个代入具体数字的例子。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
- V1 的类叫 BlockTable / MultiGroupBlockTable，V2 的叫 BlockTables（复数），
  每次提到都要写清楚是哪一个。
```
