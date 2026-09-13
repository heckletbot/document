# MTP 零气泡异步调度

- 链路：Decode 侧 MTP 投机解码（每 step 的 CPU/NPU 同步）
- 收益：同步气泡 5ms+ → ~1ms，TPOT ↓ 1ms+
- 本项目开关：`ENABLE_ZERO_BUBBLE=1`（默认 0）
- 社区入口：`--async-scheduling` + `--speculative-config`（MTP/EAGLE/draft model）
- 通用：所有 MTP 投机解码，不限 Qwen3.5

社区主干：[vllm#32951](https://github.com/vllm-project/vllm/pull/32951)（重构自 [#29957](https://github.com/vllm-project/vllm/pull/29957)）。昇腾移植：[vllm-ascend#7640](https://github.com/vllm-project/vllm-ascend/pull/7640)，Hybrid 适配：[vllm-ascend#8461](https://github.com/vllm-project/vllm-ascend/pull/8461)。

## 问题怎么识别

传统投机解码每 step 都要等 CPU 把上一轮结果备好，NPU 才能开下一轮。关键数据是 `num_accepted_tokens`、`num_computed_tokens`：rejection sampler 在设备上算完，必须 **D2H 同步回 CPU**，CPU 改请求状态、组下一拍 input，再 H2D。profiling 上每个 Decode step 有一段 NPU 空闲，原文量到 **5ms+**。

0.19.rc1 社区已经有零气泡，但适配时发现 **没完全生效**：当时 `vllm_ascend/platform.py` 在「投机解码 + async scheduling」组合上会直接关掉 async——注释写明 Ascend 还没有 GPU 侧 `num_computed_tokens` 修正（`update_num_computed_tokens_for_batch_change`），硬开会精度发散。所以不是「升个版本就亮」，要补设备端修正并去掉这道禁令。

和 ArgMax 前移分开：那条改 draft sampling 相对 AllGather 的位置；这条改的是 **接受个数要不要每 step 拉回 CPU**。

## 社区具体怎么改

一句话：**乐观假设 + 异步执行 + 设备端修正**。CPU 先当「上一轮 draft 全接受」往下走，真正的对错在 NPU 上改，不把计数拉回 CPU 再开下一拍。

### 改前（有气泡）

```text
step N:  target forward → sample → D2H sync(accepted/computed)
         CPU 改 num_computed_tokens / 组下一拍 input
step N+1: 必须等上面 sync 完，NPU 才能 launch
```

`_update_states()` 里会调 `_get_valid_sampled_token_count()`，内部 `event.synchronize()`，把准备下一拍的 CPU 工作和上一拍 draft GPU 的重叠窗口掐死。

### 改后（零气泡）

```text
step N:   target forward → sample
          valid_sampled_token_count 留在设备上（可非阻塞抄一份给 scheduler 记账）
step N+1: CPU 按「全接受」算 optimistic seq_lens / num_computed_tokens，立刻 prepare + launch
          NPU kernel 用上一拍 valid_count 修正真实长度和 positions
          CPU 状态修正推迟到本拍 forward 已经发出之后
```

设备侧权威，CPU 侧乐观。

### 三个落点

**1. 乐观假设（CPU 不阻塞）**

`_update_states` / `_prepare_inputs` 不再 `synchronize()` 等接受个数。CPU 按上一轮 draft 全接受推进：

```text
optimistic_seq_lens = num_computed_tokens_cpu + num_scheduled_tokens
```

`#32951` 里 async 路径把 `num_accepted = prev_num_draft_len`、`num_rejected = 0`，placeholder 不在此刻补进 `output_token_ids`。

**2. 设备端修正（对上拒绝）**

NPU 上的 `update_num_computed_tokens_for_batch_change`（vllm-ascend：`vllm_ascend/spec_decode/utils.py`）：

```text
participating = (上一拍还在 batch) and (上一拍有 draft)
若 participating:
    num_computed_tokens = prev_gpu_computed + valid_sampled_token_count
    num_accepted_tokens = valid_sampled_token_count
否则（新请求 / prefill）:
    用 CPU 值
```

`valid_count = 1 + accepted_drafts`。被拒掉的长度：

```text
rejected = prev_drafts + 1 - valid_count
```

CPU 上的 `correct_optimistic_seq_lens_cpu` 用同一式子把乐观 `seq_lens` 减回去，镜像 GPU kernel，避免再做一次 NPU→CPU 的 `seq_lens` 同步。positions / slot mapping 用修正后的 `num_computed_tokens` 在设备上重算（`#32951` 把 `compute_slot_mapping` 也搬到 GPU）。

早期设计里对应 `_maybe_adjust_inputs_on_gpu`；记账推迟到 `_update_states_after_model_execute`（hybrid 还要用接受个数去挪 GDN/Mamba 状态）。

**3. 异步执行（重叠）**

上一拍 draft GPU 与本拍 `_update_states` + `_prepare_inputs` + 图 launch 重叠。零气泡划得着的时候：上一拍 draft GPU 时间 < 本拍 CPU 准备时间。额外开销主要是设备上改 positions/seq_lens（社区 profile 大约 100–200μs @ bs=32），可以叠在 GPU 计算里。

### 本项目在昇腾上还补了什么

| 点 | 为什么 |
|----|--------|
| 移植 `#32951` → `#7640` | NPU runner 跟上 GPU 零气泡语义 |
| `#8461` 去掉 platform 禁令 | 0.19.rc1 默认把 async+spec 关掉 |
| `optimistic_seq_lens_cpu` + 非阻塞回抄 / CPU 侧校正 | 昇腾 page attention 要 CPU 上的 `seq_lens`，不能只留 GPU tensor |
| Hybrid（GDN）接受个数 | Qwen3.5 线性注意力只留 last-token 状态，接受个数错了会恢复错 SSM slot。社区后续还有 `#38556` / `#45100` 修 async 下 row 重排竞态 |

`ENABLE_ZERO_BUBBLE=1` 是本项目开关；社区对用户暴露的是 `--async-scheduling`。

## 效果

优化前每个 Decode step 等 `num_accepted_tokens` 等 CPU 同步，NPU 空一截；优化后 NPU 按乐观假设连发，设备端修正，气泡 5ms+ → ~1ms，TPOT 再降 1ms+。社区 GPU 侧 `#32951` 在 DeepSeek-V3.2 MTP 上 TPOT 9.19→8.90ms（约 3%），机制相同。

## 学习要点

- 识别：看 Decode step 之间的 **D2H sync 空闲**，对象是接受个数 / computed tokens，不是 MTP 算子本身慢。
- 实现：CPU 先当全接受；`valid_sampled_token_count` 留在设备上；kernel 用 `prev + valid_count` 改长度。
- 昇腾 0.19 开箱不亮，是因为缺设备端修正时 platform 主动禁了 async+spec。
