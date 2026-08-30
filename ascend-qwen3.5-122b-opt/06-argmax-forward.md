# 投机采样 ArgMax 前移

- 链路：Decode 侧 MTP draft sampling（投机解码出候选 token）
- 开关：`use_local_argmax_reduction`，默认关
- 适用：Qwen3.5 MTP / Qwen3Next MTP（已注入 `get_top_tokens`）
- 不适用：lm_head 做 TP 切分、draft 词表重映射、非 MTP draft

## 问题怎么识别

MTP draft 要用 argmax 出候选 token。原流程是：

1. 各卡算出 hidden
2. **先 AllGather 拼全局 hidden**
3. 再在拼好的大矩阵上做 sampling / argmax

两个独立浪费叠在一起：

- **计算冗余**：argmax 扫 `N × tp_size` 行，本卡其实只产生了 `N` 行
- **通信过大**：AllGather 的是 `N × hidden_dim` 的 fp32/fp16，而后面真正要的只是 token id

识别点在「draft sampling 和 TP 的顺序」，不是 MTP 接受率。

## 具体怎么改实现

把 argmax 挪到 AllGather 之前：各卡先对 **本卡 vocab 分片** 做局部 argmax，再 AllGather **token_id（int64，N×1）**，合并出全局 top-1。

| | 优化前 | 优化后 |
|--|--------|--------|
| argmax 计算量 | 全量 `N × tp_size` 行 | 本卡 `N` 行，缩至 `1/tp_size` |
| AllGather 内容 | `N × hidden_dim`（fp32/fp16） | `N × 1` token_id（int64） |

数学依据：**全局 top-1 一定落在某个 vocab 分片的局部 top-1 上**。所以局部 argmax 再合并，与先 gather 再全局 argmax 等价。

这要求 draft 的 vocab 是按 TP 分片、且没有重映射。lm_head 再切一刀 TP、或 draft 词表 remap，等价性不成立，所以开关默认关、并写明不适用条件。代码入口是 MTP 注入的 `get_top_tokens`。

## 学习要点

- 识别：看 **sampling 相对 AllGather 的位置**，通信体积是 hidden 还是 token。
- 实现：利用「全局 argmax = 分片局部 argmax 的 max」，把归约对象从向量变成标量 id。
