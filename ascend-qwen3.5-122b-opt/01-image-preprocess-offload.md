# 图片预处理下沉 Worker

- 链路：API Server → 多模态 Worker（ViT 前）
- 场景：6P(TP4)1D(TP4DP4)；11K in / 0.7K out；13.5 图/请求；视觉 token ≥30%；单卡 QPS 0.4
- 收益：API Server 236ms → 44ms（-77.55%）；Worker 新增 30.7ms；单点净 161.3ms；**端到端 TTFT ↓ 80ms+**（预处理与 ViT 异步重叠，所以不等于单点净收益）

## 问题怎么识别

不是「多模态慢」一句话，而是把 API Server 关键路径拆模块计时。

单请求 20 图（单图 250+ token）时，API Server 端到端 236ms，占多模态前处理总耗时 261ms 的 **75%**：

| 模块 | 耗时 | 根因判断 |
|------|------|---------|
| `download_picture` | 120ms | 下载保存路径有冗余编解码 |
| `shm copy2buff` | 54ms | 大 tensor pickle + 跨进程 shm |
| `pre_process` | 32ms | HF Processor resize/normalize |
| `post_process` | 15ms | 浮点 tensor cast 到 model dtype |

再压了四个「改不动」的约束，用来排除假方案：

1. **GIL**：关键模块已用线程池，20 图时 shm copy 25ms、preprocess 16ms，仍只吃满单核。扩多 API Server 只解决请求间分流，解决不了**单请求内**并行。
2. **搬运不可绕**：预处理只要留在 API Server，`pickle → shm 写 → Worker shm 读 → pickle 反序列化` 就在。Worker 还要再付 7.7ms（shm 读）+ 2ms（反序列化）。
3. **三级缓存不能破**：
   - API Server `mm_processor_cache` ← `mm_hash`
   - Engine Core APC ← `identifier + mm_position`（identifier 由 hash + position 来）
   - Worker `encoder_cache` ← `identifier`（ViT embedding）
4. **下载路径有无意义 round-trip**：`fetch_image` 对网络图走「bytes → PIL 解码 → PIL 再编码 → 临时文件」，解码+编码只是为了满足后续 `pre_process` 的 Image 入参。

排除：多 API Server 实例；API Server 内协程/子进程并行（GIL + shm 都在）。

## 具体怎么改实现

三次渐进下沉，不是一次把整条链剪断。

### Phase 1：shm 零拷贝

先干掉序列化/反序列化，验证「搬运」是独立大头。

### Phase 2：`pre_process` + `post_process` 下沉 Worker

HF 重路径离开 API Server。此时 API Server 仍可能持有部分图像数据。

### Phase 3：完整下沉

图像读取 + HF preprocess 全部到 Worker。API Server 只留：

- APC 要的 `identifier` + offset/length
- scheduler 要的 encoder token 长度
- 入口校验和 prompt 规划
- 轻量 `mm_hash`
- tokenizer

Worker 侧真正做：读盘 → PIL 解码 → HF preprocess → post_process。

### 改 1：下载直写，解码推迟到 Worker

改前（API Server）：

```python
# bytes → PIL Image → bytes → file
response_bytes = await http_get(url)
image = PIL.Image.open(BytesIO(response_bytes))  # 解码
image.save(temp_file, format="JPEG")             # 再编码写盘
```

改后：

```python
# bytes → file
response_bytes = await http_get(url)
with open(temp_file, "wb") as f:
    f.write(response_bytes)
```

为什么有效：下沉后 API Server 不再跑 `pre_process`，不再需要 Image 对象。PIL 解码只在 Worker 真正进 HF 时发生一次。单张下载+保存 120ms → 80ms（约 40ms/张）。

### 改 2：`mm_hash` 轻量化

原 `get_mm_hashes()` 吃图像原始像素。下沉后 API Server 没有像素。

看输入签名：`model_id + mm_data_items + mm_uuid_items + hf_processor_mm_kwargs`，**不依赖 prompt / tokenization_kwargs**。于是：

| 条件 | 行为 |
|------|------|
| `uuid_item is not None` 且 kwargs 空 | 直接用 uuid 当 hash |
| uuid 有、kwargs 非空 | `hash(model_id, modality=uuid_item, **kwargs)` |
| uuid 空 | 仍走内容 hash（下沉后不应走到） |

hash 8ms → 1ms。hash 语义不变 → 三级缓存 identifier 不变。

### 改 3：Worker 按 ViT-DP 切图，绕开 GIL

每个 TP Worker 是独立进程。图片按 ViT-DP 均分：

- 各 rank 只读自己那份图
- 各自：本地读 → PIL → HF preprocess → post_process
- ViT 前重建本 rank 输入

40 图：API Server 串行 47ms → Worker 并行 30.7ms。

### 缓存一致性（实现时必须一起改）

| 层 | 位置 | 依赖 | 下沉后 |
|----|------|------|--------|
| mm_processor_cache | API Server | mm_hash | hash 语义不变 |
| APC identifier | Engine Core | identifier + mm_position | hash 不变则 identifier 不变 |
| encoder_cache | Worker | identifier | 未命中走 ViT，再和命中 embedding 拼接 |

## 为什么 TTFT 不是 161ms

预处理和 ViT 异步并行，重叠掉一部分。端到端记 **80ms+**。

## 学习要点

- 识别方法是 **API Server 模块级耗时表**，不是 NPU profiling。
- 实现核心不是「多线程预处理」，而是 **把重 CPU 从持 GIL 的 API Server 挪到多进程 Worker**，并 **删掉 bytes↔PIL 的假路径**。
- 下沉能做的前提：hash 可以不碰像素，且三级缓存都只认 hash/identifier。
