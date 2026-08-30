# 图片 token 稀疏化

- 链路：API Server 规划 `grid_thw` + Worker 实际 resize，发生在 ViT 之前
- 前置：预处理已下沉（见 [01](01-image-preprocess-offload.md)）
- 场景：同上，视觉 token 约 28.5%~30%+
- 收益：TTFT 640→600ms（↓约 40ms）；`scale=0.85` 时视觉 token 约降 28%，精度按原文「无损」

## 问题怎么识别

预处理下沉后，API Server 侧已经不是主因。剩下和 **视觉 token 数量线性相关** 的两块：

- ViT 推理本身约 45ms
- LLM prefill 计算量随总 token 涨

客户分布：11K 总输入里视觉 token 占 30%+，13.5 图/请求。token 数同时打 ViT 和 LLM prefill，所以要从 **源头减 token**，而不是再抠预处理毫秒。

三个实现约束是定位「不能在 API Server 再解码像素」之后定下来的：

1. 缩太狠会伤视觉信息，要有可接受的 scale。
2. API Server 已轻量化，若为缩放重新解码像素，等于把 Phase 3 设计打回去。规划必须 **零像素 I/O**。
3. API 按缩放后尺寸算 `grid_thw` / token 数，Worker 必须用 **同一个 scale** 真缩放，否则 token 数对不上，encoder cache 维度错、推理错。

## 具体怎么改实现

不是改 ViT 结构，而是 **API 用文件头虚算尺寸，Worker 在已有 PIL 解码后多一次 resize**。

### API 侧：`_phase3_lite_preprocess` 零像素规划

1. `_get_image_size_fast()` 读 JPEG/PNG **文件头 metadata** 拿原始 `w/h`，不解码像素。
2. `w' = w × PHASE3_IMAGE_SCALE`，`h'` 同理。
3. 用 `(w', h')` 调 `_get_vision_info()`，得到的 `grid_thw` 和 token 数按缩放后尺寸走。

API Server 只做算术，没有像素 I/O。

### Worker 侧：两条加载路径都先 `_resize_pil`

- direct encoder：`_load_local_images` 读出 PIL 后 `_resize_pil(img)`，再进 HF processor
- slow path：`_phase3v3_slow_path` 同样先缩放

Worker 本来就要 PIL 解码，双线性 resize 相对可忽略，不构成新瓶颈。

### 一致性

两侧读同一个环境变量 `PHASE3_IMAGE_SCALE`：

- API 规划的 token 数 = Worker 送进 ViT 的 token 数
- scheduler 给的 encoder cache 空间 = 实际 vision embedding 维

### 数量关系（scale=0.85）

视觉 token 近似随面积走，不是随边长：

```text
token' ≈ token × scale²
5760 × 0.85² ≈ 4147   → 约 -28%
ViT 计算量 ≈ 0.72×
LLM prefill 随总 token，约 -10%
```

## 学习要点

- 识别时机：下沉做完后，用 **ViT 45ms + 视觉 token 占比** 判断下一刀该砍 token 数，而不是继续砍 API Server。
- 实现关键是 **规划侧不碰像素、执行侧用同一 scale**，不是在模型里做 token pruning。
- 面积平方关系：scale 0.85 看起来只缩 15% 边长，token 已经少近三成。
