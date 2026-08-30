# ViT 算子融合

- 链路：多模态 Worker 上的 27 层 ViT，和 [03](03-ascendc-kernels.md) 的 LLM kernel 独立
- 收益：TTFT ↓ 10ms+

## 问题怎么识别

Qwen3 ViT 在 NPU 上是 **大量细粒度算子逐个 launch**：

- 调度开销大 → NPU 利用率低
- 算子间多余 HBM 读写和核间同步
- 多模态里 ViT 已经是 TTFT 可见瓶颈（下沉后 ViT 仍约 45ms 量级）

定位在「encoder 内部执行效率」，不是图片下载、也不是 LLM prefill。

## 具体怎么改实现

三层融合，减少中间搬运和同步，不改 ViT 结构宽度/深度：

1. **RoPE + Attention → NPU 原生融合算子**  
   去掉 RoPE 写出、Attention 再读入这一段。

2. **层内 Add + LayerNorm → `npu_add_layer_norm`**  
   残差加完立刻 Norm，少一次核间同步。

3. **跨层：前一层 Add 与当前层 Norm1 融合**  
   把层边界上的独立 Add/Norm 收掉。

效果路径：中间激活少落地 → 带宽压力下降 → NPU 更连得上 → ViT 墙钟下降 → TTFT 约 10ms。

## 学习要点

- 和 LLM AscendC 是两条线：一个打 GDN/GQA，一个打 27 层 ViT 的 launch 粒度。
- 识别信号是 **小算子密度 + 利用率**，改法是融合，不是稀疏化（稀疏化见 [02](02-vision-token-sparsification.md)）。
