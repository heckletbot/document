# 总览：架构、部署与优化地图

## 调优结果

纯文本（Qwen3.5-122B-A10B-w8a8）：

| 指标 | 优化前（定长） | 优化后（不定长） | 提升 |
|------|---------------|-----------------|------|
| P50-TTFT | 2293 ms | 405 ms | 82.3%↓ |
| AVG TPOT | 21 ms | 14 ms | 33.3%↓ |
| 单卡 QPS | 0.022 | 0.57 | 2491%↑ |

多模态：

| 指标 | 优化前（定长） | 优化后（不定长） | 提升 |
|------|---------------|-----------------|------|
| P50-TTFT | 876 ms | 575 ms（含图片下载 ~30ms） | 34.4%↓ |
| P50-TPOT | 21 ms | 14.5 ms | 31.0%↓ |
| 单卡 QPS | < 0.1 | 0.425 | 325%↑ |

## 模型架构里必须记住的约束

混合架构：122.1B 总参 / 10B 激活；语言主干 48 层，按 `[L,L,L,F]×12`：每 3 层 Gated DeltaNet（线性注意力）接 1 层 Gated GQA（全注意力）；全层 MoE（256 专家 / top-8）；另有 1 层 MTP + 27 层 ViT。

和优化直接相关的点：

- **两类状态物理页必须对齐**：Full Attention 用分页 KV Cache，GDN/Mamba 用 SSM 递归状态（conv + recurrent）。APC 的 `block_size` 被这套对齐锁死 → 细粒度 APC。
- **GQA 极瘦**：32 Q 头 / 2 KV 头，`head_dim=256`。
- **线性注意力头**：`linear_num_key_heads=16`，`linear_num_value_heads=64`，`linear_*_head_dim=128`，Conv1D kernel=4。这些 kernel 就是 AscendC 重写对象。
- **MTP**：draft 头 1 层，投机解码。argmax 前移、MTP 入图、零气泡都挂在这里。
- **Gemma-style RMSNorm + MRoPE 3D**：融合算子（`split_rmsnorm_mrope_gate` 等）的来源。
- **原生 VL**：ViT 27 层 / hidden 1152，视觉 token 进 LLM prefill。

## 部署

权重：w8a8 + GDN 量化权重。PD 分离：

| 场景 | 形态 |
|------|------|
| 纯文本 | 2P(TP4) 1D(TP4DP4) |
| 多模态 | 6P(TP4) 1D(TP4DP4) |

第二阶段为了更高 QPS，SLO 从纯文本 TTFT 300ms 放到 500ms，部署从 TP8 改成 TP4。这个改动直接把 APC 粒度打坏，逼出细粒度 APC。

## 沿生命周期的三条瓶颈

```text
图片处理 ──► Prefill ──► KV 传输 / Decode
   │              │              │
   │              │              ├ 先 D 后 P：等上一轮、双份预处理、丢掉 P 首 token、TTFT 走 Decode
   │              │              ├ 逐层 sync 4ms
   │              │              ├ MTP 同步气泡 5ms+（0.19 零气泡未完全使能）
   │              │              └ 调度粒度过粗
   │              ├ 混合架构 kernel 吃不满 A3
   │              ├ 动态 shape 静态图无法复用
   │              └ MoE 通信
   ├ API Server 上 HF 预处理太重
   └ 视觉 token 太多
```

## 优化时间线（识别顺序比清单更重要）

| 阶段 | 识别出的问题 | 做了什么 |
|------|-------------|---------|
| 一 | 开箱不适配 | vllm-ascend 0.18 基础适配：GDN 量化、MoE 改 ReduceScatter、Turing AscendC、APC 适配 |
| 二 | 客户要更高 QPS；SLO 300→500ms | TP8→TP4；发现 APC 不适配 → 细粒度 APC；升 0.19，补零气泡使能；识别 ZMQ 复用、batch KV、fastokens |
| 三 | 重心转多模态 | 预处理三次下沉（80ms+）；数据分布变了再做视觉 token 稀疏化（40ms+） |
| 四 | 长期稳定 | 绑核、GIL 等（原文本节未给出） |

## 13 项在链路上的位置

```text
RouteServer:  ① 先 P 后 D 分层传输    ⑬ SLO 预测调度
API Server:   ②③ 首 token / 分词相关（fastokens 等）
MM Worker:    ④ 视觉 token 稀疏化     ⑤ 预处理下沉     ⑥ ViT 融合
Prefill:      ⑦ fastokens（编码）     ⑧ Inductor 入图  细粒度 APC
P/D 间:       ⑨ MoE AllGather+ReduceScatter
Decode:       ⑩ MTP 零气泡            ⑪ ArgMax 前移     ⑫ AscendC
```

编号按原文全景图。

五项关键优化单独成文，见 [关键优化.md](关键优化.md)。其余特性只记在本总览，不另开文档。

## 其他优化（只在总览）

| 特性 | 一句话 | 原文收益 |
|------|--------|---------|
| 图片预处理下沉 Worker | API Server 模块计时后，下载直写 + HF 预处理下到多进程 Worker，避开 GIL 和 shm | TTFT ↓ 80ms+ |
| 视觉 token 稀疏化 | API 读文件头虚算 `grid_thw`，Worker 按同一 `PHASE3_IMAGE_SCALE` resize | TTFT ↓ 40ms+ |
| AscendC 算子 | Triton 吃不满 A3，GDN/Conv1D/RMSNorm 等用 AscendC 重写、融合 | 整体 ↓ 120ms+ |
| ViT 算子融合 | RoPE+Attention、Add+LN、跨层 Add-Norm1 融合 | TTFT ↓ 10ms+ |
| MoE AllGather+ReduceScatter | TP4EP4、`top_k=8` 时替换 AlltoAll；仅 P 侧 DP=1 | TTFT ↓ 15ms |
| ArgMax 前移 | MTP draft 先局部 argmax 再 AllGather token_id | 计算量缩 1/TP，通信从 hidden 降到 id |
| fastokens | 编码换 Rust BPE，解码仍回 HF | TTFT ↓ 20ms |
| SLO 预测调度 | 预测器 + Slack 组 batch + RS 按完成度衰减负载 | vs 最小请求数：P95/P99 TTFT ↓ 16.4%/28.4% |
| ZMQ 控制面 | 热路径 log 降级 + socket 池化（原文后半截断） | 未给量化收益 |
| 绑核 / GIL | 第四阶段稳定性（原文本节未展开） | — |
