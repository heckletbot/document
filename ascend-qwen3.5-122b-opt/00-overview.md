# 总览：场景、架构、部署与优化地图

先把约束钉死，后面每条优化都挂在这条链路上。

## 场景约束（为什么这些点会被逼出来）

XHS 智搜是线上实时服务，纯文本和多模态（图文）同时跑。三个硬约束叠在一起：

| 约束 | 数据 | 对系统的含义 |
|------|------|-------------|
| 长输入、短输出 | 纯文本均长 7.3k（20~40k+），输出 650~720 | Prefill 计算和 Decode 吞吐必须同时高 |
| 多模态占比高 | 文本:图片 = 7:3；均 13~14 张/请求（最多 20）；单张约 200 token | ViT + 视觉 token 直接打 TTFT |
| 严格 SLO | 纯文本 P50-TTFT ≤ 500ms、AVG TPOT ≤ 15ms；多模态 P50-TTFT ≤ 600ms（含下载 ~30ms）、P50-TPOT ≤ 15ms | 调度不能只看吞吐，尾延迟也要管 |

栈：开源 vLLM + `vllm_ascend` + ModelArts `ascend_vllm`。特性落到 ModelArts 6.5.923 / 6.5.928 / 6.5.930。

开箱极差，社区适配刚起步，早期适配工作原文未完整记录。基线因部署方案多次改过（TP8→TP4、PD 配比），绝对值参考意义有限，以单项收益为准。

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
   │              │              ├ 先 D 后 P 空泡
   │              │              ├ 逐层 sync 4ms
   │              │              ├ MTP 气泡（待补）
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
Decode:       ⑩ MTP 零气泡（待补）    ⑪ ArgMax 前移     ⑫ AscendC
```

编号按原文全景图。学习时按生命周期读：01→02→03/04/05→07/08/09→11/12/13→06/10。
