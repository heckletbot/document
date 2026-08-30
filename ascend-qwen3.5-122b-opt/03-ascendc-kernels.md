# AscendC 算子深度优化

- 链路：Prefill / Decode 的 GDN 与门控全注意力 kernel（以及部分 RMSNorm / MoE / Matmul）
- 收益：原文「整体 ↓ 120ms+」，是多项 kernel 叠加，不是单算子

## 问题怎么识别

Qwen3.5 混合架构的核心计算（Gated DeltaNet 的 chunk/recurrent 路径、causal Conv1D、门控全注意力）走 **通用 Triton 路径** 时，吃不满昇腾 A3：

- 分核、CV 流水、核间同步、transpose、小算子 launch 都按 GPU/Triton 习惯来
- 这些 kernel 在 48 层里反复出现（36 层 GDN + 12 层全注意力），单 kernel 毫秒级浪费会乘到端到端

识别方式是 **算子级耗时表**（不是请求级 SLO）：点名 `cloud_chunk_scaled_dot_kkt`、`cloud_solve_tril`、`causal_conv1d` 等，对每个 kernel 标优化类型和收益。`causal_conv1d` 单点 45ms，是表里最大的 Prefill 项。

关联问题：RMSNorm + Bias + DynamicQuant 三步独立 launch，中间 HBM 读写和同步多余 → 融成 `cloud_add_rms_norm_bias_dynamic_quant`。

## 具体怎么改实现

手段固定四类，按算子选用，不是换算法：

1. **AscendC 重写** Triton kernel
2. **分核 / tiling**
3. **流水重排、减同步**
4. **融合**：干掉独立 transpose、Index_Add、小 prepare 算子

### Prefill（接入 AscendC）

| 算子 | 改法 | 原文收益 |
|------|------|---------|
| `cloud_chunk_scaled_dot_kkt` | CV 流水、同步 | 1.87ms |
| `cloud_solve_tril` | 计算逻辑、多核、预取 | 10ms |
| `cloud_recompute_wu` | 多核、流水、低阶实现 | 15ms |
| `chunk_gated_delta_rule_fwd_h` | 流水重排、去多余同步、transpose 融合消除 | 16ms |
| `chunk_fwd_o` | 分核、流水、计算 | 10.8ms |
| GDN 去 transpose + `prepare_chunk_offsets` | transpose 融进主算子，去掉模型侧 transpose | 8ms |
| `split_rmsnorm_mrope_gate` | RMSNorm+MRoPE+小算子融合，去 Index_Add，分核 | 6ms |
| `causal_conv1d` | 同步、scalar、分核 | 45ms |
| `cloud_rmsnorm_silu` | 融合、分核、流水 | 3.8ms |
| `addRmsnormBias` + DynamicQuant | 三步融成 `cloud_add_rms_norm_bias_dynamic_quant`，替换原 RMSNorm 并预量化 | 1.5ms |

### Decode

| 算子 | 改法 | 原文收益 |
|------|------|---------|
| `cloud_swi_glu_dynamic_quant` | SwiGlu+DynamicQuant 融合 | 4ms |
| `cloud_recurrent_gated_delta_rule` | 融 transpose、流水、scalar | 3.8ms |
| `cloud_rmsnorm_silu` | 同 Prefill 思路 | 0.3ms |
| `split_rmsnorm_mrope_gate` | 同 Prefill | 0.1ms |
| FIA（flashdecode） | 切分、分核 | 0.8ms |
| 整网 Matmul 知识库 | 计算模板、tiling | 0.4ms |
| GDN `conv1d_update` 入图兼容 | 计算逻辑、tiling | 0.6ms |
| MoE DispatchFFNCombine | Gmm+Routing 通算融合（投机路径） | 0.2ms |

`addRmsnormBias + DynamicQuant`：独立三 kernel 变成单一 `cloud_add_rms_norm_bias_dynamic_quant`，减少中间 tensor 落地。当前方案里适配开启。

## 学习要点

- 识别对象是 **混合架构反复出现的 GDN/Conv1D/RMSNorm 族**，用 kernel 表而不是「NPU 利用率低」一句。
- 实现不是调 vLLM 调度参数，是 **AscendC 替换 Triton + 融合掉模型里的 transpose/小算子**。
- Prefill 的 `causal_conv1d`（45ms）和 GDN chunk 族是大头；Decode 单步小，但每 token 都会跑。
