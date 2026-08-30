# Prefill Inductor 入图

- 链路：主模型 + MTP 的 Prefill 编译
- 收益：不定长并发下 TTFT ↓ 10ms+（短序列更明显）
- 开启：删 `--enforce-eager`；加 `--additional-config '{"ascend_turbo_graph_config":{"enabled":true}}'`；MTP 去掉 `speculative_config.enforce_eager: true`

## 问题怎么识别

Prefill 动态 shape 跨度极大（几十 token 到几十万）。先试了社区 **Piecewise ACLGraph**：

1. **分档盖不住业务**  
   bucket 有限。XHS 不定长跨度远超当时 piecewise 覆盖（已提到 8192，后续没再跟）。大量请求 miss 档位 → 编译产物复用不上，还多付编译税。实测无收益。

2. **ACLGraph 和通信重叠打架，直接把服务打崩**  
   - ACLGraph：所有算子同一 NPU stream 顺序执行  
   - `TASK_QUEUE_ENABLE=2`（二级流水）：HCCL 与计算不同 stream overlap  
   - 结果：stream 语义冲突、域错误，Prefill 侧崩溃  

所以要换一条：**能吃动态 shape，又不和多 stream 通信重叠冲突** 的入图。

## 具体怎么改实现

Prefill 全面改 `torch.compile` + Inductor（`torch_npu._inductor`），端到端：Dynamo 捕获 → VllmBackend → Inductor → NPU 算子。替代 ACLGraph。

三阶段生命周期（原文图）：捕获 / 编译 / 运行期缓存复用。

关键实现选择：

| 点 | 做法 | 为什么 |
|----|------|--------|
| 管线复用 | 复用 GPU 侧 vLLM 的 Dynamo → VllmBackend → Inductor，只换 NPU backend | 少写一套编译器 |
| `fullgraph=True` | 禁止 graph break | 整条 forward 都在图里，融合才吃得满 |
| Fix Functionalization | Post-Grad 对 `atb._npu_reshape_and_cache`、`atb._npu_rotary_embedding` 等 `auto_functionalized` 算子反函数化，还原原地写 | 否则编译后 KV 写入/RoPE 语义和 eager 不一致 |
| 禁用 Triton | `TORCHINDUCTOR_DISABLE_TRITON=1`，aten lowering 走 AscendKernel，调 NPU 原生算子 | 不要再生成 Triton kernel |
| Buffer 复用 | `allow_buffer_reuse=True`；**关掉** `reorder_for_locality` / `reorder_for_peak_memory` | 静态分析复用 buffer；NPU 上那些重排不兼容 |
| 编译缓存 | `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` | 重启不重编译 |

收益机制：算子下发次数下降、融合减 HBM 压力、静态编译 + buffer 复用。昇腾 Inductor 还能对矩阵计算单元自动调优（原文称为昇腾特有能力），对动态 shape Prefill 通用。

## 学习要点

- 识别有两层：业务上 piecewise **无收益**；工程上 ACLGraph × 二级流水 **不可用**。后者比「慢」更硬。
- 实现不是「打开 compile 开关」就完，必须处理 **NPU 原地算子的反函数化** 和 **关掉 Triton / 内存重排**。
- MTP Prefill 要单独去掉 `enforce_eager`，否则主模型入图、draft 仍 eager。
