# Prefill Inductor 入图

- 链路：主模型 + MTP 的 Prefill 编译
- 收益：不定长并发下 TTFT ↓ 10ms+（短序列更明显）
- 开启：删 `--enforce-eager`；加 `--additional-config '{"ascend_turbo_graph_config":{"enabled":true}}'`；MTP 去掉 `speculative_config.enforce_eager: true`

## 什么是 Inductor

Inductor 是 PyTorch `torch.compile` 默认的**图编译后端**，不是一种图格式，也不是 ACLGraph 那种「把已经下发的算子录下来回放」。

`torch.compile` 三截：

```text
Python forward
    → Dynamo：把 eager 代码收成一张 FX 图（带 shape/类型守卫）
    → Inductor：在这张图上做融合、调度、内存规划，再 lowering 成可执行 kernel
    → 运行：走编译产物；命中缓存就不再编
```

eager 是一算子一 launch。Inductor 把一段 forward 当成一张图优化：相邻逐点算子合成一个 kernel、少几次 HBM 往返、少几次 host 下发。GPU 上它常生成 Triton kernel；这条优化用的是昇腾后端 `torch_npu._inductor`，关掉 Triton，lowering 到 NPU 原生算子（AscendKernel）。

和这篇里先试过的 **ACLGraph** 对比：

| | ACLGraph（类 CUDA Graph） | Inductor |
|--|---------------------------|----------|
| 干什么 | 录一段已有算子序列，之后 replay | 真正编译图：融合、codegen、内存规划 |
| 动态 shape | 基本靠分档（piecewise / bucket） | 符号 shape，同一套产物能吃一档跨度 |
| 和多 stream | 整段绑在同一 stream 顺序执行 | 不强制单 stream，能和 HCCL overlap 共存 |
| 「入图」指什么 | 进 replay 图 | 进 `torch.compile` 图 |

所以「Prefill Inductor 入图」= 把 Prefill 的 forward 交给 Dynamo 收图、Inductor 编译，而不是 eager 逐算子下发，也不是 ACLGraph 回放。

## 动态 shape 的捕获是什么

「捕获」是 **Dynamo 的事**，发生在 Inductor 编译之前。第一次（或守卫失败后）跑 Prefill forward 时，Dynamo 跟着 Python 走一遍，把算子收成一张 FX 图，并记下这张图什么时候还能用——这套记录叫 **guard（守卫）**。

图上每个 tensor 都有 shape。捕获时有两条路：

**静态捕获（默认容易走成这样）**  
第一次请求是 1024 token，图里把 `seq=1024` 焊死，守卫写成「长度必须还是 1024」。下一条 800 token 进来，守卫 miss → 重新捕获、重新编译。Prefill 从几十到几十万，等于几乎一请求一编，编译税把融合收益吃光。

**动态 shape 捕获**  
预先（或由 Dynamo 自动）把**会变的维**标成符号，比如 `seq = s0`，而不是 1024。图里写的是「长度为 s0 的 matmul / RMSNorm / RoPE」，守卫只卡 rank、dtype、device，以及「这维是动态的」。800 和 1024 共用同一张图、同一份 Inductor 产物；运行时把 `s0=800` 填进去。

```text
捕获（Dynamo）
    静态:  input[1024, H] → 图焊死 1024，守卫: seq==1024
    动态:  input[s0, H]   → 图留下符号 s0，守卫: rank/dtype 对，s0 可变

编译（Inductor）
    对着这张带 s0 的图做融合 / lowering，生成「按运行时长度工作」的 kernel

运行
    命中缓存：代入本次的 s0，直接跑
    未命中：再走一遍捕获+编译（结构变了，或之前按静态编过）
```

这就是原文说的三阶段：**捕获 / 编译 / 运行期缓存复用**。动态指的是捕获阶段留下符号维，不是 Inductor 每次现场重编。

和 ACLGraph 分档的差别：piecewise 是「准备好几张固定长度的回放图（1、2、4、…、8192），请求 pad 进某个桶」。桶盖不住就 miss，还多付编译税。动态 shape 捕获是「一张带 s0 的编译图，很多长度都能套」。业务不定长跨度远超当时 8192 档，所以 piecewise 无收益，才换 Inductor 这条。

代价：符号维上有些优化更保守（不能按某个固定长度死展开）。`fullgraph=True` 是为了整条 forward 都留在同一张带 s0 的图里，融合才吃得满；中途 graph break 会把后面踢回 eager，动态捕获也救不了那截。

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

- Inductor 是 `torch.compile` 的编译器后端：收图之后融合/lowering，不是 ACLGraph 那种录制回放。
- 「动态 shape 捕获」= Dynamo 把会变的维收成符号 `s0`，不是焊死本次长度。编译一次，运行时代入不同 seq。
- 识别有两层：业务上 piecewise **无收益**；工程上 ACLGraph × 二级流水 **不可用**。后者比「慢」更硬。
- 实现不是「打开 compile 开关」就完，必须处理 **NPU 原地算子的反函数化** 和 **关掉 Triton / 内存重排**。
- MTP Prefill 要单独去掉 `enforce_eager`，否则主模型入图、draft 仍 eager。
