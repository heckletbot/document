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

## 符号维是怎么做出来的（为什么不必焊死 shape）

「编译必须固定 shape」对的是 **ACLGraph / CUDA Graph**：回放图把 launch 配置、tensor 地址、grid 都录死，长度一变就不能 replay。Inductor 不是回放器，它是 **codegen**：生成「长度当参数」的 kernel，所以编译期不必知道 800 还是 1024。

实现上分三步，符号是在捕获时种下的，不是 Inductor 事后猜的。

**1. 捕获时用 FakeTensor + SymInt，不拿真实数往前推**

Dynamo 不跑真 NPU。每个中间结果是 FakeTensor：dtype / rank / device 是真的，尺寸可以是符号 `s0`（`torch.SymInt`），派生长度是表达式（`s0 * 2`、`s0 // 32`）。第一次请求的 1024 只当 **hint**（给 Inductor 看个例子、估个 grid），不写进守卫。

维变成符号的来源：

- 显式：`torch._dynamo.mark_dynamic(x, 0)`，或 `torch.compile(dynamic=True)`
- 自动：第一次常按静态编；第二次换了长度，守卫失败，Dynamo 把这维升成 `s0` 再编一次（automatic dynamic）

图结构（有哪些算子、谁连谁、几维、dtype）是编译期固定的；**各维有多长**是运行期才填的。

**2. Inductor 对着符号表达式 lowering，kernel 带运行时长度**

IR 里 size 是 `sympy` 表达式，不是常量。生成出来的 kernel 类似：

```text
kernel(x, y, s0):          # s0 是入参
    for i in 0..s0:        # 循环上界运行时才知道
        y[i] = f(x[i])
grid = cdiv(s0, BLOCK)     # launch 配置也按本次 s0 算
```

融合、buffer 复用看的是「这两段同长度 / 生命周期不重叠」，用的是符号等式（`s0 == s0`），不依赖具体是 800。昇腾侧关掉 Triton 之后，同样把 `s0` 传给 AscendKernel / NPU 原生算子——这些 API 本来就接受运行时 dim。

**3. 守卫卡住的是结构，不是某个长度**

还要固定、会触发重编的：rank、dtype、device、stride 模式、以及「`if seq > 512:`」这种按具体值走的控制流。  
不必固定的：token 数、batch 里实际 token 总和。

所以不是「编译突然不需要 shape 了」，而是 **编译需要的是 shape 的结构，不是 shape 的数值**。ACLGraph 把数值也录进去了；Inductor 只把结构编进去，数值当参数。

仍会被迫焊死的情况：某个 NPU 算子没有动态实现、或 Inductor 为了对齐假设了 `s0 % 8 == 0`。之后来了不满足的长度，守卫 miss，再编一份。动态不是无限万能，是把「每个长度一份图」收成「同一结构一份图」。

## 是不是必须算子先有「动态实现」才能用 Inductor

不是。打开 Inductor **不要求** GPU/NPU 上每个算子都另做一套动态版。分两类：

**Inductor 自己生成的 kernel**（逐点、融合、不少 reduction）  
动态能力在 codegen 里：`s0` 当参数写进 kernel。不存在「先向厂家要一份动态实现」这一步。GPU 上是 Triton/C++，这里关掉 Triton 后走 AscendKernel，同样按运行时 dim 生成。

**Inductor 调不着、只能调用现成库算子的**（GEMM、FlashAttn、HCCL、`reshape_and_cache`、RoPE…）  
这些本来就是「调用时传入 m/n/k」的 API，大多数 **eager 就接受运行时长度**，并没有单独的 static/dynamic 两个实现。Inductor 只是在图里把这次的 `s0` 传进去。

会卡住的是第三种：算子内部自己是 **固定 shape 回放**（里面套了 ACLGraph / 只编过一个长度），或符号维 lowering 失败。这时 Inductor 只能：

- 把这维焊死（每个长度一份守卫/一份编译），或
- graph break，这段回 eager

所以：

- **能用 Inductor**：不依赖「全家都有动态实现」
- **一整张 Prefill 图都吃动态 shape**：热路径上的库算子必须能按运行时 dim 工作（或 Inductor 能自己生成）。做不到的那截就会特化或掉出图

这篇优化关掉 Triton、走 NPU 原生算子，前提是这些 ACLNN/ATB 接口本身吃运行时 dim。`reshape_and_cache` / RoPE 要改的是原地写语义（反函数化），不是「缺动态版」。

## 融合一次下发，和动态 shape 是两件事

容易把 Inductor 收成一句话：「融合了，中间不必带动态 shape，输入传一次长度，launch 一次就下发多个算子。」前半（融合少 launch）对，后半（所以才能动态）不对。

Inductor 做两件**正交**的事：

| | 融合 / 少下发 | 动态 shape |
|--|----------------|------------|
| 何时发生 | **编译期**把 add+mul+rms 收成一个 kernel | **编译期**把长度写成 kernel 参数 `s0` |
| 运行期（launch） | 一次 launch 跑完原来好几个小算子 | 把本次 `s0=800` 填进去 |
| 没有它时 | 仍可动态：每个小算子各自带 `s0` launch | 仍可融合：焊死 1024 也能融，只是换长度要重编 |

eager：`rms` launch 一次、`mul` 一次、`add` 一次，中间结果写 HBM。  
融合后：一个 kernel 里在寄存器里做完，**少的是 launch 次数和 HBM 往返**，不是「中间不再有长度」。循环仍是 `for i in 0..s0`，中间值跟着同一个 `s0` 走，只是不再变成独立的全局 tensor。

没融进去的大算子（matmul、attention）照样各自 launch，它们的输入/输出 tensor 该有动态 shape 还是有，只是 Inductor 做了 buffer 复用。

动态 shape 能用，是因为 **codegen 把长度当参数**；融合只是顺带让一次 launch 多干点活。ACLGraph 也能「一次 replay 很多算子」，但长度焊死——说明少下发 ≠ 能动态。

时间线也要分开：编译（融合 + 生成带 `s0` 的 kernel）在捕获之后做一次；之后每个请求只是 launch，带上这次的长度。不是每个请求 launch 时现场再融合。

## 关掉 Triton、fallback 到 NPU 原算子，这些优化还在吗

还在的是**图级**优化；变薄的是「很多小算子收成一个自研 kernel」那一档。

Inductor 先在 FX / Inductor IR 上做一遍，再决定每个节点 lowering 到哪。不写 Triton、改调 ACLNN/ATB，只换最后一截，前面那遍还在。

| 优化 | fallback 到 NPU 原算子之后 |
|------|---------------------------|
| 消除冗余（DCE / CSE / 常量折叠 / 代数简化） | **还在**。编译期从图里删掉死代码、合并重复计算，和是不是 Triton 无关 |
| Buffer reuse（存活区间、中间 tensor 复用同一块显存） | **还在**，这篇的 `allow_buffer_reuse=True` 就是干这个。管不到算子内部自己申请的 workspace |
| Host 下发 | **还在一点**：编好的图不再走 Python 逐算子调度，少一层解释器税。**薄很多**：每个还在的库算子仍要单独 launch，没有「10 个逐点合成 1 个 Triton kernel」那种数量级下降 |
| 逐点融合进寄存器、中间不落 HBM | **基本没了**，除非 NPU 侧正好有对应的融合算子（AscendKernel / 融合 ACLNN）可被 lowering 选中 |

所以：能在 NPU 上把 Inductor 跑起来，不等于还拥有 GPU+Triton 那套融合收益。图还是图，冗余能消、buffer 能复用、动态 `s0` 还能传；host 大头取决于 lowering 之后还剩多少次库算子 launch。

这篇 Prefill 优化不是纯 1:1 fallback。`torch_npu._inductor` 走 AscendKernel，仍可能把一部分节点降到 NPU 融合接口，再加 buffer 复用。原文说的「下发次数下降、融合减 HBM」指的是这一层，不是「每个 aten 都原样调一次 ACLNN」。若真做成纯 fallback，TTFT 那 10ms+ 里融合/下发的部分会小很多，剩下主要是去 Python 和消冗余。

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
- 固定 shape 是 ACLGraph 回放的约束。Inductor 是 codegen：结构固定、长度当 kernel 参数。FakeTensor + SymInt 种符号，lowering 时不把 1024 写进指令。
- 用 Inductor 不要求每个算子另做动态版。自己生成的 kernel 自带 `s0`；库算子大多本来就吃运行时 dim。只有内部焊死 shape 或 lowering 失败的，才会特化 / graph break。
- 融合（一次 launch 多个小算子）和动态 shape（长度当参数）是两件事。中间结果进寄存器，不是「中间不需要 shape」；少下发也不等于能动态（ACLGraph 就是反例）。
- 关掉 Triton、改调 NPU 原算子：DCE/CSE 和 buffer reuse 还在；逐点融合成一个 kernel 基本没了，host 下发收益变薄。这篇实际走的是 AscendKernel，不是纯 1:1 fallback。
- 识别有两层：业务上 piecewise **无收益**；工程上 ACLGraph × 二级流水 **不可用**。后者比「慢」更硬。
- 实现不是「打开 compile 开关」就完，必须处理 **NPU 原地算子的反函数化** 和 **关掉 Triton / 内存重排**。
- MTP Prefill 要单独去掉 `enforce_eager`，否则主模型入图、draft 仍 eager。
