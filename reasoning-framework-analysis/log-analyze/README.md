# vLLM 日志分析（与融合算子分开）

这是**框架 / Host 路径**上的观察手册：读部署日志、引擎日志、benchmark、profiling 文本，把现象对上 Scheduler、KV、排队、预处理，再决定改哪段代码。

不是 `fusion/` 那套：这里不写 Ascend C、tiling、图 pass。Decode 气泡若最后证明是小 Kernel / GM 往返，再转到融合笔记。

接到总流程的哪一步：跑基线之后、判 Bound 之前——用关键字把问题按到请求路径的某一跳。

```mermaid
flowchart LR
  L[日志 / bench / profiling 文本] --> D[调优方向]
  D --> K[关键字切片]
  K --> C[vLLM 文件与函数]
  C --> H[先讲根因和改法]
  H -->|确认后再改代码| V[同一负载回归]
```

纪律：先结论和改点，**确认后再动代码**。收益经验阈值：吞吐或 TTFT/TPOT 相对基线超过约 5% 再视为有收益（有指定 SLA 则跟 SLA）。

---

## 文档

| 部分 | 要能说清什么 |
|------|----------------|
| [01-metrics-keywords.md](01-metrics-keywords.md) | 指标含义；方向 → 日志关键字 |
| [02-code-map.md](02-code-map.md) | 现象落到 vLLM 哪些文件 |
| [03-case-preprocess.md](03-case-preprocess.md) | 多模态 Host 预处理：日志分析怎么走到代码（已脱敏） |
| [04-speak.md](04-speak.md) | 口头：怎么从日志讲到改点 |

回到 [系统分析](../README.md)。融合验收仍走 [fusion/](../fusion/README.md)。
