# 指标、方向、日志关键字

先定「这次看哪类病」，再按表去日志里切片。不要把 decode 气泡的关键字拿去扫 TTFT。

---

## 指标（和总分析同一套，这里补日志侧）

| 指标 | 含义 | 日志/bench 上常往哪想 |
|------|------|------------------------|
| TTFT | 首 token | prefill、调度 waiting、**Host 预处理**（多模态尤其） |
| TPOT | 每输出 token | decode batch、schedule、KV 是否把 batch 打散 |
| Throughput | QPS 或 tok/s | batch 组不满、KV 不够、preempt |
| E2E | 端到端 | 上面几段的加总，先拆不要先揉 |
| Decode Bubble | decode 段设备空转 | idle/gap、组 batch 失败、KV preempt 后重算 |

---

## 方向 → 关键字

| 调优方向 | 日志里先搜 | 对上总分析的哪一跳 |
|----------|------------|--------------------|
| decode 气泡 | `profiling`、`bubble`、`idle`、`gap`；decode 时间分布 | Scheduling Bound；P/D 互抢 |
| TTFT 过高 | `prefill`、`scheduler`、`waiting`；首 token | 排队 vs Prefill vs **进模型前的 Host** |
| throughput 低 | `batch_size`、`kv_cache`、`preempted`、`block_manager` | Framework Bound |
| KV 问题 | `cache_hit`、`evict`、`swap`、`preempted` | 显存/缓存策略 |

三类来源分开读，不要混成一段 stdout：

1. **Profiling 文本**：decode 空闲段、P/D 时间占比。  
2. **引擎日志**：命中率、preempt 频率、batch 曲线。  
3. **Benchmark**：吞吐和时延分位；对比两轮时只看同一负载。文件命名和「升/降算收益」见 [05-benchmark-compare.md](05-benchmark-compare.md)。

TTFT 高时，多模态请求要先问：时间是耗在 **下图/解码图** 还是 tokenizer 之后的 Prefill。关键字表里没有 `PIL`/`download`/`media` 时，自己补 Host 预处理这一跳。

下一篇：[代码落点](02-code-map.md)。
