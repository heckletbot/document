# 案例（脱敏）：多模态图片下载拖高 TTFT

用来练习「方向是 Host 预处理时，日志分析怎么走到代码」。数字和模型名已去掉；只保留判断结构和改法类型。

---

## 场景骨架（可复述，不要背配置）

- 多模态 image-text；PD 分离 + 路由。  
- 压测高并发，SLA 更痛的是 **TTFT / E2E**，不是单卡 Kernel。  
- 方向明确：图片进入 Prefill 之前的下载与解码。

这和融合算子无关：NPU 还没算，时间已经花在 Host 上。总路径里对应「多模态预处理」，Bound 更像 Framework / Host 准备，不是 Compute Bound。

---

## 基线怎么用

先部署、健康检查、**warm up**，再跑同一套 conc / 请求数。原始输出进 `tune_reports/round_{N}_result.txt`（基线另存 `baseline.txt`）。记下 QPS、输出 tok/s、TTFT（均值和尾部）、E2E。后面只对比这一张表。

---

## 三个典型根因（分析阶段）

看预处理代码与日志对齐后，常见是这三类（一次改一类更好）：

1. **解码线程池过小**：并发请求远大于 PIL/解码 worker，图在 CPU 队列里排，TTFT 里全是 waiting。  
2. **死代码 IO**：上游已经带了 `original_bytes`，fallback 仍 `open()` 读盘；或 fallback 永不触发，只是拖复杂。  
3. **解析串行**：`resolve_items` 一张一张处理，高并发时 Host 做不成流水。

对应修改类型（示例逻辑，不是某仓库补丁原文）：

- 信任上游 bytes，去掉必不走的二次读文件。  
- 用环境变量把 media loading 线程数提到与并发匹配的量级（过大也会抢 CPU，要回归）。  
- 能并发的解析再下一轮做，避免和线程池改动搅在一轮里说不清。

---

## 回归时怎么读结果

这一类优化 **TTFT / E2E 应明显下降**；QPS、tok/s 小幅抖动（几个百分点）往往是噪声，不要因为吞吐没涨就回退。吞吐没动、首 token 快了，说明瓶颈本来就不在 decode Kernel。

PD 分离：只改 Prefill 预处理 → 只清 Prefill 卡和端口，Decode 可不动。RouteServer 在 P 重启后若 500，把 RS 再拉起来。

下一篇：[口头讲稿](04-speak.md)。
