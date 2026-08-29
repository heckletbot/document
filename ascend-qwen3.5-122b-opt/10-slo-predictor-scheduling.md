# SLO 预测调度

- 链路：P 实例内部组 batch + RouteServer 选实例
- 背景：第二阶段客户要更高单卡 QPS；纯文本 TTFT SLO 300ms→500ms，指标从 AVG 改成 P50；多模态还有图片下载抖动
- 收益：
  - RS 细粒度负载 vs 最小请求数：P95/P99 TTFT ↓ 16.4% / 28.4%
  - vs least-total-load：P95/P99 ↓ 8.9% / 11.5%
  - P 侧 SLO 优先级组 batch：1P1D 下 avg TTFT 持平，P90 ↓ 8.9%

## 问题怎么识别

旧调度的信息粒度是 **整请求**：

- 跑到 99% 的长请求和刚入队的长请求，在 RS 眼里一样重
- 短请求被这些「看起来很忙、其实快结束」的实例挡住
- 负载只在到达/完成两个时点更新，中间是盲区

多模态把这个问题放大：同样 prompt 长度，图数量和下载速度不同，实际 TTFT 差一截。只看 token 数或请求数，预测不准。

SLO 从 AVG 改 P50/P90/P95 之后，均值好看不够，必须管尾部，逼出「预测执行时间 + 按 Slack 组 batch + RS 按完成度衰减负载」。

实验室侧用马尔可夫模拟先验证策略，再上线。

## 具体怎么改实现

三块解耦：预测器、P 侧组 batch、RS 侧负载。

### 1. 独立执行时间预测器

挂在 P 实例，和 Engine 一起拉起，可在线训练。接口：

```python
class PredictorInterface(ABC):
    def __init__(args):                 # 基准数据 / 初始模型
    def predict(args) -> float:         # 给定 batch 配置 → TTFT（秒）
    def update(args) -> None:           # 用观测更新
```

唯一实例在 `EngineCore.__init__()` 创建。两个调用点：

- 请求入队：乐观预测
- 组 batch：对每个候选请求预测「加进当前 batch 后的执行时间」

特征：KV Cache 总命中率、平均 Prompt 长度、Batch Size。

训练数据用 **分桶 FIFO**：按 KV 命中率 × 平均 Prompt 长度分桶，桶内固定长度队列，避免新分布把旧桶冲掉。异步训练，固定频率更新，不堵预测。

零数据可启动，靠在线反馈校准 → 换硬件/模型也能迁。

### 2. SLO 感知组 batch（P 侧）

1. 入队时用预测器估 TTFT，算 Slack（距截止还有多久）
2. 以 Slack 为优先级做模拟组 batch
3. 对组成的 batch 再预测
4. SLO 不破则 FIFO 收进；预测会破 SLO 的请求降权，放到不影响别人的后续 batch

算法骨架：

1. 最高优先级请求 H 当 batch 基底
2. 用 H 的截止时间算剩余可用时间
3. 逐个试候选：预测加入后延迟；SLO 不违约且 token 不超预算才收，否则跳过

### 3. RS 细粒度负载

P 的 Scheduler 组完 batch 后，回调 Proxy/RS，上报：

- 批次开始时间戳
- 请求 ID 列表
- 预测完成时间

RS 选实例前，用「当前时间 vs 预估完成」算每个在途请求的完成度，修正实例真实 token 负载，再走最低负载。

对实例 `i` 上请求 `r`：

```text
waiting_r  = orig_r * (R_r / P_r)     # 还没进 batch 的份额
running_r  = orig_r * (S_r / P_r)     # 已进 batch 的份额
progress_r = min(E_r / T_r, 1 - ρ)    # ρ=0.05，禁止看到 100% 空闲
actual_r   = waiting_r + running_r * (1 - progress_r)
```

`P_r, R_r, S_r, T_r` 来自 batch 事件（总数 / 剩余 / 已进 batch / 预计耗时），`E_r` 是距首次上报的墙钟。running 随时间衰减，waiting 不衰减；保底 5% 防止误判空闲。这样即将跑完的长请求会把实例「让出来」给新请求。

## 学习要点

- 识别：负载单位从「请求个数 / 静态 token」换成「**还剩多少工作**」；多模态下载抖动说明必须用实测反馈，不能只用 prompt 长度。
- 实现分两层：P 内用预测 **拒收会打爆 SLO 的组 batch**；RS 用 batch 事件 **让负载随时间单调下降**。
- 预测器和调度算法解耦，调度可以换到别的框架上。
