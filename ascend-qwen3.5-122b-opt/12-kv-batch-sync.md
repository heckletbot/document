# KV Cache 批量传输（batch sync）

- 链路：Prefill 每层 forward 之后 → 把该层 KV 写到 Decode（与计算共享 NPU stream）
- 场景：2P1D，P: tp4dp1，D: tp4dp4，输入均长 8k，不定长并发
- 收益：TTFT 606ms → 531ms（↓75ms）
- 开关：`VLLM_CLOUD_KV_BATCH_SIZE`，默认 0=禁用；>0 表示一次 sync 覆盖的层数

## 问题怎么识别

分层传输策略：每层 forward 完立刻传 KV，和计算共用 NPU stream，用 `Event::record` → `Event::sync` 对齐。

profiling 结论：

- 每次 `aclrtSynchronizeEvent` 约 **4ms**
- 阻塞后续 kernel 下发
- **record→record 的 enqueue 间隙占 84.4%** —— 大头不是 DMA 拷贝，是同步把 launch 管道掐断

48 层满同步的话，光 sync 墙钟就不可接受。所以问题定义为 **同步粒度太细**，不是传输协议太慢。

## 具体怎么改实现

Prefill 侧加 KV buffer：

1. 每层 forward 后把 KV **写入 buffer 并立刻返回**，不在该层做 `Event::sync`
2. 每 N 层才 sync 一次，然后 `batch_transfer_sync_write` 一次送出 N 层 KV

```text
改前:  layer_i compute → record → sync(4ms) → transfer → layer_{i+1} ...
改后:  layer_i compute → write buffer → layer_{i+1} ...  (每 N 层才 sync+batch write)
```

N 由 `VLLM_CLOUD_KV_BATCH_SIZE` 控制。N 越大，sync 次数越少、下发越连；但 buffer 更大，D 侧看到 KV 的延迟也更批量化（分层 overlap 的窗口变粗）。原实验在不定长并发下 606→531ms。

## 学习要点

- 识别手段是 **Event sync 耗时 + enqueue 间隙占比（84.4%）**，直接指向 host/device 同步，而不是 RDMA 带宽。
- 实现是 **延迟同步 + 攒层传输**，计算 kernel 不再每层被 sync 卡住。
- 和 ZMQ 控制面（[13](13-zmq-control-plane.md)）分离：这里是数据面 stream。
