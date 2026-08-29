# ZMQ 控制面优化（原文截断，待补）

- 链路：Mooncake 控制面（不走 KV 数据）
- 状态：**原文在实现中段截断**，收益数字未给出

## 问题怎么识别

Mooncake 把控制面和数据面拆开：ZMQ 只做协调，不传 KV。职责包括：元数据握手、KV 传输参数、首 token 信号、完成/失败信号。

Decode **热路径上每请求高频收发** 时露出两件事：

1. 热路径 `logger.info` 量大、对排障价值低，本身有 CPU/IO 税
2. 每次发信令 `with zmq_ctx(...)` **新建/销毁 Context 与 Socket**，短消息场景下建连成本高于消息本身

识别位置在控制面 hot path，不是 Transfer Engine 带宽。

## 具体怎么改实现（已给出的部分）

1. 热路径 **6 处** `logger.info` → `logger.debug`
2. ZMQ Socket **池化复用**：`_req_socket_pool`，以 `addr` 为 key 的 `deque[zmq.Socket]`，线程安全

预期方向（需原文后半确认）：借出/归还 socket，避免每信令一次 `ctx+socket` 生命周期。

## 待补

原文在「连接池，线程安全。」之后结束。尚未记录：

- 池的上限、空闲回收、失败重连
- 哪些消息类型走池（握手 / 首 token / 完成）
- 量化收益（TTFT / CPU）
- 与 Layerwise-CPCD、batch KV 的时序关系

下一条材料若补 3.3.3 后半、3.4 MTP 零气泡、3.5 稳定性，接到本文件和新建 Decode/稳定性笔记。
