# 分层传输先 P 后 D（Layerwise-CPCD）

- 链路：RouteServer ↔ Prefill ↔ Decode 的 KV 握手与调度顺序
- 收益：原文「先 P 后 D」单并发 TTFT ↓ 70ms+；适配后 PD 配比调整 + P 执行与等 D 地址异步化，纯文本再降 20ms+
- 配置：`VLLM_ASCEND_ENABLE_TRANSFER_MM_FEATURES=1`；RS `--infer-mode Layerwise-CPCD`

## 问题怎么识别

攻关初期社区只有 **先 D 后 P + 分层传输**：

- Decode 要先把 KV 接收缓冲分配好，Prefill 才能开始传（甚至才能安心跑）
- 这段「等 D 侧 malloc/登记地址」是 **计算空泡**，打进 TTFT
- 后面还要上多模态（P 更重、图处理更长），先 D 后 P 会把空泡和重 P 绑死，后续优化空间被堵

所以这是 **提前识别的架构债**，不是上线后才从 trace 里扫出来的。结论：必须改成 P 先跑，D 的地址握手和 P 的计算重叠。

## 具体怎么改实现

支持 PD 并行：Decode 在 Prefill **还在算** 时就开始按层收 KV。调度时序：

```text
User → RouteServer
         │
         ├─ 调度 Prefill，请求挂起
         │     Prefill 预处理完成
         │     ──回调──► RS /v1/metaserver  (kv_transfer_params)
         │
         ├─ RS 调度 Decode，把 KV 元数据转过去
         │
         ├─ Prefill 在等 Decode「地址就绪」通知期间：异步推理
         │     首 token 先回客户端
         │     收到 Decode 消息后，按层传 KV → Decode
         │
         └─ Decode 收齐 KV 后开始解码，后续 token 回客户端
```

和先 D 后 P 的本质差：

| | 先 D 后 P | 先 P 后 D（本方案） |
|--|-----------|-------------------|
| 谁先被 RS 选中 | Decode（先占收端内存） | Prefill |
| TTFT 里等 KV 地址 | 在 P 启动前，全同步 | P 计算与等 D 消息异步重叠 |
| 首 token | 往往等 PD 握手更完整 | P 算完就可回，不绑死 D 开算 |

现场还改了 PD 配比，并把「prefill 执行」与「等待 decode kvcache 地址」彻底异步。这为多模态打底：图处理和 ViT 可以在 P 侧先走，不必先等 D 分配。

## 学习要点

- 识别的是 **调度顺序造成的空泡**，不是传输带宽不够。
- 实现落在 RS 模式（`Layerwise-CPCD`）+ P 侧「先算后传、等地址不堵 forward」+ 分层 KV。
- 和 [12](12-kv-batch-sync.md) 正交：一个改谁先启动，一个改每层 sync 粒度。
