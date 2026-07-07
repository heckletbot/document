# 推理框架负载均衡路由器

> PPT 素材：Mermaid + 图下要点

---

## 1. 两种模式

```mermaid
flowchart TB
    R["Load Balance Router"]
    R --> M["混合模式 · 1 hop"]
    R --> PD["PD 分离 · 2 hop"]
    M --> W["Worker\nPrefill+Decode"]
    PD --> P["P Worker"] -->|"kvTransferParams"| D["D Worker"]
```

| | 混合 | PD 分离 |
|---|---|---|
| Worker | 同质 | P + D |
| 路由 | 1 次 | P 选路 + D 选路 |
| KV 跨节点 | 否 | 是 |
| 队列 | `queue` | `queue_P` + `queue_D` |

---

## 2. 请求输入（必填）

```mermaid
flowchart LR
    A["Agent"] --> R["Router"]
    A -.->|messages| R
    A -.->|priority| R
    A -.->|output_length| R
    A -.->|session_id| R
    A -.->|prefix_hash 推荐| R
```

| 字段 | 用途 |
|---|---|
| `priority` | WSPT 排序；入队处理一次 → 透传 vLLM/SGLang |
| `output_length` | osl；WSPT cost；透传后端 |
| `session_id` | 多轮 / affinity |
| `prefix_hash` | 渐变选 Worker；kv_preset |

```json
{
  "messages": [...],
  "metadata": {
    "orla": {
      "session_id": "run-42",
      "priority": 5,
      "output_length": 1024,
      "cache": { "prefix_hash": "sha256:abc" }
    }
  }
}
```

- 缺 `priority` / `output_length` / `session_id` → **400**

---

## 3. Router 状态表

| 表 | 内容 | 更新 |
|---|---|---|
| `in_flight` | `worker_id → count` | 到达 +1，离开 -1 |
| `kv_preset` | Worker 上 PrefixHash 快照 | Prefill/Decode 完成 Upsert |

```text
L = Σ in_flight[w]    // 总在途，上限 32
```

---

## 4. 优先级队列 + WSPT

### 准入

```mermaid
flowchart TD
    Req["新请求"] --> Q{"有 Worker\n负载适宜?"}
    Q -->|是| Go["直发"]
    Q -->|全池过载| Enq["入队 WSPT"]
    Enq --> Wait["等待"]
    Wait --> Deq["出队"]
    Deq --> Go
    Go --> W["Worker"]
    W --> Idle["in_flight -= 1"]
    Idle --> Wait
```

| 路径 | 条件 |
|---|---|
| 直发 | ∃ Worker：`in_flight ≤ T_low` |
| 入队 | ∀ Worker：`in_flight > T_low` |
| 出队 | 某 Worker 回落 → WSPT pop |

### WSPT

```text
wspt_key = (1 + priority) / f(isl_tokens, output_length)
```

- 出队结合 **最低 load Worker** + **WSPT 序**
- `priority` 入队读一次；dispatch 透传后端（vLLM 需极性转换）

```mermaid
sequenceDiagram
    participant A as Agent
    participant R as Router
    participant Q as queue
    participant W as vLLM/SGLang
    A->>R: priority + output_length + session_id
    R->>R: wspt_key（一次）
    R->>Q: Enqueue
    Q->>R: Dequeue
    R->>W: forward(priority, output_length, session_id)
```

### Priority 透传

| 层 | 职责 |
|---|---|
| Router | 准入 + WSPT |
| vLLM / SGLang | 引擎排队 + KV eviction |

- PD：P、D 各透传一次，D 不重算 WSPT
- 后端需开启 priority-aware 调度

### 队列拆分

| 模式 | 队列 |
|---|---|
| 混合 | `queue` |
| PD | `queue_P`（Prefill 前）· `queue_D`（Decode 前） |

---

## 5. 渐变选 Worker

```text
w_prefix(L) = 1 - L/32
w_load(L)   = L/32

score(w) = w_prefix × hash_hit(w, H) + w_load × load_norm(w)
hash_hit = kv_preset.Has(w, prefix_hash)
```

| L | 倾向 |
|---|---|
| 0~16 | 优先 PrefixHash；无命中 → 最低 load |
| 17~32 | load 权重上升，少选 Hash 热点 |

```mermaid
flowchart TD
    H["prefix_hash"] --> L["L = Σ in_flight"]
    L --> S["score 最大 Worker"]
    S --> Inc["in_flight += 1"]
    Inc --> Fwd["转发"]
```

- 队列决定 **何时**；渐变决定 **发给谁**
- 出队用最新 in_flight / kv_preset

---

## 6. 混合模式

### 架构

```mermaid
flowchart TB
    A["Agent"] --> E["入口"]
    E --> Ad["准入"]
    Ad --> Q["queue"]
    Ad --> Sel["渐变选 Worker"]
    Q --> Sel
    Sel --> IF["in_flight"]
    Sel --> KV["kv_preset"]
    Sel --> W1["Worker"]
    Sel --> W2["Worker"]
    W1 --> D["drain"]
    D --> Q
```

### Process Map

```mermaid
flowchart LR
    S1["收请求"] --> S2["准入"]
    S2 --> S3["渐变选 Worker"]
    S3 --> S4["in_flight += 1"]
    S4 --> S5["推理"]
    S5 --> S6["-= 1 · drain"]
```

```mermaid
sequenceDiagram
    participant A as Agent
    participant R as Router
    participant W as Worker
    A->>R: priority + output_length + session_id + prefix_hash
    R->>R: WSPT or 直发
    R->>R: 渐变选 Worker
    R->>W: 转发
    W-->>R: 完成
    R->>R: drain queue
```

### Data Flow

```mermaid
flowchart LR
    IN["messages + meta"] --> R["Router"]
    IF["in_flight"] --> R
    KV["kv_preset"] --> R
    R --> W["Worker"]
    W --> OUT["响应"]
```

---

## 7. PD 分离模式

### 架构

```mermaid
flowchart TB
    A["Agent"] --> R["Router"]
    R --> QP["P 准入 / queue_P"]
    QP --> P["P Prefill"]
    P --> KVT["kvTransferParams"]
    KVT --> QD["D 准入 / queue_D"]
    QD --> D["D Decode"]
    D --> A
```

### Process Map

```mermaid
flowchart LR
    S1["收请求"] --> S2["P 准入"]
    S2 --> S3["选 P"]
    S3 --> S4["Prefill"]
    S4 --> S5["kvTransferParams"]
    S5 --> S6["D 准入"]
    S6 --> S7["选 D"]
    S7 --> S8["Decode · drain"]
```

```mermaid
sequenceDiagram
    participant A as Agent
    participant R as Router
    participant P as P Worker
    participant D as D Worker
    A->>R: priority + output_length + session_id + prefix_hash
    R->>P: Prefill
    P-->>R: kvTransferParams
    R->>D: Decode + kvTransferParams
    D-->>R: 完成
    R-->>A: 响应
```

### P 选路

- 公式同 §5，`L` 仅统计 **P 池**
- `in_flight_P`：进 P +1，Prefill 完成 -1
- `kv_preset_P`：P 侧 PrefixHash

### kvTransferParams

| 字段 | 说明 |
|---|---|
| `source_p_worker` | 来源 P |
| `kv_block_handles` | KV 传输句柄 |
| `prefix_hash` | 可选 |
| `token_count` | Prefill tokens |

### D 选路（接口预留）

```text
SelectDWorker(prefix_hash, kvTransferParams, in_flight_D, kv_preset_D) → d_id
// TODO: PrefixHash Decode + load 惩罚渐变（结构同 P 池）
// v0 fallback: argmin in_flight_D
```

| 表 | 更新 |
|---|---|
| `in_flight_D` | 进 D +1，Decode 完成 -1 |
| `kv_preset_D` | D 侧 PrefixHash（预留） |

### Data Flow

```mermaid
flowchart LR
    IN["请求"] --> R["Router"]
    IFP["in_flight_P"] --> R
    KVP["kv_preset_P"] --> R
    R --> P["P"]
    P --> KVT["kvTransferParams"] --> R
    IFD["in_flight_D"] --> R
    KVD["kv_preset_D"] --> R
    R --> D["D"]
    D --> OUT["响应"]
```

---

## 8. 接口汇总

```text
// 准入 / 队列
ShouldBypassQueue(pool) → bool
Enqueue(pool, req) / DequeueWSPT(pool) → req
OnWorkerIdle(w, pool)

// 选路
SelectWorker(H, in_flight, kv_preset) → worker_id      // 混合 / P
SelectDWorker(H, kvTransferParams, …) → d_id           // D · TODO

// 状态
in_flight[worker] += 1 / -= 1
HasPreset(worker, prefix_hash) / UpsertPreset(...)

// 透传
NormalizeIngress(req) → {priority, output_length, session_id}
MapEnginePriority(backend, priority)
AttachToForward(req, …)
```

---

## 9. 对照

| | 混合 | P 池 | D 池 |
|---|---|---|---|
| 渐变选路 | ✓ | ✓ | 预留 |
| WSPT 队列 | `queue` | `queue_P` | `queue_D` |
| kv_preset | ✓ | ✓ | 预留 |
| priority 透传 | 引擎 | P 引擎 | D 引擎 |
