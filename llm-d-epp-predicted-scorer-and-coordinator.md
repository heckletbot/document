# llm-d：EPP 预测调度 Scorer 与 Coordinator

> 基于本地仓库 `D:\docs\llm-d`（[llm-d/llm-d](https://github.com/llm-d/llm-d.git)）的 docs / guides / proposals 整理。  
> 主仓是 umbrella（文档与部署指南）。EPP 插件与 Coordinator 的 Go 实现在 [llm-d-router](https://github.com/llm-d/llm-d-router)；延迟预测 sidecar 在 [llm-d-latency-predictor](https://github.com/llm-d/llm-d-latency-predictor)。

---

## 1. EPP 侧预测调度如何作为独立 Scorer 实现

### 1.1 结论

预测调度在 EPP 里不是一个单体插件，而是调度框架 **Filter → Score → Pick** 中的一个独立 Scorer。

核心路径是：

| 插件 | 角色 | 作用 |
|------|------|------|
| `predicted-latency-producer` | DataProducer（自动绑定） | 调用 latency-predictor sidecar，预测 TTFT / TPOT，写入 request context |
| `latency-scorer` | **独立 Scorer** | 用预测延迟（或 SLO headroom）给每个候选 endpoint 打 0–1 分 |
| `weighted-random-picker` | Picker | 按分数抽选 endpoint |

多 Scorer 的合并公式：

```text
final_score = Σ(scorer_score × weight)
每个 scorer_score ∈ [0, 1]
```

插件图用 `EndpointPickerConfig` YAML 声明（不是 K8s CRD），EPP 启动时读取。Helm 通过 `router.epp.pluginsCustomConfig` 注入。

### 1.2 调度框架位置

EPP（Endpoint Picker）是 InferencePool 的选路引擎。请求路径：

```text
Client → Proxy/Gateway (ext_proc) → EPP
  ├─ Request Handler（解析 OpenAI / vLLM 请求）
  ├─ Flow Control（可选，饱和时排队）
  ├─ DataProducer
  │    └─ predicted-latency-producer → sidecar 预测 TTFT/TPOT
  │    └─ approx/precise-prefix-cache-producer → prefix 索引
  ├─ Scheduler（ProfileHandler 选 profile）
  │    ├─ Filters
  │    ├─ Scorers（latency-scorer 在这里）
  │    └─ Picker
  └─ 返回 endpoint → Proxy 转发到 vLLM / SGLang
       └─ 请求结束后回传样本 → 训练 server 更新 XGBoost
```

相关文档：

- EPP 总览：`docs/architecture/core/router/epp/README.md`
- 调度框架：`docs/architecture/core/router/epp/scheduling.md`
- 配置图：`docs/architecture/core/router/epp/configuration.md`
- 延迟预测架构：`docs/architecture/advanced/latency-predictor.md`
- 部署指南：`guides/predicted-latency-routing/`

### 1.3 Scorer 如何注册、如何被调用

在 `EndpointPickerConfig` 里，插件是节点，profile 是边：

```yaml
plugins:
- type: predicted-latency-producer
- type: latency-scorer
  name: ttft-scorer          # 可选实例名；P/D 时会开两个实例
  parameters:
    ttftWeight: 1
    tpotWeight: 0
- type: weighted-random-picker

schedulingProfiles:
- name: default
  plugins:
  - pluginRef: latency-scorer
    weight: 1.0
  - pluginRef: weighted-random-picker
```

要点：

- `plugins`：实例化（type / name / parameters）。
- `schedulingProfiles`：把插件接到 Filter / Scorer / Picker 角色。
- DataProducer（如 `predicted-latency-producer`）实现对应接口后，在 `plugins` 里声明即可自动绑定，不必在 profile 里 `pluginRef`。
- 系统按接口分类：Filter 先跑，再 Scorer，最后 Picker。
- 一个 EPP 进程只允许一个 ProfileHandler。

Helm 开启 predictor sidecar：

```yaml
router:
  latencyPredictor:
    enabled: true
```

参考：`guides/predicted-latency-routing/router/predicted-latency.values.yaml`。

### 1.4 `latency-scorer` 的输入、输出与行为

**输入**（由 `predicted-latency-producer` 预先写入）：

- 每个候选 pod 的 predicted TTFT / TPOT
- pod 状态：KV 利用率、队列深度、running requests、prefix match %、in-flight input tokens 等

**输出**：每个 endpoint 的 `[0, 1]` 分数，交给 picker。

**行为**：

| 条件 | 打分策略 |
|------|----------|
| 无 SLO 头 | 选 predicted latency 最低的 endpoint |
| 有 SLO 头 `x-llm-d-slo-ttft-ms` / `x-llm-d-slo-tpot-ms` | 按 headroom（SLO − predicted）评分；`headroomSelectionStrategy: least\|most` |
| predictor 不可达 | 回退到 KV 利用率 + 队列深度 + prefix match 的启发式组合分 |

ML 侧：`llm-d-latency-predictor` 与 EPP 同 pod，colocate 训练 server + 预测 server。XGBoost 在线重训，TTFT / TPOT 各一个模型。

### 1.5 与其他 Scorer 的关系

预测延迟 Scorer 是对启发式加权组合的**替换**，不是简单叠加。

| 场景 | 典型组合 | 关系 |
|------|----------|------|
| Optimized Baseline | `queue-scorer` + `kv-cache-utilization-scorer` + `prefix-cache-scorer` + `no-hit-lru-scorer` | 纯启发式，需手调 weight |
| Predicted Latency | `latency-scorer` | 模型直接学 latency surface，不必调那组 weight |
| P/D + Predicted Latency | prefill：`ttft-scorer`；decode：`tpot-scorer`（两个 `latency-scorer` 实例） | 各阶段只优化一半延迟 |
| Precise Prefix Cache | `prefix-cache-affinity-filter` + `token-load-scorer` | 精确 KV 事件驱动，不是 ML 延迟预测 |

P/D 时同一个 `latency-scorer` 开两个实例：

```yaml
- type: latency-scorer
  name: ttft-scorer
  parameters:
    ttftWeight: 1
    tpotWeight: 0
- type: latency-scorer
  name: tpot-scorer
  parameters:
    ttftWeight: 0
    tpotWeight: 1

schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: ttft-scorer
- name: decode
  plugins:
  - pluginRef: tpot-scorer
```

参考：`guides/predicted-latency-routing/router/predicted-latency-pd.values.yaml`。

另有一类前缀 / KV「预测」（非 ML），与延迟预测正交：

- `prefix-cache-scorer` + `approx-prefix-cache-producer`（启发式 LRU）
- `precise-prefix-cache-producer` + `prefix-cache-affinity-filter` + `token-load-scorer`（vLLM KV ZMQ 精确索引）
- 旧插件 `precise-prefix-cache-scorer` 已废弃

### 1.6 使用场景与配置要点

**适合 `latency-scorer`：**

- prompt / completion 长度方差大，队列深度不能代表真实负载
- 需要 per-request SLO（TTFT / TPOT 头）
- 静态 scorer weight 已经调不动
- 池内 pod 同质（同 GPU / 同模型 / 同配置）
- P/D 模式需要 `"stream": true`（`streamingMode: true`）

**不适合：**

- 异构池（混合 GPU、模型变体、不同 serving 配置）
- OpenShift 上 predictor sidecar 目前不稳定（guide 有 NOTE）

**关键配置：**

| 字段 | 含义 |
|------|------|
| `router.latencyPredictor.enabled` | 启用 sidecar |
| `predicted-latency-producer.parameters.streamingMode` | `false` = e2e latency；`true` = 分离 TTFT/TPOT |
| `prefix-cache-affinity-filter.parameters.ttftSource: latencyPredictor` | filter 与 scorer 共用预测 TTFT |
| `latency-scorer.parameters.ttftWeight` / `tpotWeight` | P/D 分阶段评分 |
| `router.epp.pluginsConfigFile` + `pluginsCustomConfig` | 完整插件 DAG |

Go 接口（`Scorer`、`RegisterPlugin`）在 `llm-d-router/pkg/epp/framework/...`，本 umbrella 仓只有架构描述。

---

## 2. llm-d Coordinator 是什么：工作模式与使用场景

### 2.1 结论

**Coordinator（`llm-d-coordinator`）是 Gateway 入口前的独立请求编排服务。** 它是 P/D disaggregation routing sidecar 的**实验性替代**：不再在 decode pod 里一次选好 Prefill + Decode，而是按阶段、按需调用 Gateway / EPP。

- 镜像：`ghcr.io/llm-d/llm-d-router-coordinator:main`
- 实现仓：[llm-d-router](https://github.com/llm-d/llm-d-router)（`docs/coordinator_architecture.md`）
- 实验孵化：`proposals/coordinator.md` 提议 `llm-d-incubation/coordinator`
- 成熟度：guide 明确标 **experimental**，不如 `guides/pd-disaggregation` 成熟

### 2.2 架构位置

```text
Client → Gateway → Coordinator（无 EPP-Profile）
                 ↓ 每个 phase 出站一次
              Gateway → EPP（带 EPP-Profile: encode | prefill | decode）
                 ↓
              Model Server pods
```

默认 pipeline 写在 ConfigMap，可改，不必重建镜像（`guides/coord-disaggregation/coordinator/configmap.yaml`）：

```yaml
pipeline:
  kv_connector: kv-nixl
  ec_connector: ec-nixl
  steps:
  - type: replace-media-urls
  - type: render
  - type: conditional-decode
  - type: encode
  - type: prefill
  - type: decode
```

相对 sidecar P/D 的三个核心差异：

1. **模块化**：pipeline steps 是 ConfigMap 列表，可增删改顺序。
2. **Deferred decoding**：EPP 只在某阶段即将执行时才被调用；decode pod 基于最新池状态选择，而不是 prefill 前的快照。
3. **Optimistic decode**：`conditional-decode` 先尝试 decode；若 pod 已有缓存则直接服务，否则 412 再走完整 `encode → prefill → decode`。

Deployment 关键参数：

```yaml
- name: coordinator
  image: ghcr.io/llm-d/llm-d-router-coordinator:main
  args:
  - --config
  - /etc/coordinator/coordinator.yaml
```

### 2.3 工作模式

| 维度 | 模式 |
|------|------|
| 对客户端 | 同步 HTTP |
| 内部阶段 | encode / prefill / decode **串行**（encode 多模态条目可 `max_parallel: 8`） |
| 控制面 / 数据面 | Coordinator = 控制面编排（选 phase、发 HTTP）；KV / EC 传输仍走 NIXL（`kv-nixl` / `ec-nixl`） |
| 与 EPP | 每 phase 带 `EPP-Profile` header，触发对应 scheduling profile |
| 与 vLLM | 不替代 vLLM；调用 OpenAI API；render 用 sidecar `vllm launch render` |
| 与 Gateway | **必须 Gateway Mode**；Standalone Router 不支持此 guide |

EPP 拓扑两种选法（`guides/coord-disaggregation/README.md`）：

1. **单 EPP + `header-profile-handler`**：一个 InferencePool 覆盖 encode / prefill / decode，按 header 选 profile。
2. **三个独立 EPP**：每 role 一个 pool，HTTPRoute 按 `EPP-Profile` 分发。此时每个 EPP 只看见一种 role，chart 默认 `single-profile-handler` 即可。

单 EPP 时各 profile 的 scorer 不同：

```yaml
- name: encode
  plugins:
  - pluginRef: encode-filter
  - pluginRef: active-request-scorer
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: prefix-cache-affinity-filter
  - pluginRef: token-load-scorer
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: active-request-scorer
```

参考：`guides/coord-disaggregation/router/coord-disaggregation.values.yaml`。

### 2.4 与经典 P/D Sidecar 对比

| | P/D Sidecar（`pd-disaggregation`） | Coordinator |
|--|-----------------------------------|-------------|
| 编排位置 | decode pod 内 routing sidecar | 独立 Coordinator Service |
| EPP 调用 | 一次 ext_proc 同时选 P + D | 每 phase 一次 ext_proc |
| 流水线 | 固定 prefill → decode | ConfigMap 可配 EPD / PD-only |
| Decode 选 pod | prefill 前的快照 | encode / prefill 完成后再选 |
| 成熟度 | 生产 well-lit path | experimental |
| 额外 hop | 无独立编排层 | 多一次 Gateway / Coordinator；benchmark 显示 TTFT 约 +2–5%，ITL 几乎无差 |

请求始终打到选出的阶段 endpoint。有 `x-prefiller-host-port` 时，sidecar 先 remote prefill 再本地 decode。Coordinator 路径把「选 P、选 D」从一次调度拆成多次。

### 2.5 使用场景

**该用 Coordinator：**

- 多模态 EPD（如 Qwen3-VL）：需要 encode + prefill + decode，以及 render / replace-media-urls
- 需要可配置 pipeline（增删 phase；PD-only 只需 patch ConfigMap）
- 需要 deferred pod selection（decode 在 encode / prefill 完成后再选）
- 探索 `conditional-decode` 快路径（缓存命中跳过 encode / prefill）

**不该用：**

- 纯文本 P/D 生产流量：优先 `guides/pd-disaggregation`（sidecar 更成熟，少一次 hop）
- 不可信客户端：`EPP-Profile` 是普通 HTTP 头，可被伪造绕过 Coordinator pipeline
- Standalone Router：guide 不支持
- 未验证环境（guide 以 GKE / base 为主）

**关键配置：**

| 配置 | 位置 | 说明 |
|------|------|------|
| `pipeline.steps` | `coordinator/configmap.yaml` | 流水线步骤 |
| `gateway.address` | 同上 | Coordinator 出站 Gateway URL |
| `patch-pd-only.yaml` | `coordinator/` | 去掉 encode / render，变成 PD-only |
| `router/coord-disaggregation*.values.yaml` | `guides/coord-disaggregation/router/` | EPP profiles + scorers |
| `coordinator/httproute.yaml` + `router/httproute.yaml` | 路由分流 | 无 header → Coordinator；有 `EPP-Profile` → EPP |

PD-only 切换：scale encode=0，再 patch pipeline 只留 `conditional-decode` / `prefill` / `decode`。

---

## 3. 实例 Role（补充）

同一 InferencePool 内用 Pod 标签 `llm-d.ai/role` 区分能力：

| 值 | 含义 |
|----|------|
| `encode` | 只做多模态 encode |
| `prefill` | 只做 prefill |
| `decode` | 只做 decode |
| `encode-prefill` | encode + prefill |
| `prefill-decode` | 混部，P 和 D 都能做 |
| `encode-prefill-decode` | 全阶段一体 |
| `both` | 已废弃，等同 `prefill-decode` |

内置 filter：

- `encode-filter`：`encode` / `encode-prefill` / `encode-prefill-decode`
- `prefill-filter`：`prefill` / `encode-prefill` / `prefill-decode` / `both` / `encode-prefill-decode`
- `decode-filter`：`decode` / `prefill-decode` / `both` / `encode-prefill-decode`，以及**未打该 label 的 pod**

ProfileHandler 是 EPP 启动时写死的，不会按请求或实例类型在 `single-profile-handler` 与 `disagg-profile-handler` 之间切换。池里只要有 PD 分离，就要用 `disagg-profile-handler`；每条请求走不走分离由 **decider**（如 `always-disagg-pd-decider` / `prefix-based-pd-decider`）决定。

---

## 4. 本仓关键文件索引

| 主题 | 路径 |
|------|------|
| 延迟预测 guide | `guides/predicted-latency-routing/` |
| 精确 prefix cache guide | `guides/precise-prefix-cache-routing/` |
| EPP 调度 / 配置 | `docs/architecture/core/router/epp/` |
| 延迟预测架构 | `docs/architecture/advanced/latency-predictor.md` |
| P/D 架构（sidecar 对比） | `docs/architecture/advanced/disaggregation/README.md` |
| Coordinator guide | `guides/coord-disaggregation/` |
| Coordinator proposal | `proposals/coordinator.md` |

外部实现仓：

| 组件 | 仓库 |
|------|------|
| EPP 插件（scorer / filter / producer / picker / Coordinator） | [llm-d/llm-d-router](https://github.com/llm-d/llm-d-router) |
| 延迟预测 ML sidecar | [llm-d/llm-d-latency-predictor](https://github.com/llm-d/llm-d-latency-predictor) |
| 精确 KV 索引 | [llm-d/llm-d-kv-cache](https://github.com/llm-d/llm-d-kv-cache) |
| EPP 宿主框架 / ext-proc | [kubernetes-sigs/gateway-api-inference-extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) |
