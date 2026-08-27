# 推理框架系统分析流程

这个目录用来整理「推理框架」的系统分析流程：先定分析顺序，再按模块往下填结论、图和源码对照。

当前是占位，内容后续慢慢补。仓库里已有的相关笔记：

- [推理框架负载均衡路由器](../reasoning-lb-router-design.md)
- [llm-d：EPP 预测调度 Scorer 与 Coordinator](../llm-d-epp-predicted-scorer-and-coordinator.md)
- [Mooncake 功能学习计划](../mooncake-learning-plan/README.md)

---

## 怎么用

1. 先在本 README 里把分析流程（阶段、输入、产出）写清楚。
2. 每个阶段单独开文档，放到 `process/`。
3. 结论、接口、图、源码路径尽量写在对应阶段文档里，避免全堆在这一页。

---

## 分析流程（待填）

| 阶段 | 文档 | 状态 |
|------|------|------|
| 范围与目标 | [process/01-scope.md](process/01-scope.md) | 待填 |
| 请求路径 | [process/02-request-path.md](process/02-request-path.md) | 待填 |
| 调度与路由 | [process/03-scheduling.md](process/03-scheduling.md) | 待填 |
| 执行引擎 | [process/04-engine.md](process/04-engine.md) | 待填 |
| KV / PD | [process/05-kv-pd.md](process/05-kv-pd.md) | 待填 |
| 对照结论 | [process/06-comparison.md](process/06-comparison.md) | 待填 |

阶段划分可改，填的时候直接改表。

---

## 约定

- 每个阶段文档顶部写：目标、要看的内容、要掌握的知识。
- 源码路径写本地仓库根（例如 `D:\docs\...`）和上游链接。
- 图优先用 Mermaid，和图下要点一起放。
