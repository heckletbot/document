# vLLM Model Runner：V1 和 V2

读 `docs/`。名词用代码里的英文，说明用中文。函数短就贴原码，长就写它做哪几步，并链到 [vLLM `e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8)。

| 文档 | 内容 |
|------|------|
| [docs/01-overview.md](docs/01-overview.md) | 三处差别，以及什么配置走 V2 |
| [docs/02-persistent-batch.md](docs/02-persistent-batch.md) | Persistent batch：`condense` 与 `idx_mapping`，以及 `input_ids` 怎么填 |
| [docs/03-async.md](docs/03-async.md) | 异步调度：batch queue 优先排下一批，V1 的 `synchronize`，V2 的 `UvaBufferPool` |
| [docs/04-sampler.md](docs/04-sampler.md) | Sampler：参数放进 slot，V1 和 V2 的 7 处不同，为什么换成 Triton，以及 Triton 算子表 |
| [docs/05-other-features.md](docs/05-other-features.md) | V2 其他特性：GPU 上准备输入和 UVA、模块拆分、`dummy_run`、`CudaGraphManager`、融合多步 draft 解码 |

`prompts/` 是提问稿。正文以 `docs/` 为准。
