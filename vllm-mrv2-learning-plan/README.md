# vLLM Model Runner：V1 和 V2

读 `docs/`。名词用代码里的英文，说明用中文。函数短就贴原码，长就写它做哪几步，并链到 [vLLM `e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8)。

| 文档 | 内容 |
|------|------|
| [docs/01-overview.md](docs/01-overview.md) | 三处差别，以及什么配置走 V2 |
| [docs/02-persistent-batch.md](docs/02-persistent-batch.md) | Persistent batch：`condense` 与 `idx_mapping`，以及 `input_ids` 怎么填 |
| [docs/03-async.md](docs/03-async.md) | 异步调度：batch queue 优先排下一批，V1 的 `synchronize`，V2 的 `UvaBufferPool` |
| [docs/04-sampler.md](docs/04-sampler.md) | `expand_batch_to_tokens`、`expanded_idx_mapping`、`ModelState` |

`prompts/` 是提问稿。正文以 `docs/` 为准。
