# vLLM Model Runner：V1 → V2

读 `docs/`。四篇都对照 vLLM [`e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8) 的源码，V2 的改法附了代码和文件链接。

| 文档 | 内容 |
|------|------|
| [docs/01-overview.md](docs/01-overview.md) | 三处改法、runner 怎么选、两边各自不支持什么 |
| [docs/02-persistent-batch.md](docs/02-persistent-batch.md) | 固定行、staged write、gather、GPU 上填 `input_ids` |
| [docs/03-async.md](docs/03-async.md) | 输入屏障、`AsyncOutput`、CUDA graph |
| [docs/04-sampler.md](docs/04-sampler.md) | `expanded_idx_mapping`、logprobs、`ModelState` / speculator |

`prompts/` 是提问稿。正文以 `docs/` 为准。
