# vLLM Model Runner：V1 → V2

读 `docs/`。四篇，按改法合并，每篇只回答三件事：V1 怎么做、为什么要改、V2 改成了哪段代码。

| 文档 | 内容 |
|------|------|
| [docs/01-overview.md](docs/01-overview.md) | 三处改法、代码在哪、什么配置跑 V2 / 回退 V1 |
| [docs/02-persistent-batch.md](docs/02-persistent-batch.md) | 持久批、状态写入、block table、输入准备 |
| [docs/03-async.md](docs/03-async.md) | 异步竞态、结果回传、CUDA graph 与 `dummy_run` |
| [docs/04-sampler.md](docs/04-sampler.md) | 采样器、模块化、往 V2 加特性时要检查的七条 |

`prompts/` 是生成这四篇时用的提问稿，读文档不必打开。

V1 在 `vllm/v1/worker/gpu_model_runner.py`，V2 在 `vllm/v1/worker/gpu/`。路径对不上时：

```bash
ls vllm/v1/worker/gpu/
rg -n "use_v2_model_runner|_get_v2_model_runner_unsupported_features" vllm/config/vllm.py
```
