# V1 → V2：三处改法

代码锚在 vLLM [`e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8)。V1 是 [`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py)，V2 是 [`vllm/v1/worker/gpu/`](https://github.com/vllm-project/vllm/tree/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu)。两个类都叫 `GPUModelRunner`。实例化在 [`gpu_worker.py` `_make_model_runner`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_worker.py#L567-L583)。

官方主张在 [`docs/design/model_runner_v2.md`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/docs/design/model_runner_v2.md)。下面三篇按改法写：V1 的代码、为什么要改、V2 改成的代码。

| 改法 | V1 | V2 | 文档 |
|------|----|----|------|
| 持久批与输入准备 | 持久张量就是本步输入，结束请求要 `condense()` 挪行，block table 整段 `copy_to_gpu` | 请求占固定行，本步用 `idx_mapping` gather；增量走 `StagedWriteTensor` | [02](02-persistent-batch.md) |
| 异步调度 | `synchronize_input_prep()` 等上一步的 H2D；`_to_list()` 在主路径上 `synchronize` | 真值不 pin，拷进轮转槽；同步留在 `AsyncOutput.get_output()` | [03](03-async.md) |
| 采样器与模块化 | `SamplingMetadata` 每步重建；拒绝采样里 `expand_batch_to_tokens` 把参数铺成 token 行 | 参数留在请求行，`gumbel_block_argmax` 用 `expanded_idx_mapping` 查 | [04](04-sampler.md) |

## 什么时候跑 V2

判定在 [`use_v2_model_runner`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L703-L740)，从上往下短路：

1. HiSparse：强制 V2。环境变量设成 `0` 时抛 `ValueError`。
2. watermark：强制 V2。设成 `0` 只打日志，照样返回 `True`。
3. `VLLM_USE_V2_MODEL_RUNNER` 有值：听这个值。
4. 没装 Triton：回退 V1。
5. [`_get_v2_model_runner_unsupported_features`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L3011-L3045) 非空：回退 V1。清单是 `external_launcher` 且 PP>1、投机方法 `suffix` / `custom_class`、非 dflash/dspark 的 parallel drafting、异构词表 draft，以及 `use_ubatching` 时并入的 DBO 项。
6. 其余：V2。

显式要求 V2 时同一份 V2 清单走 `_validate_v2_model_runner()`，改成抛异常。

[`_get_v1_model_runner_unsupported_features`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L3047-L3093) 列出 V1 直接拒绝的配置：PCP、PP 加异步调度、DSpark、自适应验证、draft lm_head 量化、混合 sliding/full 的 DFlash、DFlash candidate-head、diffusion、batch-sharded sampling。注释写明 candidate-head 在 V1 上会静默走 `DFlashProposer`，退化成 DFlash1；MRV1 已标成 deprecated，PP+异步调度因此不修、直接禁用。
