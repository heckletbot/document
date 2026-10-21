# V1 和 V2 差在哪

代码版本：[vLLM `e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8)。

V1 的 `GPUModelRunner` 在 [`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py)。V2 的同名类在 [`vllm/v1/worker/gpu/`](https://github.com/vllm-project/vllm/tree/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu)。Worker 在 [`_make_model_runner`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_worker.py#L567-L583) 里二选一。官方设计说明是 [`docs/design/model_runner_v2.md`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/docs/design/model_runner_v2.md)。

| | V1 | V2 | 说明 |
|--|----|----|------|
| Persistent batch | 这些 tensor 就是这一 step 的 model input。Request 结束要 `condense()` 把后面的 row 往前挪 | 每个 request 一个固定 slot。这一 step 的 input 用 `idx_mapping` gather 出来 | [02](02-persistent-batch.md) |
| Async scheduling | `synchronize_input_prep()` 等上一 step 的 H2D 结束，CPU 才能改同一块 pinned memory。`_to_list()` 在 sample 之后立刻 `synchronize` | CPU 写的 tensor 不 pin。每 step 拷到 pool 里的另一个 buffer，GPU 读那一份。`AsyncOutput.get_output()` 才 `synchronize` | [03](03-async.md) |
| Sampler | `SamplingMetadata` 每 step 重建。Speculative decoding 里 `expand_batch_to_tokens` 把 temperature 复制成和 logits 行数一样长 | Temperature 留在 request 的 slot 上。`gumbel_block_argmax` 用 `expanded_idx_mapping` 查 | [04](04-sampler.md) |

## 哪个配置走 V2

[`use_v2_model_runner`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L703-L740) 按这个顺序返回：

1. 配了 HiSparse：返回 V2。`VLLM_USE_V2_MODEL_RUNNER=0` 会抛 `ValueError`。
2. 配了 watermark：返回 V2。设成 `0` 只打一条日志。
3. 环境变量有值：用这个值。
4. 没有 Triton：V1。
5. [`_get_v2_model_runner_unsupported_features`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L3011-L3045) 非空：V1。里面是 `external_launcher` 且 pipeline parallel > 1、speculative method 为 `suffix` 或 `custom_class`、EAGLE 的 parallel drafting、异构词表的 draft model，以及打开 micro-batch 时并进来的 DBO 项。
6. 其余：V2。

调用方如果已经认定要用 V2，同一份清单改走 `_validate_v2_model_runner()`，命中就抛异常。

[`_get_v1_model_runner_unsupported_features`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L3047-L3093) 是 V1 拒绝的配置：PCP、pipeline parallel 加 async scheduling、DSpark、adaptive verification、draft `lm_head` 量化、sliding 和 full attention 混用的 DFlash、带 candidate head 的 DFlash、diffusion、batch-sharded sampling。注释写明 candidate head 在 V1 上会走 `DFlashProposer`，效果退化成 DFlash1。同函数注释也写明 MRV1 已 deprecated，pipeline parallel 加 async scheduling 这个组合不修，直接禁用。
