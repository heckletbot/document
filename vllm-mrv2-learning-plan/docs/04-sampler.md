# Sampler 和模块拆分

代码版本：[vLLM `e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8)。

V1 的采样参数按这一 step 的 batch 行重建。一条 request 对应多条 logits 时，要把参数复制成和 logits 一样长。V2 的参数留在 request 的 slot 上，kernel 用 `expanded_idx_mapping` 从一条 logits 查回那个 slot。

## 参数放哪，logits 怎么对上

V1 的 [`SamplingMetadata`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/metadata.py#L15-L48) 里有 `temperature` tensor、`generators: dict[int, torch.Generator]`，以及 `all_greedy` 这种整批一个布尔。它由 [`_make_sampling_metadata`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L851) 在 `refresh_metadata` 时重建。`generators` 的 key 是 request 下标，`condense` 挪 row 时要跟着改 key（[`swap_states` 第 685 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L685)、[`condense` 第 809–811 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L809-L811)）。

Speculative decoding 里，拒绝采样把长度为 batch 的参数复制成长度为 logits 行数。[`expand_batch_to_tokens`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/rejection_sampler.py#L581-L592) 的 docstring：`x = [a, b, c]`、`cu_num_tokens = [2, 5, 6]`，结果是 `[a, a, b, b, b, c]`。Temperature、top-k、top-p 都走它（[第 550–574 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/rejection_sampler.py#L550-L574)）。Draft model 那边还有一处 `temperature.repeat_interleave`（[`llm_base_proposer.py` 第 462–470 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/spec_decode/llm_base_proposer.py#L462-L470)）。

V2 不复制这行参数。[`gumbel_block_argmax`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/gumbel.py#L237-L239)：

```python
req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0)
```

`token_idx` 是一条 logits 的下标。`expanded_idx_mapping[token_idx]` 是它所属 request 的 slot，temperature 从这行读取。

随机数没有 `torch.Generator`。[`murmur3_uniform32`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/gumbel.py#L119) 用 seed、位置、offset 在 kernel 里算出来。每个 draft step 的 seed 由同文件的 `spec_step_seed(seed, step)` 从请求的 seed 派出来。

[`Sampler.__init__`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/sampler.py#L69-L76) 里 logits processor 的顺序是 `logit_bias`、`penalties`、`bad_words`，然后才是自定义的。注释写明：bias 做加法，penalties 做缩放，次序换了结果不同。

同文件第 88–94 行：`return_sampling_mask=True` 时不用 flashinfer sampler，也不用 XPU sampler。XPU 分支还要求 `use_fp64_gumbel` 为假，注释写的是 XPU kernel 只抽 fp32 的指数噪声。

Logprobs 的差别：

- V1 [`compute_logprobs`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/sampler.py#L368-L370) 对整张词表做 `logits.log_softmax(dim=-1)`，再 `gather_logprobs` 取出要的 token。
- V2 [`compute_topk_scores`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/logprob.py#L124-L133) 先 `torch.topk(logits, num_logprobs)`，再 `compute_token_logprobs` 只算这些 token 的 logprob。

## 不同模型、不同投机方法放在哪

V1 的 runner 同时继承三个 mixin（[`gpu_model_runner.py` 第 466–468 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L466-L468)）：`LoRAModelRunnerMixin`、`KVConnectorModelRunnerMixin`、`ECConnectorModelRunnerMixin`。Speculative decoding 的入口是同一个类上的 [`propose_draft_token_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L4743)，各种 proposer 在 [`vllm/v1/spec_decode/`](https://github.com/vllm-project/vllm/tree/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/spec_decode)。

V2 的 runner 只继承 `LoRAModelRunnerMixin`（[`gpu/model_runner.py` 第 197 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L197)）。其余按类型分文件：

| 要换的东西 | 改哪里 |
|------------|--------|
| 模型类型（Mamba、encoder-decoder 等） | [`resolve_model_state_cls`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_states/__init__.py#L28-L64)。模型自己有 `get_model_state_cls` 就用它；否则按有没有 `CrossAttention`、是不是 encoder-only、是不是 hybrid，最后落到 `DefaultModelState` |
| Speculative method | [`init_speculator`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/spec_decode/__init__.py#L13) 按 `speculative_config.method` 返回类。`suffix` 和 `custom_class` 没有分支，配置校验在 `_get_v2_model_runner_unsupported_features` 里直接拒绝 |
| KV connector | [`KVConnector`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/kv_connector.py#L30) 的方法是空的，`ActiveKVConnector` 才真正传 KV。这一 step 没有 token 要算时，`execute_model` 调 `no_forward` |
| Mamba 的状态迁移 | [`MambaHybridModelState.preprocess_state`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L180-L191)。Docstring 要求它在 `prepare_attn` 读取 `num_accepted_tokens` 之前跑，这样跨 block 的状态迁移能被这一 step 的 attention 看到 |

PCP 的实现在 [`pcp_manager.py`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/pcp_manager.py)。V1 的不支持清单里写的是 "PCP runtime support is implemented only by the V2 model runner"。

往 V2 加一个功能时，代码里已经有的接入点和上面这张表一致：每个 request 一份状态就放 `RequestState`，或者这个功能自己的 `StagedWriteTensor` / `UvaBackedTensor`；要对齐到每一条 logits，用 `expanded_idx_mapping`，不要做 `expand_batch_to_tokens` 那种复制；batch 的 token 数、request 数、活跃 LoRA 数变了，就要进 `BatchExecutionDescriptor` 的字段，否则对不上已捕获的 CUDA graph；和现有配置互斥，就写进 `_get_v2_model_runner_unsupported_features`，或者单独一个 `_validate_*`。Adaptive verification 的校验在 [`_validate_adaptive_verification`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L3095-L3122)：不能和 LoRA 同时开，要 full CUDA graph，pipeline parallel 必须是 1。
