# 采样器与模块化

V1 的采样参数按本步 batch 行重建，拒绝采样要把它们铺成和 logits 行数一样长。V2 把参数留在请求的固定行上，kernel 用 `expanded_idx_mapping` 从一条 logits 查回那一行。模型族和投机方法同样从 runner 的分支里挪到单独的类。

## 采样参数和查表

V1 的 [`SamplingMetadata`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/metadata.py#L15-L48) 带 `temperature` 张量、`generators: dict[int, torch.Generator]`、`all_greedy` 这类整批布尔。它由 [`InputBatch._make_sampling_metadata`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L851) 在 `refresh_metadata` 时重建。`generators` 的键是请求下标，[`swap_states` 第 685 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L685) 和 `condense` 都要搬这个 dict。

投机解码下，拒绝采样把 `[batch]` 铺成 `[num_tokens]`。[`expand_batch_to_tokens`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/rejection_sampler.py#L581-L592) 的注释举例：`x = [a, b, c]`、`cu_num_tokens = [2, 5, 6]` 得到 `[a, a, b, b, b, c]`。temperature、top-k、top-p 都走这个函数，见 [第 550–574 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/rejection_sampler.py#L550-L574)。草稿模型侧还有 `temperature.repeat_interleave`，在 [`llm_base_proposer.py` 第 462–470 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/spec_decode/llm_base_proposer.py#L462-L470)。

V2 的 [`gumbel_block_argmax`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/gumbel.py#L215-L239) 不铺张量：

```python
req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0)
```

`token_idx` 是一条 logits，`temp_ptr + req_state_idx` 是该请求的温度行。随机数由 [`murmur3_uniform32`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/gumbel.py#L119) 从 seed、位置、offset 算出，没有 `torch.Generator`。草稿步的 seed 用同文件的 `spec_step_seed`。

[`Sampler.__init__`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/sampler.py#L69-L76) 把 logits processor 排成 `logit_bias`、`penalties`、`bad_words`、自定义。注释写明 bias 是加、penalties 是缩放，两者不可交换。`return_sampling_mask` 为真时 `use_flashinfer` 和 `use_xpu_sampler` 都是假；XPU 分支还要求 `not use_fp64_gumbel`，见同文件第 88–94 行。

logprobs：V1 的 [`compute_logprobs`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/sampler.py#L368-L370) 是 `logits.log_softmax(dim=-1)`，再 `gather_logprobs`。V2 的 [`compute_topk_scores`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/sample/logprob.py#L111-L133) 先 `torch.topk(logits, num_logprobs)`，再 `compute_token_logprobs` 只算这些 token。

## 类怎么拆

V1 runner 的基类是三个 mixin，见 [`gpu_model_runner.py` 第 466–468 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L466-L468)：`LoRAModelRunnerMixin`、`KVConnectorModelRunnerMixin`、`ECConnectorModelRunnerMixin`。投机入口是同一个类上的 [`propose_draft_token_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L4743)，proposer 在 [`vllm/v1/spec_decode/`](https://github.com/vllm-project/vllm/tree/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/spec_decode)。

V2 runner 只继承 `LoRAModelRunnerMixin`，见 [`gpu/model_runner.py` 第 197 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L197)。

| 差异 | V2 代码 |
|------|---------|
| 模型族 | [`resolve_model_state_cls`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_states/__init__.py#L28-L64)：模型自带 `get_model_state_cls`，否则按 CrossAttention、encoder-only、hybrid、Default 选 |
| 投机 | [`init_speculator`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/spec_decode/__init__.py#L13) 按 `method` 返回具体类。`suffix` / `custom_class` 没有分支，配置侧在 `_get_v2_model_runner_unsupported_features` 里拒绝 |
| KV / EC | [`kv_connector.py`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/kv_connector.py#L30) 的 `KVConnector` 空方法和 `ActiveKVConnector`。零 token 的 step 调 `no_forward` |
| Mamba 顺序 | [`MambaHybridModelState.preprocess_state`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L180-L191) 的 docstring：跨 block 迁移要在 `prepare_attn` 收集 `num_accepted_tokens` 之前做 |

PCP 的运行时在 [`pcp_manager.py`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/pcp_manager.py)。V1 侧没有对应实现，`_get_v1_model_runner_unsupported_features` 里写明 "PCP runtime support is implemented only by the V2 model runner"。

往 V2 加特性时，代码里已经固定的接入点就是上面这些：持久列用 `RequestState` 或该特性自己的 `StagedWriteTensor` / `UvaBackedTensor`；和 logits 对齐用 `expanded_idx_mapping`，按 [`expand_batch_to_tokens`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/sample/rejection_sampler.py#L581-L592) 那种方式铺开是 V1 的做法；批形状变化要进 `BatchExecutionDescriptor` 的字段；互斥进 `_get_v2_model_runner_unsupported_features` 或单独的 `_validate_*`（自适应验证的三条在 [`_validate_adaptive_verification`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L3095-L3122)：LoRA、full CUDA graph、PP>1）。
