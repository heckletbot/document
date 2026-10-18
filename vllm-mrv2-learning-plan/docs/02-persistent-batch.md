# 持久批与输入准备

V1 的持久行就是本步要喂给模型的行，所以中间有请求结束就必须把后面的行挪下来，block table 也按这个顺序整段拷上 GPU。V2 让行号在请求活着期间不变，本步的顺序只存在于 `idx_mapping`，再按它 gather。

## 行号

V1 的 [`remove_request`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L523-L532) 注释要求调用后必须 `condense()`。[`condense`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L698-L706) 把非空请求滑到低位空槽。换行时 [`swap_states`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L580-L587) 要搬 token 前缀，[`condense` 里](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L809-L811) 还把 `generators` 从旧下标挪到新下标。block table 跟着 [`move_row` / `swap_row`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/block_table.py#L137-L151)，在 numpy 上拷一行。持久行会被覆盖，所以另外有一份 [`CachedRequestState`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L35-L74)。

V2 的槽位是 [`req_id_to_index` / `free_indices`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L28-L30)。[`add_request`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L91-L102) 从 `free_indices` 取一个下标，[`remove_request`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L127-L134) 还回去。没有 `condense`，也没有 `CachedRequestState`。

```python
req_idx = self.free_indices.pop()
self.req_id_to_index[req_id] = req_idx
```

抢占和结束走同一条删除。[`finish_requests`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1165-L1176) 把 `preempted_req_ids` 并进 `finished_req_ids` 再逐个 `_remove_request`。恢复时重新 `add_request`，`free_indices.pop()` 给出的是当时的空槽，不一定是原来那一行。

本步顺序由 [`sort_batch_req_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L2456-L2468) 在 CPU 上排：有草稿的在前，再按 token 数。排的是本步的 req id 列表，不改 `RequestState` 的行。

V2 有三层。[`RequestState`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py) 跨步。[`InputBuffers`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L19-L35) 按 `max_num_tokens` 预分配、每步重填。[`InputBatch`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L38-L50) 是每步新建的 dataclass，`idx_mapping[batch_idx] = req_state_idx`。

## 写入

V1 的 [`CpuGpuBuffer`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/utils.py#L110-L143)：`cpu` 默认 pinned，另有同形状的 `gpu`。`copy_to_gpu(n)` 拷前 n 行。block table 的 [`commit_block_table`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/block_table.py#L181-L190) 就是这一下，然后 `get_device_tensor` 把 `gpu[:num_reqs]` 直接交给注意力。行序必须和模型输入一致，这是 `condense` 存在的原因。

V2 的 [`StagedWriteTensor.apply_write`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L196-L211) 只打包本步 `stage_write` 记下的下标、起点和内容，再进 `_apply_write_kernel`。block table 的 [`append_block_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/block_table.py#L111-L144) 对每一组调用 `stage_write`；一组时 `apply_write()`，多组时 `FusedStagedWriter.apply` 一次 kernel。基张量默认在 GPU（`StagedWriteTensor` 里 `uva_instead_of_gpu=False` 时 `torch.zeros(..., device=device)`）。

交给注意力的是另一张表。[`gather_block_tables`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/block_table.py#L146-L176) 用 `idx_mapping` 做下标，grid 是 `(num_kv_cache_groups, num_rows)`：

```python
_gather_block_tables_kernel[(self.num_kv_cache_groups, num_rows)](
    idx_mapping,
    self.block_table_ptrs,
    out_ptrs,
    ...
)
```

小列用 [`UvaBackedTensor`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L109-L128)：真值 `cpu` 是 `pin_memory=False`，`copy_to_uva` 拷进 `UvaBufferPool` 的下一个 pinned 槽。`prompt_len` / `prefill_len` 是这种。[`states.py` 的注释](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L43-L51) 写明 `prefill_len` 可以包含抢占前已经生成、恢复时重算的 token，所以 `prefill_len >= prompt_len`；prompt logprobs 和 frequency penalty 必须分开用。

`all_token_ids` 形状是 `(max_num_reqs, max_model_len)`，构造时 `uva_instead_of_gpu=not use_dense_all_token_ids`。注释说 ngram 会反复扫活跃行，所以那种情况改放 GPU。[`buffer_utils.py` 第 19–20 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L19-L26) 要求轮转深度不少于在途 step 数；[`GPUModelRunner.__init__`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L242) 传入的是 `vllm_config.max_concurrent_batches`（异步且 V2 时为 `pp_size + 1`，见 [`max_concurrent_batches`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L578-L588)），函数里再和 2 取最大。

## 本步输入

V1 在 [`_prepare_inputs`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L1915) 里用 numpy 拼，再 H2D。上一步采出的 token 要回到 CPU，对应 [`InputBatch.update_async_output_token_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L1038-L1046)。

V2 的 prefill 段在 [`_prepare_prefill_inputs_kernel`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L272-L303)：`program_id(0)` 是 batch 行，经 `idx_mapping` 找到 `req_state_idx`，从 `all_token_ids` 的那一行把 token 写入 `input_ids`。`all_token_ids` 走 UVA 时，这个指针就是 CPU 内存的 accelerator view。

decode 段在 [`_combine_sampled_and_draft_tokens_kernel`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L407-L464)。`seq_len > prefill_len` 时，把 `last_sampled_tokens[req_state_idx]` 和 `draft_tokens` 写进 `input_ids`。这两个张量都在 GPU 上，这个 kernel 不把它们读回 CPU。

`query_start_loc` 的前缀和仍在 [`prepare_inputs`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1438-L1448) 里用 numpy 算，因为选 CUDA graph、做 padding 发生在 kernel 之前。`num_computed_tokens` 同时有 GPU 上的 `StagedWriteTensor` 和 numpy 镜像，[`states.py` 第 64 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L64-L65) 把 numpy 那份标成 GPU 值的上界。
