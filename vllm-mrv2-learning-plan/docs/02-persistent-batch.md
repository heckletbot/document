# Persistent batch 和 input preparation

代码版本：[vLLM `e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8)。

Persistent batch 是跨 step 保存的请求状态。V1 把这批 tensor 直接当作这一 step 的 model input，所以 row 必须紧凑、顺序必须对。V2 把两件事拆开：`RequestState` 里每个 request 占一个固定 slot；这一 step 的 model input 是另组的 `InputBatch`，用 `idx_mapping` 从 slot 里 gather 出来。

## 为什么 V1 要 `condense`

Attention 读的就是 persistent batch 里 block table 的前 `num_reqs` 行。中间一个 request 结束，这一行空了，后面的 request 还停在原下标，和 model input 对不上。所以要把后面的 row 往前挪。

[`remove_request`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L523-L532) 的 docstring 要求调用后必须再调 `condense()`。[`condense`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L698-L730) 做的事：从高位找非空 request，填进低位的空槽。跟着搬的有：

- [`swap_states`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L580-L587)：交换两个下标上的 token 前缀
- 同一次 `condense` 里把 `generators[旧下标]` 改挂到新下标（[第 809–811 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L809-L811)）
- [`BlockTable.move_row`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/block_table.py#L137-L146)：在 numpy 上把一行 block id 拷到目标 row，源 row 清零

Row 会被别的 request 盖掉，所以 V1 另存一份 [`CachedRequestState`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L35-L74)。

## V2 的 slot 不挪

[`RequestState`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L28-L30) 用 `req_id_to_index` 和 `free_indices` 管 slot。[`add_request`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L99-L102) / [`remove_request`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L127-L134)：

```python
req_idx = self.free_indices.pop()
self.req_id_to_index[req_id] = req_idx
self.index_to_req_id[req_idx] = req_id
```

```python
req_idx = self.req_id_to_index.pop(req_id, None)
if req_idx is None:
    return None
self.index_to_req_id.pop(req_idx, None)
self.free_indices.append(req_idx)
return req_idx
```

没有 `condense`，也没有 `CachedRequestState`。

Preempt 和正常结束走同一条删除。[`finish_requests`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1165-L1176) 把 `preempted_req_ids` 并进 `finished_req_ids`，再对每个 id 调 `_remove_request`。这个 request 再次被调度时重新 `add_request`，拿到的是当时 `free_indices` 里的空 slot。

这一 step 里 request 的先后由 [`sort_batch_req_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L2456-L2468) 在 CPU 上排：带 draft token 的在前，再按 token 数。它排的是 req id 列表，不改 `RequestState`。

这一 step 用到的三块数据：

| 名字 | 生命周期 | 作用 |
|------|----------|------|
| [`RequestState`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py) | 跨 step | 每个 request 一个 slot |
| [`InputBuffers`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L19-L35) | 进程内复用 | 按 `max_num_tokens` 预分配的 GPU buffer，每 step 重写 |
| [`InputBatch`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L38-L50) | 只活一个 step | `idx_mapping[batch 下标] = slot` |

## 状态怎么写到 GPU

V1 只有 [`CpuGpuBuffer`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/utils.py#L139-L143)。`copy_to_gpu(n)` 拷前 n 行，不管这一 step 改了几格：

```python
def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
    cpu, gpu = self.cpu, self.gpu
    if n is not None:
        cpu, gpu = cpu[:n], gpu[:n]
    return gpu.copy_(cpu.pin_memory() if PIN_MEMORY else cpu, non_blocking=True)
```

Block table 的提交就是这一下，然后把同一张 GPU tensor 的前 n 行交给 attention（[`block_table.py`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/block_table.py#L181-L190)）：

```python
def commit_block_table(self, num_reqs: int) -> None:
    self.block_table.copy_to_gpu(num_reqs)

def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
    return self.block_table.gpu[:num_reqs]
```

V2 的 block table 不整表拷。[`append_block_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/block_table.py#L111-L130) 对每个 KV cache group 调 `stage_write(slot, start, block_ids)`，只记下这一 step 新加的 block id。[`apply_staged_writes`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/block_table.py#L132-L144)：一个 group 时调 `apply_write()`；多个 group 时用 `FusedStagedWriter.apply`，一次 kernel 写完。

`apply_write` 本身约 40 行（[`buffer_utils.py` 第 196 行起](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L196-L211)）。它做三步：把暂存的 row 下标、起点、内容打成三段连续 buffer；拷上 GPU；`_apply_write_kernel` 按这三段写回基 tensor。基 tensor 默认就在 GPU 上（`uva_instead_of_gpu=False` 时 `torch.zeros(..., device=device)`）。

Attention 不读这张基 tensor。[`gather_block_tables`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/block_table.py#L165-L176) 按 `idx_mapping` 把 slot 上的行拷到这一 step 的 output。Grid 是 `(KV cache group 数, row 数)`：

```python
_gather_block_tables_kernel[(self.num_kv_cache_groups, num_rows)](
    idx_mapping,
    self.block_table_ptrs,
    out_ptrs,
    self.block_table_strides,
    self.num_blocks.gpu,
    self.num_blocks.gpu.stride(0),
    num_reqs,
    BLOCK_SIZE=1024,
)
```

`prompt_len`、`prefill_len` 用 [`UvaBackedTensor`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L109-L124)。CPU 上的那份不 pin，每 step 拷进 pool 里的下一个 pinned buffer，GPU 读那一份：

```python
self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
self.np = self.cpu.numpy()
self.pool = UvaBufferPool(size, dtype, max_concurrency)
```

`prompt_len` 是用户 prompt 的 token 数。`prefill_len` 是送进 model runner 的 token 数，可以包含 preempt 之前已经生成、恢复时要重算的 token，所以 `prefill_len >= prompt_len`。Prompt logprobs 和 frequency penalty 要用 `prompt_len` 把 prompt 和 output 分开。这段 distinction 写在 [`states.py` 第 43–51 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L43-L51) 的注释里。

`all_token_ids` 的形状是 `(max_num_reqs, max_model_len)`。构造时 `uva_instead_of_gpu=not use_dense_all_token_ids`（[第 37–42 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L37-L42)）：默认让 GPU kernel 直接读 CPU 内存。Ngram 要反复扫这些 row，才改放到 GPU。

Pool 的槽数在 [`GPUModelRunner.__init__`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L242) 里设成 `max_concurrent_batches`，[`set_default_max_concurrency`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L24-L26) 再和 2 取最大。V2 且 async scheduling 时，`max_concurrent_batches` 等于 `pp_size + 1`（[`vllm.py` 第 578–588 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/config/vllm.py#L578-L588)）。

## 这一 step 的 `input_ids` 从哪来

V1 的 [`_prepare_inputs`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L1915) 有三百多行，在 CPU 上用 numpy 拼 `input_ids`、`positions`、`query_start_loc`，再 H2D。Async scheduling 下，上一步 sample 出的 token 还在 GPU 上，要先拷回 CPU 才能拼进下一步。这个回填是 [`update_async_output_token_ids`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_input_batch.py#L1038-L1042)，docstring 写的是：等 sampled token 拷完，再写进 `sampling_metadata.output_token_ids`。

V2 的 prefill 段是 [`_prepare_prefill_inputs_kernel`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L272-L303)，约 30 行。每个 program 处理 batch 里的一个 request：

1. `idx_mapping[batch_idx]` 得到 slot。
2. 读这个 slot 的 `prefill_len` 和 `num_computed`。`num_computed >= prefill_len` 就返回。
3. 从 `all_token_ids[slot]` 的 `num_computed` 处取出这一 step 的 query，写入 `input_ids`。

Decode 段是 [`_combine_sampled_and_draft_tokens_kernel`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L407-L464)，约 60 行。`seq_len <= prefill_len` 时直接返回。否则：

1. 用 `idx_mapping` 找到 slot。
2. 把 `last_sampled_tokens[slot]` 写入 `input_ids` 里这条 logits 的起点。
3. 若有 draft token，从 `draft_tokens[slot]` 写入 `input_ids` 的末尾。

`last_sampled_tokens` 和 `draft_tokens` 都是 GPU tensor。这个 kernel 不把它们读回 CPU。

`query_start_loc` 的前缀和仍在 [`prepare_inputs`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1438-L1448) 里用 numpy 算。选 CUDA graph 和 padding 发生在这些 kernel 之前，CPU 需要这个前缀和。

`num_computed_tokens` 有两份。[`states.py` 第 61–65 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/states.py#L61-L65)：GPU 上的 `StagedWriteTensor` 是实际值；旁边的 numpy 数组注释写的是这份 GPU 值的 upper bound，给 CPU 做调度用。
