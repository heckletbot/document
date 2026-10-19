# Async scheduling

代码版本：[vLLM `e80b6c6`](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8)。

Async scheduling 下，GPU 还在读上一 step 的 buffer 时，CPU 已经在准备下一 step。V1 用一次 `Event.synchronize()` 把这段重叠隔开。V2 让 CPU 和 GPU 读写不同的 buffer，主循环里不再为这个等。

## Input 侧

V1 的 [`synchronize_input_prep`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L3601-L3613) 整个函数就是这个 event：

```python
def synchronize_input_prep(self):
    if self.prepare_inputs_event is None:
        yield
        return
    # Ensure prior step has finished with reused CPU tensors.
    # This is required in the async scheduling case because
    # the CPU->GPU transfer happens async.
    self.prepare_inputs_event.synchronize()
    try:
        yield
    finally:
        self.prepare_inputs_event.record()
```

[`execute_model`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L3969-L3974) 用它包住 `_update_states`。`CpuGpuBuffer.cpu` 默认是 pinned 的，下一 step 会改同一块，所以上一 step 的 `copy_to_gpu` 必须先结束。这个 `with` 外面的写入不受它保护。

V2 没有 `synchronize_input_prep`。[`UvaBackedTensor`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L118-L124) 的 `cpu` 是 `pin_memory=False`。`copy_to_uva` 把内容拷进 `UvaBufferPool` 的下一个 pinned buffer，GPU 读那一份。Pool 有几个槽由 `max_concurrent_batches` 决定，见 [persistent batch](02-persistent-batch.md) 那篇。

`execute_model` 在 `dummy_run=False` 时才改 request 状态（[`第 1743–1750 行`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1743-L1750)）。`capture_model`、profiling、进程启动这些路径仍然会同步。

## Output 侧

V1 把 sampled token 变回 Python list 的函数是 [`_to_list`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L7149-L7162)。注释引用 issue #22754：直接 `tolist()` 会同步当前 device 上所有 stream，把别的 stream 里的拷贝也堵住。现在改成先拷到 pinned memory，但 `synchronize` 仍在这个函数里，sample 一结束就要等：

```python
pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
pinned.copy_(sampled_token_ids, non_blocking=True)
self.transfer_event.record()
self.transfer_event.synchronize()
return pinned.tolist()
```

哪些 request 结束、采了几个 token，集中在 [`_bookkeeping_sync`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L3463-L3488)。Draft token 和有效 token 数另有 `_copy_draft_token_ids_to_cpu`、`_copy_valid_sampled_token_count`。

V2 的 [`AsyncOutput.__init__`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/async_utils.py#L126-L171) 约 50 行，在 `copy_stream` 上做，不在这里等：

1. `copy_stream.wait_stream(main_stream)`，等 main stream 上的 sample 写完这些 tensor。
2. `async_copy_to_np` 拷 `sampled_token_ids`、logprobs、`num_sampled_tokens`、sampling mask、prompt logprobs。
3. `copy_event.record(copy_stream)` 后返回。

[`get_output`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/async_utils.py#L173-L184) 才等，然后再 `tolist()`：

```python
def get_output(self) -> ModelRunnerOutput:
    self.copy_event.synchronize()
    sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
    num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
```

`copy_event` 用 `torch.cuda.Event(blocking=True)` 创建。同函数注释写了两点：blocking 是为了避免忙等去抢 CUDA driver 的锁；必须留着 GPU tensor 的引用，因为拷贝在另一条 stream 上，引用丢了 tensor 可能先被释放。NaN 个数和 EP all2all 的 fault 标志也在这个 `__init__` 里发起 D2H，在同一次 `get_output` 之后读。

Request 的 `num_computed_tokens` 在 GPU 上改。[`_post_update_kernel`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L558-L616) 按 `idx_mapping` 找到 slot，把这一 step 接受的 token 数加进去。Python 入口是 [`post_update`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L618) 和 [`post_update_num_computed_tokens`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L675)。

## CUDA graph

V1 的 [`_dummy_run`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L5476-L5894) 从 5476 行到 5894 行。显存 profiling、`torch.compile`、CUDA graph capture、warmup 都进这个函数。Capture 的调用链是 `capture_model` → `_warmup_and_capture` → `_capture_cudagraphs`。运行时选 eager、piecewise 还是 full graph，在 `_determine_batch_execution_and_padding`。

V2 的 dummy forward 是 `execute_model(..., dummy_run=True)`。`dummy_run` 为真时跳过 request 增删和 `apply_staged_writes`（[第 1743–1750 行](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1743-L1750)）。Capture 单独在 [`capture_model`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1066-L1088)，里面交给 [`CudaGraphManager`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/cudagraph_utils.py#L150)。

一张图对应一个 [`BatchExecutionDescriptor`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/cudagraph_utils.py#L71-L85)：`cg_mode`、`num_tokens`、`num_reqs`、`uniform_token_count`、`max_query_len`、`num_active_loras`、`num_ubatches`。运行时 `dispatch` 用 `_is_compatible` 去对已捕获的 descriptor。Data parallel 的每个 rank 必须用同一个 descriptor，这次协商在 [`dispatch_cg_and_sync_dp`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/dp_utils.py#L211)。

多个 draft step 能不能进同一张图，由 [`_configure_decode_metadata_update`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/spec_decode/speculator.py#L308-L323) 决定。它扫每个 attention group 的 `supports_draft_decode_metadata_update`。有一个为假，就打日志，改成每 step 重建 attention metadata。设计文档 [第 9 节](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/docs/design/model_runner_v2.md) 写了 `update_draft_decode_metadata` 在 capture 时执行；replay 时只重放它发出的 GPU 操作，Python 函数体不再跑。
