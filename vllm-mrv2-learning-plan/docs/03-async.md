# 异步调度

V1 的异步是在已有循环上补的：共享 pinned buffer 用 event 挡住，采样结果在主路径上拷回 CPU。V2 把「CPU 写的内存」和「GPU 读的内存」拆开，主路径上的同步留到调度器取结果的时候。

## 输入侧

V1 的 [`synchronize_input_prep`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L3601-L3613)：

```python
# the CPU->GPU transfer happens async.
self.prepare_inputs_event.synchronize()
try:
    yield
finally:
    self.prepare_inputs_event.record()
```

[`execute_model`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L3969-L3974) 用它包住 `_update_states`。`CpuGpuBuffer.cpu` 默认 pinned，下一步会改同一块，所以上一步的异步拷贝必须先结束。屏障只覆盖 `with` 里面的写入，没有类型或接口要求某个 buffer 必须进这个块。

V2 没有这个函数。[`UvaBackedTensor`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/buffer_utils.py#L109-L128) 的真值不 pin，`copy_to_uva` 写到池子里的另一个槽，GPU 读槽。槽的个数由 `max_concurrent_batches` 决定，见上一篇。初始化、profiling、`capture_model` 这些路径仍然可以同步；`execute_model` 在 `dummy_run=False` 时才改请求状态，见 [`第 1743–1750 行`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1743-L1750)。

## 输出侧

V1 的 [`_to_list`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L7149-L7162) 注释写着 issue #22754：直接 `tolist()` 会做设备级 stream 同步，卡住别的 stream 上的拷贝，在分离式部署里落在每步关键路径上。现在的写法仍是主路径上的同步：

```python
pinned.copy_(sampled_token_ids, non_blocking=True)
self.transfer_event.record()
self.transfer_event.synchronize()
return pinned.tolist()
```

簿记在 [`_bookkeeping_sync`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L3463-L3488)。草稿和有效 token 数各有 `_copy_draft_token_ids_to_cpu`、`_copy_valid_sampled_token_count`。

V2 的 [`AsyncOutput.__init__`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/async_utils.py#L115-L171) 在 `copy_stream` 上做。先 `copy_stream.wait_stream(main_stream)`，再 `async_copy_to_np` 拷 `sampled_token_ids`、logprobs、`num_sampled_tokens`、sampling mask、prompt logprobs，最后 `copy_event.record(copy_stream)`。构造函数返回时这些 numpy 还不能读。[`get_output`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/async_utils.py#L173-L184) 才 `copy_event.synchronize()`，然后 `tolist()`。

`copy_event` 用 `torch.cuda.Event(blocking=True)` 创建，注释说是为了避免忙等去抢 CUDA driver 锁。注释同时要求保住 GPU 张量引用，因为拷贝发生在创建张量的另一条 stream 上。NaN 计数和 `check_ep_fault` 的 fault 标志也在这个 `__init__` 里发起 D2H，在同一次 `get_output` 同步之后读。

持久状态的步末更新在 GPU kernel [`_post_update_kernel`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L558)（由 [`post_update`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L618) 启动）和 [`post_update_num_computed_tokens`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/input_batch.py#L675)。这条路径不经过 `_bookkeeping_sync`。

## CUDA graph

V1 的 [`_dummy_run`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu_model_runner.py#L5476) 从 5476 行接到 5894 行，profiling、编译、捕获、warmup 都进这个函数。捕获链是 `capture_model` → `_warmup_and_capture` → `_capture_cudagraphs`。运行时形态在 `_determine_batch_execution_and_padding`。

V2 把 dummy 收进 `execute_model` 的 `dummy_run` 参数：上面引用的 `if not dummy_run` 跳过请求增删和 `apply_staged_writes`。捕获在 [`capture_model`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/model_runner.py#L1066-L1088)，进去之后交给 [`CudaGraphManager`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/cudagraph_utils.py#L150)。

一种形态是一个 [`BatchExecutionDescriptor`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/cudagraph_utils.py#L71-L85)：`cg_mode`、`num_tokens`、`num_reqs`、`uniform_token_count`、`max_query_len`、`num_active_loras`、`num_ubatches`。运行时 `dispatch` 用 `_is_compatible` 对上已捕获的描述符。DP 各 rank 必须同一描述符，协商在 [`dispatch_cg_and_sync_dp`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/dp_utils.py#L211)。

草稿多步能否收进一张图，由 [`DraftModelSpeculator._configure_decode_metadata_update`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/vllm/v1/worker/gpu/spec_decode/speculator.py#L308-L323) 决定：任何一个 attention group 的 `supports_draft_decode_metadata_update` 为假，就打日志并退回每步重建 metadata。设计文档 [`第 9 节`](https://github.com/vllm-project/vllm/blob/e80b6c642f21fd5b24439825592f5b9ced36add8/docs/design/model_runner_v2.md) 写了这个钩子在捕获期执行，replay 时只重放它发出的 GPU 操作。
