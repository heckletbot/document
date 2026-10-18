# 异步调度

V1 的执行循环里有 CPU 同步点，异步调度是后补上去的：共享的 pinned buffer 用屏障守住，结果回传在主路径上停下来等，`dummy_run` 被四件事共用。V2 把核心循环定义成一条只排队、不等待的 CUDA stream，三件事分别拆掉。

## 竞态：屏障换成不共享

异步 H2D 期间，CPU 下一步会改同一块 pinned 内存。V1 的 `CpuGpuBuffer.cpu` 默认就是这块内存。

V1 用 `synchronize_input_prep()`（`gpu_model_runner.py`）把输入准备包进临界区：先 `prepare_inputs_event.synchronize()` 等上一步的拷贝结束，`yield` 出去做 CPU 写入，`finally` 里 `record()`。调用点在 `execute_model` 和 `_dummy_run`。三个代价：哪些 buffer 受保护没有机制约束，漏了就是静默竞态；所有 CPU 工作被关在同一个 `with` 里；`synchronize()` 这段时间 CPU 不能干别的。

V2 去掉屏障。`UvaBackedTensor.cpu` 设 `pin_memory=False`，CPU 只写它；`copy_to_uva()` 做一次 CPU→CPU 拷贝，拷进 `UvaBufferPool` 的下一个 pinned 槽，GPU 只读那个槽。读写落在不同内存上。正确性条件是一条：轮转深度 ≥ 在途 step 数。代价是每步一次 CPU 拷贝，加上池子的常驻内存。

这条约束只覆盖每步主路径（`execute_model` / `sample_tokens` / `prepare_*`）。初始化、profiling、捕获、退出仍然可以同步。

## 结果回传：同步挪到真的要用的时候

两边都把一步拆成 `execute_model` 和 `sample_tokens`，中间传入结构化输出的 grammar bitmask，并让 CPU 在 forward 期间做别的事。差别在采样结果回到 CPU 的方式。

V1 的簿记集中在 `_bookkeeping_sync()`，函数名里的 sync 是实的。`_to_list()` 的注释记着一次补丁（issue #22754）：直接 `tolist()` 会做设备级 stream 同步，卡住其他 stream 上的拷贝，PD 分离部署里成为每步关键路径上的性能问题。补丁改成 pinned 拷贝 + `transfer_event.synchronize()` + `tolist()`，同步还在主路径上。草稿 token 和有效 token 计数各有一套 `_copy_*_to_cpu` / `_get_*`。

V2 的 `AsyncOutput`（`gpu/async_utils.py`）在构造时切到 `copy_stream`：`wait_stream(main_stream)` 等采样算完，然后非阻塞 D2H（sampled ids、logprobs、num_sampled、sampling mask、prompt logprobs），`copy_event.record()` 后立刻返回。CPU 这时还读不到数据。`get_output()` 才 `copy_event.synchronize()`，再 `tolist()`。调度器真正要 token 之前，CPU 已经在准备下一步。

`copy_event` 用 `blocking=True` 创建，避免忙等去抢 CUDA driver 锁。拷贝和创建张量不在同一条 stream，所以 `AsyncOutput` 必须持有 GPU 张量引用，否则拷贝完成前张量可能被释放。

簿记改在 GPU 上做：`post_update` / `post_update_num_computed_tokens` 更新持久状态。NaN 计数和 EP 故障标志走同一批 D2H，在 `get_output()` 那一次同步里一起取回。

## CUDA graph 与 dummy_run

V1 的 `_dummy_run()` 约 420 行，同时服务显存 profiling、`torch.compile`、图捕获、warmup，再加上 EP+DP 空转。调用方往同一个函数里加参数，它和 `execute_model` 逐渐变成两套逻辑，行为不一致就出 bug。图的生命周期散在 `capture_model` / `_warmup_and_capture` / `_capture_cudagraphs` / `_check_and_update_cudagraph_mode`，运行时选哪种执行形态在 `_determine_batch_execution_and_padding`。

V2 拆成三条：

1. `execute_model` 接受 `dummy_run` 等参数，dummy 不改持久状态。`_dummy_run` / `profile_run` 只是调用它。
2. 图捕获走 `capture_model()` → `CudaGraphManager.capture()`，不经过 `execute_model`。
3. 一种执行形态是一个 `BatchExecutionDescriptor`（token 数、request 数、cg_mode、是否 uniform decode、活跃 LoRA 数、是否 micro-batch）。`_init_candidates` 枚举、`capture` 逐个捕、`dispatch` 用 `_is_compatible` 匹配。匹配不上就 padding 到某个已捕获的形状，或走 eager。

DP 下所有 rank 必须选同一个描述符，否则集合通信对不上。V2 把选图和 padding 协商放进同一次 `dispatch_cg_and_sync_dp()`。

显式的图还有一个直接结果。V1 每个草稿 token replay 一张图。V2 把 prefill 之后的全部草稿步捕进同一张 full graph：注意力元数据在循环前建好，步间变化的张量地址不变、原地更新。后端要提供 `AttentionMetadataBuilder.update_draft_decode_metadata()`，且每个 draft attention group 声明 `supports_draft_decode_metadata_update`，否则退回每步重建元数据。这个钩子在捕获期间运行，录进图的只有它发出的 GPU 操作，replay 时 Python 体不再执行，所以状态必须放在持久存储里。位置不变的草稿模型不需要这个更新。
