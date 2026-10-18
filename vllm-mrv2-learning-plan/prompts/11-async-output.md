# 11 · 异步输出与跨 step 流水

目标：读懂 MRV2 怎么把「采样结果回传 CPU」这件事从主路径上摘出去——唯一的同步点被推迟到调度器真正需要结果的那一刻。对应设计文档第 2 条（Async-First）。

---

## 要看的内容

1. `vllm/v1/worker/gpu/async_utils.py` 全文（不长，建议逐行）
   - `AsyncOutput.__init__` / `get_output()`
   - `AsyncPoolingOutput`
   - `async_copy_to_np()`、`stream()` 上下文管理器
   - `StepTimingCollector` / `StepTimingEvents` / `StepTimingSample`
2. `vllm/v1/worker/gpu/model_runner.py` · `sample_tokens()` 末尾构造 `AsyncOutput` 的地方、`pool()`、`main_stream` 属性
3. `vllm/v1/worker/gpu/buffer_utils.py` · `set_default_max_concurrency()`（轮转深度 ↔ 在途 step 数）
4. 调用侧：engine core 里消费 `AsyncModelRunnerOutput` 的地方（搜 `get_output`）

---

## 要掌握的知识

### `AsyncOutput` 的双流结构

构造时在 **copy_stream** 上发起一批 D2H 异步拷贝，并 `record` 一个 event；`get_output()` 才 `synchronize` 这个 event 然后把数据转成 Python list。

```text
构造（在 copy_stream 上）
  copy_stream.wait_stream(main_stream)     # 等 main_stream 上的采样算完
  非阻塞 D2H：sampled_token_ids / logprobs / num_nans /
             num_sampled_tokens / sampling_mask / prompt_logprobs ...
  copy_event.record(copy_stream)

get_output()（CPU 真正要用的时候）
  copy_event.synchronize()                  # ← 全流程唯一的同步点
  .tolist() 转换 + 按 num_sampled_tokens 截断
```

三个必须注意的细节：

1. **必须持有 GPU 张量的引用**。代码注释写明：拷贝发生在和创建张量不同的 stream 上，不保留引用可能被提前释放。这是跨 stream 编程的经典坑。
2. `copy_event` 用 `blocking=True` 创建，注释说明是为了**避免 busy-poll CUDA driver lock**——忙等会抢锁，拖慢正在排队的其他 CPU 工作。
3. `get_output()` 里的注释直说当前把 numpy 转 list 只是为了兼容旧 model runner 接口，**理想形态是一路保持 numpy**。读的时候知道这是过渡态就行。

### 为什么这样就能流水起来

`sample_tokens()` 返回 `AsyncOutput` 后立刻返回，CPU 马上可以去准备第 N+1 步的输入；GPU 还在跑第 N 步的拷贝。只有调度器真的要用第 N 步的 token 时才会 `get_output()` 阻塞。

这也解释了第 05 章 `UvaBufferPool` 轮转深度的约束：**在途 step 数 = 池子最小深度**。`set_default_max_concurrency()` 就是把 engine 的 `batch_queue_size` 传下来的通道。两个机制是配套的。

### 顺带夹带的两件事

- **EP 故障检查**：`check_ep_fault` 为真时查询 all2all 的故障标志，也走同一个异步拷贝，在 `get_output()` 里 `.item()` 后决定要不要抛错。把一个本来需要同步的健康检查塞进已有的同步点，是很典型的优化手法。
- **NaN 检查**：`num_nans` 同理，配合 `VLLM_RAISE_ON_LOGIT_NANS`。

### `StepTimingCollector`

用 CUDA event 对 forward 和 drafter 分别计时（`forward_start/end`、`drafter_start/end`），`collect()` 是个生成器。注意它也是**异步**的——event 的 elapsed time 要等 event 完成才能读，所以采集和消费是分开的。

### `AsyncPoolingOutput`

pooling / embedding 模型的对应物，结构同构但输出是 `PoolerOutput` + `finished_mask`。

---

## Prompt

```text
你是 vLLM 源码导读助手。仓库在 <VLLM_REPO>。
本轮只看 Model Runner V2，即 vllm/v1/worker/gpu/ 这个包。

本轮任务：讲透 MRV2 的异步输出机制和跨 step 流水。

第一部分 · AsyncOutput
1. 逐行讲 AsyncOutput.__init__（vllm/v1/worker/gpu/async_utils.py）：
   - stream(copy_stream, main_stream) 这个上下文管理器做了什么？
   - copy_stream.wait_stream(main_stream) 这一行在等什么？不加会怎样？
   - 列出所有被异步拷贝回 CPU 的字段，每个说明它的来源和用途。
   - copy_event.record(copy_stream) 之后，构造函数就返回了。此刻 CPU 能用这些数据吗？
2. 照注释解释"We must retain references to the GPU tensors"这条：
   不保留引用具体会发生什么？为什么跨 stream 会引入这个问题？
3. 照注释解释 copy_event 为什么用 blocking=True 创建：
   busy-poll CUDA driver lock 会造成什么后果？
4. 逐行讲 get_output()：
   - copy_event.synchronize() 是不是整个 step 流程里唯一的同步点？
     请在 execute_model + sample_tokens + get_output 的全路径上验证这个说法。
   - sampled_token_ids 按 num_sampled_tokens 截断这一步在做什么（为什么会有多余的）？
   - 注释说"Going forward, we should keep the data structures as NumPy arrays
     rather than Python lists"，说明现在为什么还不能这么做。
5. 讲 async_copy_to_np() 的实现，以及它和直接 .cpu() 的区别。

第二部分 · 流水语义
6. 画一张时序图：连续三个 step 里，CPU 的 prepare、GPU 的 forward、copy_stream 的回传、
   以及调度器调用 get_output() 的时刻，标出它们怎么重叠。
7. 在 engine 侧找到消费 AsyncModelRunnerOutput 的代码（搜 get_output 的调用方），
   说明调度器在什么时机被迫阻塞。
8. 把这个机制和第 05 章 UvaBufferPool 的轮转深度联系起来：
   - set_default_max_concurrency() 被谁调用、传的是什么值？
   - 为什么"在途 step 数"同时决定了缓冲池深度和异步输出的并发度？
   - 如果 batch_queue_size 调大但池子没跟着变深，会出现什么症状？

第三部分 · 夹带的检查
9. 讲 check_ep_fault 这条路径：它查什么、为什么能"搭车"到已有的同步点上、
   get_output() 里 _has_fault.item() 之后的处理逻辑。
10. 讲 num_nans 和 VLLM_RAISE_ON_LOGIT_NANS 的配合，以及
    envs.VLLM_COMPUTE_NANS_IN_LOGITS 默认关闭的原因。
11. 讲 pending_aux_output 这条路径（enqueue_cpu_copy / process_output）。

第四部分 · 计时与 pooling
12. 讲 StepTimingCollector / StepTimingEvents / StepTimingSample：
    用 CUDA event 计时为什么也必须是异步采集的？collect() 作为生成器的消费时机是什么？
    record_batch() 记录了哪些批特征？
13. 讲 AsyncPoolingOutput 和 model_runner.py 的 pool()，
    说明它和 AsyncOutput 的同构之处与差异。

约束：
- 只依据这个仓库里的真实代码回答，不要凭记忆补全 vLLM 历史版本的行为。
- 每个结论都要给出 文件路径:行号 和函数名；拿不准就直接说"代码里没找到"。
- 每次提到一个操作时，明确它跑在 main_stream 还是 copy_stream 上。
- 先给 10 行以内的结论，再展开细节。
- 不要贴超过 40 行的大段源码，只贴关键几行并解释。
```

---

## 自测题

1. `AsyncOutput` 构造完成的那一刻，CPU 能读到采样结果吗？为什么？
2. `copy_stream.wait_stream(main_stream)` 在等什么？去掉会出什么错？
3. 为什么必须保留对 GPU 张量的引用？这个坑和「跨 stream」有什么关系？
4. `copy_event` 为什么用 `blocking=True`？忙等会伤害谁？
5. 整个 step 流程里唯一的 CPU-GPU 同步点在哪一行？
6. `get_output()` 为什么要按 `num_sampled_tokens` 截断 `sampled_token_ids`？多出来的是什么？
7. 调度器在什么时刻会被迫等待 GPU？怎么让这个等待尽可能晚发生？
8. 在途 step 数同时约束了哪两个机制？两者不一致会怎样？
9. EP 故障检查为什么能「搭车」而不是自己开一个同步点？
10. 用 CUDA event 测 forward 耗时，为什么采集和读取必须分开？
