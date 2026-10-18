# V1 → V2：三处改法

Model Runner V2 按第一性原理重写过一轮。官方九条主张（`docs/design/model_runner_v2.md`）收成三处改法，后面三篇各讲一处：V1 原先怎么做、为什么扛不住、V2 改成了哪段代码。

| 改法 | V1 卡在哪 | V2 改成什么 | 文档 |
|------|-----------|-------------|------|
| 持久批与输入准备 | 持久张量**就是**模型输入，请求进出要整表挪行、整块拷贝；输入在 Python 里拼 | 每请求固定一行，每步按顺序 gather；大表只传 diff；输入由 GPU kernel 填 | [02](02-persistent-batch.md) |
| 异步调度 | 异步是后补的：靠屏障挡竞态，主路径上多处同步，`dummy_run` 兼做四件事 | 核心循环不设同步点；竞态靠「读写不同内存」消除；图捕获独立出去 | [03](03-async.md) |
| 采样器与模块化 | 采样参数每步重建、投机解码要展开成 logits 形状；差异逻辑内联在 7200 行单文件里 | 每请求一行 + kernel 内间接寻址；差异收进可替换的类 | [04](04-sampler.md) |

代码位置：V1 是 `vllm/v1/worker/gpu_model_runner.py`，V2 是 `vllm/v1/worker/gpu/`。两个类都叫 `GPUModelRunner`。选哪个看 `VllmConfig.use_v2_model_runner`（`vllm/config/vllm.py`），实例化在 `gpu_worker.py` 的 `_make_model_runner()`。

## 什么时候跑 V2

`VLLM_USE_V2_MODEL_RUNNER` 不设时默认 V2。判定在 `use_v2_model_runner`：HiSparse、watermark 强制 V2；用户显式设置优先于自动判断；没装 Triton，或命中 `_get_v2_model_runner_unsupported_features()`，回退 V1。显式要求 V2 时同一份清单改走 `_validate_v2_model_runner()`，直接报错。

V2 还缺、会回退 V1 的：`external_launcher` 下的 PP、投机方法 `suffix` / `custom_class`、EAGLE 的 parallel drafting、异构词表 draft，以及开启 micro-batch 时 `_get_dbo_unsupported_features()` 列出的项。

只在 V2、V1 直接拒绝的：PCP、PP + 异步调度、DSpark、自适应验证、draft lm_head 量化、混合 sliding/full 的 DFlash、DFlash candidate-head（V1 上会静默退化成 DFlash1）、diffusion、batch-sharded sampling。清单在 `_get_v1_model_runner_unsupported_features()`，注释写明 MRV1 已废弃。
