# 从日志到代码：先点名文件再谈改法

定位用当前工作区里的 vLLM（或 vllm-ascend 衍生树）。路径随版本会挪，开口用模块职责，核对时再对具体文件。

| 现象 | 优先翻 | 常见文件（upstream 口径） |
|------|--------|---------------------------|
| 组 batch、抢占、waiting | Scheduler | `vllm/core/scheduler.py` 或 `vllm/scheduler/` |
| 命中/换出/block 不够 | KV / Block | `vllm/worker/cache_engine.py`、block manager |
| `execute_model`、P/D 一步里干什么 | Model Runner | `vllm/worker/model_runner.py` |
| NPU 适配、图编译开关 | 平台层 | `vllm/ascend/` 或下游 `vllm_ascend/` |
| 采样默认值 | Sampling | `vllm/sampling_params.py` |
| 多模态下图、解码、线程池 | 预处理 / 补丁 | 模型的 image input 路径；下游常有独立 patch |

输出固定四段，且第四段默认停住：

1. 根因（Bound + 路径上的哪一跳）  
2. 文件和函数  
3. 改思路（逻辑层：少一次 IO、加大线程池、别在 decode 里同步下图）  
4. **确认后再改代码**

改完用同一套 benchmark 回归；只动 Prefill 补丁时，不必无故重启 Decode（PD 分离时按改到的进程杀端口）。

NPU 环境噪音（不是算法问题，但日志会误导）：

- `npu-smi` 仍占显存但 `/proc/<pid>` 没了 → 驱动残留，考虑对该卡 `device-reset`（先确认卡上没有别的服务）。  
- Decode 报 free memory 不足 → 先看 HBM 是否被残留抬高。  
- RouteServer 健康检查 500 → 先看 P/D 端口还在不在，RS 是否在 P/D 重启后没拉起来。

下一篇：[预处理案例](03-case-preprocess.md)。
