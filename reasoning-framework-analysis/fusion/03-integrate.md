# 接到 vllm-ascend：A 调通、B 换层、C 改图

阶段 6。kernel 有了不等于 Serving 会走。三条路径从人接到自动。

| 路径 | 机制 | 何时 |
|------|------|------|
| A | `direct_register_custom_op` → `torch.ops.vllm.*` 调 `_C_ascend` | 先验证正确性和单算子加速 |
| B | `CustomOp.forward_oot` 在 NPU 上顶掉 `forward_cuda`，可回退 CANN | 融合固定在某一层（如 RMSNorm） |
| C | FX `pattern_matcher`：序列换成融合 op | 序列跨多个 op，靠编译期发现 |

口头纪律：A → B → C。C 的 pass 要开门控（架构、量化模式、`compile_range`）。`fake_impl` 必须返回对的 shape，否则 Dynamo 追踪挂。

---

## C++ 落地链（缺一环图上就没有）

`csrc/moe/<op>/`：`op_host`（OpDef、tiling、`EXEC_NPU_CMD` adapter）+ `op_kernel`（入口、模板、`arch35/`）。

然后：CMake `add_op_to_compiled_list` → `torch_binding.cpp` 里 `m.def` → Python 封装 →（可选）`graph_fusion_pass_manager` 注册 pass。

OpDef 要为每个目标架构 `AddConfig`（`ascend910b` / `ascend910_93` / 需要时 `ascend310p`）。kernel 入口用 `__DAV_C310__` 分 Regbase / Membase；大 hidden 用 `SetTilingKey` 走 split_d。

---

## 和模式怎么对应（举例）

- Residual+RMSNorm：路径 B（`layernorm.py`）  
- RMSNorm+Quant、AllReduce+RMSNorm、QK-RMSNorm+RoPE：路径 C（各 `*_fusion_pass.py`）  
- MoE gating / dequant_swiglu：主要是 C++ op + Python 在 MoE 层显式调（偏 A/B）  

图 pass 的 `get_pattern` / `get_replacement` 必须和真实图上的 op 名一致（例如已经是 `npu_add_rms_norm_bias` 再接 `quantize`，而不是再写一遍 naive RMSNorm）。

下一篇：[profiling 与跨架构](04-profile-arch.md)。
