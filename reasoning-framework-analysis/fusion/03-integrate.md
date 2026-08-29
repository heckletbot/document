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

路径 B 注册名要用 `@CustomOp.register_oot(name="original_op_name")`，对上 vllm 原层名字，否则 `forward_oot` 顶不掉。

---

## 仓库里已经有的图 pass（先查再写新的）

| Pass | 融什么 | 文件 | 为何门控 |
|------|--------|------|----------|
| `AddRMSNormQuantFusionPass` | add-rms → quant | `norm_quant_fusion_pass.py` | 310P 无这条 BF16/量化链；还要 `enable_custom_op()` |
| `QKNormRopeFusionPass` | QKV split → Q/K RMSNorm → RoPE | `qknorm_rope_fusion_pass.py` | 只覆盖 `head_size==128` 且 BF16 |
| `MatmulAllReduceAddRMSNormPass` | Gemm → AllReduce → AddRMSNorm | `allreduce_rmsnorm_fusion_pass.py` | `compile_range.start > 512`，batch 太小重叠没意义 |
| `MulsAddFusionPass` | `x * scale + y` | `muls_add_pass.py` | 非 310P |
| `NoOpEliminationPass` | 删 no-op | `noop_elimination.py` | 总是；给后面 pattern 清场 |
| `AllGatherChunkNoopPass` | Chunk+AllGather 的空操作 | `allgather_chunk_noop_pass.py` | 总是 |

开口：新模型先说「这些 pass 开了没、门控过不过」，再决定是调开关还是新写 kernel。No-op 两类不是性能融合，但会改变图，配 pattern 时要按消掉之后的图来写。

---

## 新增时的 10 步（对应 A 必做，B/C 可选）

1. `csrc/moe/<op>/{op_host,op_kernel}`  
2. OpDef：输入输出属性 + 每架构 `AddConfig` + `OP_ADD`  
3. Tiling：硬件用 Platform API 读；设 BlockDim、TilingKey、workspace  
4. Kernel：`GET_TILING_DATA`、`TILING_KEY_IS`、`__DAV_C310__`、TPipe 做融合  
5. torch adapter：`EXEC_NPU_CMD`；多输出 `std::tuple`  
6. 两级 CMake：`add_op_to_compiled_list` / `add_subdirectory`  
7. `torch_binding.cpp`：`m.def("npu_…")`  
8. Python：`_impl` + `_fake` + `direct_register_custom_op`（路径 A）  
9. 可选：`CustomOp` + `forward_oot` + `@register_oot`（路径 B）  
10. 可选：`BasePattern` + `VllmInductorPass` + `GraphFusionPassManager.configure`（路径 C）  

1–8 缺一环 Python 调不通。9/10 决定 Serving 会不会自动走。步骤 10 的 `is_applicable_for_range` 就是 AllReduce+RMSNorm 那种 batch 门槛。

下一篇：[profiling 与跨架构](04-profile-arch.md)。
