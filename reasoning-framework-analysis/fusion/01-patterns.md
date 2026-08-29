# LLM 融合模式：按 Bound 选，不按名字背

每个模式先问三句：融哪几个 op、中间谁不再进 GM、在 vllm-ascend 是层替换还是图 pass。完整目录对应材料里的 13 条；下面按瓶颈归类，便于开口。

---

## 内存带宽 / 中间物化

同一块 hidden 扫两遍以上，或全精度中间量写 GM 再读回。

| # | 融合 | 要点 | vllm-ascend |
|---|------|------|-------------|
| 1 | Residual + RMSNorm | 两阶段：add+方差，再 rsqrt×weight；写出 new residual | `npu_add_rms_norm_bias`；`AscendRMSNorm.forward_oot` |
| 2 | SwiGLU | 读拼接 `[gate,up]`，寄存器里 `silu*up` | 多用 `npu_swiglu`；step+mul 有 Triton 变体 |
| 3 | RMSNorm + Quant | 归一化后在 UB 上直接量化，省全精度中间量 | 图 pass：`npu_add_rms_norm_bias→quant` → `npu_add_rms_norm_quant`；W4A4 / 310P 关掉 |
| 4 | 激活 + Quant | 同 #3，激活链一次 pass | 内核或图，看序列是否在一层里 |
| 7 | RoPE + KV 写 | 旋转后的 K/V 直接进 cache | 少一次 RoPE 输出物化 |
| 10 | Dequant + SwiGLU + Quant | MoE 专家：int8 权重解量化 → bias → SwiGLU → 再量化 | `dequant_swiglu_quant`（910B/910C）；950 另有 `swiglu_group_quant` |
| 12 | Attention 输出 + Quant | attention 写出 FP8，不落全精度 | 写路径里带量化 |

#1 的 tiling key 要能举例：general(10)、split_d(11)、merge_n(12)…；dtype 后缀 1x=FP16、3x=BF16。Hidden 很大走 split_d，不要只会说「有个融合 op」。

---

## Launch 过碎

一串 &lt;50µs 的小 kernel，间隙比计算还显眼。

| # | 融合 | 要点 | vllm-ascend |
|---|------|------|-------------|
| 6 | split QKV + Q/K RMSNorm + RoPE | 按 (token, head) 在片上做完 | `split_qkv_rmsnorm_rope`；常门控 `head_size==128` 且 BF16 |
| 8 | MoE TopK 路由 | softmax→topk→renorm→索引一次做完 | `moe_gating_top_k` |
| 9 | MoE 专家分发 | permute / grouped GEMM / 激活 / unpermute | `moe_init_routing_custom`（大量 tiling 变体） |

#9 的瓶颈常常是 **permute 的带宽 + GEMM 间激活物化**，不是再融一个 softmax 就能好。

---

## 通信挡计算

| # | 融合 | 要点 | 门控 |
|---|------|------|------|
| 5 | AllReduce + add + RMSNorm | 分块到达就做 norm，与剩余通信重叠 | 图 pass 匹配 Gemm→AllReduce→add_rms；`compile_range.start > 512` 才上（batch 太小重叠没意义） |
| 13 | MC² / 通算 | AllGather+Matmul 按 M 维切分重叠 | `Hccl`+`Matmul` 高层 API，**不能 kernel 直调**；`flash_comm_v1_enabled` 一类开关 |

Serving 里还常见：AllGather 与 unpad、pad 与 AllReduce、matmul 与 reduce 绑在同一 Python 包装里。这是通算/通信包装，不要和 `__mix__` CV 混谈。

---

## Matmul + 逐元素：CV

| # | 融合 | 数据流 |
|---|------|--------|
| 11 | Matmul + ReLU/GELU 等 | A×B→L0C→Fixpipe→UB→激活→GM |

编程模型用 `__mix__(1,2)`，`GetTensorC<true>` 进 UB。`baseM*baseN*dtype` 必须装进 UB，否则这条路径作废。

---

## 开口对照表

| 你看到的 | 先想哪些模式 |
|----------|----------------|
| Arithmetic 低、GM 带宽高、同一 hidden 扫两遍 | #1 #2 #3 #4 #7 #10 |
| trace 里一串小 kernel | #6 #8 #9 |
| AllReduce 后立刻 RMSNorm / GEMM | #5 #13 |
| Matmul 的 Vector 利用率接近 0 | #11 |
| MoE 专家 int8 路径 | #8 #9 #10 |

下一篇：[实现与 tiling](02-implement.md)。
