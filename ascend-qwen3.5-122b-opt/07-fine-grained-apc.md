# 细粒度 APC（混合架构页对齐）

- 链路：Prefill 前缀缓存（Automatic Prefix Caching）
- 收益：TTFT ↓ 10ms+（TP4，前缀命中 1k）
- 开关：`VLLM_ASCEND_KV_CACHE_PAD_FACTOR`，默认 1；N=2 时 block 减半、粒度加倍
- 代价：PAD_FACTOR=2 时 KV 显存大约浪费一半
- 通用：所有 Hybrid（Attention + Mamba/GDN）模型

## 问题怎么识别

APC 能命中的最小前缀长度 ≈ `block_size`。Hybrid 架构里这件事被物理页对齐锁死：

- Full Attention：KV 按 token 分页
- GDN/Mamba：SSM 状态（conv state + recurrent state）也按页
- **两类 block 必须落在相同物理页大小上**

公式：

```text
attn_single_token_k_page_size × attn_block_size == ssm_block_page_size

block_size = cdiv(ssm_block_page_size,
                  kernel_block_size × attn_single_token_k_page_size)
```

`attn_single_token_k_page_size` 由模型参数决定，不能改。

部署从 **TP8 改到 TP4** 之后：

- 单卡上的 SSM/KV 布局变了
- Qwen3.5 TP=4 算出 `block_size = 2048`
- 前缀必须攒到 2048 token 才可能命中 → APC 形同虚设
- 原 TP8 上还能用的 APC，在 TP4 上直接不适配

这是第二阶段「客户要更高 QPS → 改 TP」连带打出来的，不是一开始就设计细粒度 APC。

## 具体怎么改实现

不能改 SSM 页，也不能改模型算出来的单 token K 页真实大小。做法是 **人为把 attn 侧单 token 页算大**：padding `attn_single_token_k_page_size`，同样物理页下 `attn_block_size` 按比例变小。

```text
原始:
  block_size = cdiv(ssm_page, kernel_block × attn_single_token_k_page_size)

引入 pad_factor = N:
  attn_single_token_k_page_size' = attn_single_token_k_page_size × N
  block_size' = block_size / N
```

N=2 → block 2048→1024，前缀共享最小 token 数减半，命中率上去。实现上就是多分配「空洞」KV 页：以显存换粒度。

环境变量 `VLLM_ASCEND_KV_CACHE_PAD_FACTOR` 注入这个 N，默认 1（行为与改前一致）。

## 学习要点

- 识别链路：SLO/QPS 压力 → TP8→TP4 → 代入页对齐公式 → `block_size=2048` → APC 命中崩。
- 实现不改注意力算法，只改 **页大小记账**：把 attn 单 token 页乘 N，逼 block 变细。
- 必须同时保证 Attention block 与 Mamba block 仍对齐到同一物理页，否则 Hybrid 无法 APC。
