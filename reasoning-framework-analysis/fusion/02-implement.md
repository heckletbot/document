# 实现：编程模型、片上层次、tiling

阶段 2–4：定融哪些 op、数据走哪条路、用哪种 kernel、怎么切 tile。

---

## 选编程模型

```
有 Matmul？
 ├ 后面紧跟 Vector → __mix__(1,2)  CV
 └ 否则 → 纯 Cube
无 Matmul？
 ├ 规则密集 → SIMD __vector__ / __aicore__
 ├ 仅 3510 且稀疏/乱跳 → SIMT __global__
 └ 标量/IO → AI-CPU
```

Matmul **不要**走 SIMT。CV 只在解耦架构（2201+）。`__mix__(1,2)` 是 Vector 偏重的默认配比。

SIMD 标准节奏：CopyIn → Compute → CopyOut。`TQue` depth=2 才是 DoubleBuffer（copy≈compute 时 Vector 利用率才从约 1/3 拉起来）；depth=0 是高迭代 inplace。`EnQue/DeQue` 管写后读，`Alloc/Free` 管读后写。

---

## 片上层次（顺着说）

GM ⇄ L1 ⇄ L0A/L0B → Cube → L0C →（Fixpipe）→ UB → Vector → GM。

- CV 吃的是 **L0C→UB**，不是 Matmul 写 GM 再读回来做激活。  
- 通算吃的是通信轮次与 Matmul tile 重叠，不是这条 L0C 路径。  
- UB 每 AIV 大约 256KB，32 字节对齐。3510 有 Regbase 寄存器。

一个 AI Core：Scalar 发指令，AIV 做 Vector，AIC 做 Cube，MTE 搬数。耦合（2002/3002）Cube/Vector 同址；解耦后核间 2201 走 GM、3510 走 SSBuffer。

---

## Tiling 硬约束

| 层 | 切什么 |
|----|--------|
| 多核 | 全局 M/K/N → `singleCoreM/K/N` |
| 核内 | 片上能放下 → `baseM/baseN/baseK` |

CV：**`GetTensorC` 进 UB 时 `baseM * baseN * sizeof(dtype)` ≤ UB**。切太大，融合假设不成立，中间量回流 GM。

Host 用 `MultiCoreMatmulTiling`，buffer 用 `GetCoreMemSize`，`SetBufferSpace(-1,-1,-1)` 表示吃满硬件，禁止写死 256KB。Workspace = 用户块 + `GetLibApiWorkSpaceSize()`。变体用 `TILING_KEY_IS`（策略×dtype）。Tiling 函数可用 `ContextBuilder` 单独测。

Vector 计算优先连续 API（`Add(dst,src,count)`）；带 mask 的多维 API 会挡住编译器自动融合，Launch 又碎回去。

下一篇：[接到 vllm-ascend](03-integrate.md)。
