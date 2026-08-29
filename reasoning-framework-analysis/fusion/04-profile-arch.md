# Profiling 验证，以及换卡会不会把融合做废

阶段 5 和 7。融没融成功看 **GM 流量和流水线**，不看「kernel 名字里有 fused」。

---

## 两层 profiling

| 层 | 工具 | 用来找什么 |
|----|------|------------|
| 模型 | `VLLM_ASCEND_ENABLE_PROFILER` / `torch_npu.profiler` | 小 kernel 间隙、op 间大 GM 分配、AllReduce 后立刻算、是否像已知模式 |
| 算子 | `msprof op`（板）或 `msprof op simulator`（无卡，仅 SIMD） | 单 kernel 的 Cube/Vector、UB/GM、bank 冲突、CopyIn 重叠 |

对比方法固定：未融合基线 → 融合后再 profile。应看到 **kernel 数下降、GM 带宽下降、端到端时延下降**；Arithmetic 若原来被 launch/访存拖着，才可能升高。

---

## csv 怎么反证 Bound

| 文件 | 融合成功时 | 失败时先查 |
|------|------------|------------|
| Memory.csv | GM 降、UB 升 | tiling 太大 C 块进不了 UB；或根本没走融合路径 |
| PipeUtilization.csv | CopyIn/Compute/CopyOut 重叠 | 没开 DoubleBuffer；copy≪compute 则不必硬开 |
| ArithmeticUtilization.csv | Memory Bound 缓解后占比升；Matmul 里 Vector 仍极低 → 试 CV | 已经 Compute Bound 还融，这里救不了 |
| ResourceConflictRatio.csv | bank 冲突 &lt;5% | stride/padding；3510 换了 bank 结构必须重算 |
| L2Cache.csv | 重复访问命中尽量 &gt;80% | tile 太跳、复用差 |

优化顺序（影响从大到小）：DoubleBuffer → 避 bank 冲突 → 连续 Vector API → `uint16_t` 无分支硬件循环（VF）→ 拆依赖链双发射 → 高迭代 inplace → 小矩阵常驻 L1 → 调 `baseM/N/K`。

VF 自动融合要求：循环变量 `uint16_t`、从 0 步长 1、体内无 if、边界启动后不变。

---

## 架构：换代是换通路

| | 2201（910B） | 3510（910C/950） |
|--|--------------|------------------|
| 模式 | 解耦，核间 GM | 解耦+SIMT，AIC:AIV=1:2，核间 SSBuffer |
| 直连 | 有 L1→GM、GM→L0 | **都砍了**；必须 GM→L1→L0；新增 UB↔L1、L0C→UB |
| Vector | Membase | Regbase |
| L0A | FRACTAL_ZZ | FRACTAL_NZ |
| UB bank | 16×3×4KB | 8×2×16KB |
| BF16 | 有 | 有；310P（3002）没有 |

隔离三层不要混：kernel 宏（`__DAV_C310__` / `__NPU_ARCH__`）；host `GetSocVersion` + `GetCoreMemSize`；Python `is_310p()` / `get_ascend_device_type()`。高层 `Matmul`/`DataCopy`/`Fixpipe` 相对能跨代；BuiltIn / SIMT C **不跨代**。

soc 整数大致：200–205 310P，220–225 910B，250–255 910C，260 950。310P 不要开 BF16 融合 pass。

迁移时最容易说漏的：int4 Cube 先转 int8；次正规 `LnConfig`；删 `SetLoadDataBoundary`；bank 冲突按新结构重配 padding。

下一篇：[口头讲稿](05-speak.md)。
