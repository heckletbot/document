# fastokens 分词加速

- 链路：API Server / 引擎入口的 tokenizer **编码**（进 Prefill 之前）
- 收益：实测 TTFT ↓ 20ms；社区数据编码约 9~10×，长上下文 TTFT 最多约 40%
- 配置：`pip install fastokens==0.2.0`，`VLLM_USE_FASTOKENS=1`

## 问题怎么识别

HF tokenizer 编码走 Python/慢路径，长 prompt（均长 7.3k，可到 20~40k）时编码墙钟已经能在 TTFT 里看见。这是 **CPU 分词**，不是 NPU kernel。

社区新版本已集成 fastokens；调优时版本还没带上，所以是 **迁代码集成**，不是发明新分词器。

## 具体怎么改实现

加载分词器前，把 BPE 引擎换成 Rust 实现（fastokens），对上层 API 透明：

- **编码**走 fastokens
- **解码**仍回退 HF 原生，避开词表越界等边界

不改 prompt 规划、不改模型。开关用环境变量，可随时退回 HF。

## 学习要点

- 识别：把 TTFT 里「还没进 NPU」的一段单独计时，长序列时 tokenizer 会露出来。
- 实现：只换 encode 引擎，decode 保守回退。
