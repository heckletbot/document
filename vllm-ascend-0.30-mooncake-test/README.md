# 在 vLLM-Ascend 0.30 上测自己的 Mooncake

目标：在 **vLLM-Ascend v0.30.0**（对齐上游 vLLM v0.30.0）上，先跑通官方 PD 直传，再把进程里的 Mooncake 换成自己编译的那一份，用同一套请求确认传输还在走。

对照源码标签：[vllm-project/vllm-ascend `v0.30.0`](https://github.com/vllm-project/vllm-ascend/tree/v0.30.0)（稳定版，2026-10-29）。接口背景见 [KV Connector V1](../kv-connector-v1/README.md)；Mooncake 概念见 [学习计划](../mooncake-learning-plan/README.md)。

官方部署原文（比本文长，多机、大模型、KV 池参数都在里面）：

- 单机 1P1D：[pd_disaggregation_mooncake_single_node.md](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md)
- 多机：[pd_disaggregation_mooncake_multi_node.md](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md)
- KV 池：[kv_cache_pool_ascend_store.md](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md)

---

## 先分清测的是哪条路

0.30 里 Mooncake 出现在两条互不替代的路径上。第一次冒烟只走第一条。

| 路径 | 配置名 | 实现 | 数据怎么走 |
|------|--------|------|------------|
| PD 直传（本教程） | `MooncakeConnectorV2` | `vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector.MooncakeConnector` | Decode 发起、异步把 Prefill 的 KV 拉到 Decode 的 paged buffer |
| PD 直传（旧实现） | `MooncakeConnectorV1` | `kv_p2p/mooncake_connector.py` 里的 `MooncakeConnector` | 同样是 P2P；要在 `kv_connector_extra_config` 里写 P/D 的 `dp_size` / `tp_size` |
| 分层推送 | `MooncakeLayerwiseConnector` | `mooncake_layerwise_connector.py` | Prefill 按层 push。冒烟先不要用 |
| KV 共享池 | `AscendStoreConnector`（别名 `MooncakeConnectorStoreV1`） | `kv_pool/ascend_store` | 经 `mooncake_master` Put/Get，按 block hash 复用 |

`vllm_ascend/distributed/kv_transfer/__init__.py` 里 `register_connector()` 把这些名字注册进 `KVConnectorFactory`。Worker 真正 import 的是：

```python
from mooncake.engine import TransferEngine
engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
```

协议字符串是 **`ascend`**。自己的构建必须提供这个 Python 模块，并且 `initialize(..., "ascend", ...)` 返回 0。

---

## 版本钉在 0.30.0

发行说明里和这次测试绑在一起的依赖（v0.30.0rc1 的 Dependencies，稳定版沿用这条线）：

| 组件 | 版本 |
|------|------|
| vLLM-Ascend | **v0.30.0** |
| vLLM | **v0.30.0** |
| CANN | 9.2.0 |
| Ascend HDK | 26.0.RC1 |
| PyTorch / torch_npu | 2.10.0 / 2.10.0.post4 |
| Python | 3.12 |
| 镜像内 Mooncake wheel | **mooncake-transfer-engine-npu 0.3.11.post1** |

官方单机教程正文仍写着 `git clone -b v0.3.9`。那是文档里的编译示例，和发行说明钉的 wheel 不是同一份。基线用镜像里已经装好的 **0.3.11.post1**，确认请求能通，再把自己的构建盖到 `PYTHONPATH` / `LD_LIBRARY_PATH` 前面。不要在还没跑通基线时先按 v0.3.9 重编一套。

KV 池文档对 Mooncake 还有更细的下限：默认租户 `>= 0.3.11.post1`；非默认租户 `>= 0.3.12`；layerwise 池 `>= 0.3.14`。PD 直传冒烟不走这些。

---

## 1. 起容器

同机 1P1D 只要两张 NPU。下面只挂 `davinci0`、`davinci1`。网卡名、IP、模型路径换成自己的。镜像用发行标签 `v0.30.0`：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0
docker run --rm -it \
  --name vllm-ascend-030 \
  --net=host \
  --shm-size=16g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /path/to/models:/models \
  -v /path/to/your/mooncake:/opt/mooncake-src \
  "$IMAGE" bash
```

容器里先确认三件事：

```bash
cat /etc/hccn.conf
python -c "import vllm, vllm_ascend, mooncake.engine; print(vllm.__version__)"
python -c "import mooncake.engine as e; print(e.__file__)"
```

`hccn.conf` 必须在容器里看得到。`mooncake.engine` 这条路径记下来，后面换成自己的包时要变。

同机冒烟不必先做跨机 `hccn_tool -ping`。要上第二台机器时，再按官方单机/多机教程把链路、TLS、NPU IP 查完：A2 用 `hccn_tool -i $i -ip/-ping`，A3 用 `-vnic` / `-hccs_ping`。

---

## 2. 基线：同机 1P1D，MooncakeConnectorV2

模型用 **Qwen3-8B**（纯 GQA，一张卡一个角色）。P、D 必须是同一份权重。GQA 在 V2 上只验证过 `LBHNC` 和 `HND`，两边设成同一个值。

端口：V2 的 KV 端口不是 HTTP `--port`。每个实例从 `kv_port` 起连续占用 `DP * PP * PCP * TP + DP` 个端口。这里全是 1，所以每边 2 个：P 用 `30000–30001`，D 用 `30100–30101`。HTTP 分别是 `8100` / `8200`。

三个进程，三个终端。把 `192.168.0.1` 和 `eth0` 换成 `ip -br addr` 里这台机器的地址和网卡。

Prefill：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_HOST_IP=192.168.0.1
export HCCL_IF_IP=192.168.0.1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export VLLM_KV_CACHE_LAYOUT=LBHNC
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800

vllm serve /models/Qwen3-8B \
  --host 0.0.0.0 --port 8100 \
  --served-model-name qwen3 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":30000}'
```

Decode：同一组环境变量，只改设备和端口。

```bash
export ASCEND_RT_VISIBLE_DEVICES=1
# VLLM_HOST_IP / HCCL_IF_IP / 网卡 / VLLM_KV_CACHE_LAYOUT 与 Prefill 相同

vllm serve /models/Qwen3-8B \
  --host 0.0.0.0 --port 8200 \
  --served-model-name qwen3 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":30100}'
```

两边都出现服务就绪后，在 vllm-ascend 源码根目录起代理（脚本在标签 `v0.30.0` 的 `examples/disaggregated_prefill_v1/`）：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts 192.168.0.1 --prefiller-ports 8100 \
  --decoder-hosts 192.168.0.1 --decoder-ports 8200
```

请求打到代理，不要打到 8100 或 8200：

```bash
curl http://192.168.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"用一段话说明 prefill 和 decode 的差别。"}],"max_tokens":128,"temperature":0}'
```

基线算通过，需要同时看到：

1. 代理返回一段正常 completion，`finish_reason` 不是 error。
2. Prefill、Decode 日志里都创建了 connector（工厂日志里的类名是 `MooncakeConnector`，配置名仍是 `MooncakeConnectorV2`）。
3. 没有 `TransferEngine initialization failed`，也没有 `Please install mooncake`。
4. Decode 没有把这次请求记成 KV load 失败。V2 会把失败交给 `kv_load_failure_policy`（默认 `fail`，请求直接失败）。

`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 的单位是秒，默认不够长请求时 Prefill 会提前丢掉还没被 Decode 拉走的 block。冒烟 1800 足够。把它调大只会让失败的 Decode 更久占着 Prefill 的 HBM，不改变正确请求的结果。

---

## 3. 换成自己的 Mooncake

vLLM 和 vLLM-Ascend 保持 0.30.0 不动。只替换 `mooncake.engine`。

编译沿用 0.30 教程里的 Ascend 开关（自己的分支如果改了 CMake 选项名，以该分支的 `CMakeLists.txt` 为准，运行时协议字符串仍然是 `ascend`）：

```bash
cd /opt/mooncake-src
bash dependencies.sh -y          # 容器里已有依赖时可跳过
mkdir -p build && cd build
cmake .. -DUSE_ASCEND_DIRECT=ON
cmake --build . -j"$(nproc)"
cmake --install .
```

安装前缀和镜像自带的 wheel 往往会叠在一起。启动 P/D 之前把**自己的** Python 包和 `.so` 放到前面，并用 import 路径确认：

```bash
export PYTHONPATH=/usr/local/lib/python3.12/site-packages:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/usr/local/lib64/python3.12/site-packages/mooncake:/usr/local/lib64:${LD_LIBRARY_PATH:-}
python - <<'PY'
import mooncake.engine as e
print(e.__file__)
from mooncake.engine import TransferEngine
print("TransferEngine", TransferEngine)
PY
```

打印出的文件必须落在你刚安装的目录里。仍指向镜像里的 `mooncake-transfer-engine-npu` 时，这次测试没有用到自己的代码。

然后**原样重跑**第 2 节的 Prefill、Decode、代理和 curl。通过标准和基线相同，再加一条：import 路径是自己的构建。

同一台机器上不要留两套 `libmooncake` 同时被加载。P 和 D 必须指向同一份构建。

---

## 4. 冒烟失败时看哪里

| 现象 | 先查 |
|------|------|
| `Please install mooncake` / `ImportError: mooncake.engine` | `PYTHONPATH` 没指到带 `mooncake/engine` 的目录 |
| `TransferEngine initialization failed with ret_value` | 协议后端不是 Ascend 构建，或 NPU 设备不可见。`initialize` 的第三个参数固定是 `ascend` |
| `Mooncake memory registration failed` | KV tensor 没注册成功。先确认基线 wheel 能过，再看自己的 `register_memory` |
| 一边起来、另一边一直等 | `kv_port` 撞了，或网卡环境变量不一致。同机两边必须不同的 `kv_port` 区间 |
| HTTP 200 但内容明显胡话，且 Decode 很快 | KV 没灌进 block，Decode 在空 cache 上生成。回头看 load 失败日志 |
| 只有短 prompt 能过，长 prompt 在 Prefill 超时 | `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` |
| GQA 形状或 layout 报错 | 两边 `VLLM_KV_CACHE_LAYOUT` 都是 `LBHNC` 或都是 `HND`。V2 的 GQA 验证范围不包括 PCP/DCP 和 INT8 KV |

V2 起不来、需要先证明「Mooncake 库本身能在 0.30 上搬 KV」时，改回官方单机教程里的 `MooncakeConnectorV1`，并补上 `prefill` / `decode` 的 `dp_size` 与 `tp_size`（本例都是 1）。V1 和 V2 不要在同一次请求里混用。

---

## 5. 第二步才测 KV 池

直传通过之后，如果要测自己的 Mooncake **Store**（Master + Put/Get），换一条配置，不要和上面的 1P1D 叠在同一次冒烟里。

要点（细节以标签上的 KV 池文档为准）：

1. 所有进程 `export PYTHONHASHSEED=0`。
2. `mooncake.json` 里 `protocol` 为 `ascend`，`metadata_server` 为 `P2PHANDSHAKE`，`master_server_address` 指向 master。
3. 只在一台机器起 `mooncake_master --port 50088 ...`。
4. vLLM 侧用 `AscendStoreConnector`，`kv_role` 按该文档的 PD 或单实例示例。`MooncakeConnectorStoreV1` 是同一实现的别名。
5. 再发两条相同前缀的请求，第二条应少算一段 prefill。这是池命中，不是 P2P 直传。

`global_segment_size` 按 1GB 对齐。A3 上 Store 和 PD 要拆链路时用文档里的 `ASCEND_GLOBAL_RESOURCE_CONFIG`，那是池和直传同时开的配置，不属于第 2 节。

---

## 和 Connector 接口的对应

V2 的类继承 `KVConnectorBase_V1`。冒烟时引擎实际走到的是：

- Scheduler：`get_num_new_matched_tokens` → `update_state_after_alloc` → `build_connector_meta`；Prefill 结束时 `request_finished` 暂扣 block。
- Worker：`register_kv_caches` 里 `register_memory`；`start_load_kv` 发起拉取；`get_transfer_results` 把完成和失败交回 Scheduler。
- `wait_for_layer_load` / `save_kv_layer` 在这条直传上是空的。要看逐层搬运，换 `MooncakeLayerwiseConnector`，那是另一次测试。

所以「自己的 Mooncake」若只改 Transfer Engine，第 3 节的替换就够。若改的是 vLLM-Ascend 里的 connector 类，保持 `register_connector()` 里的模块路径不变，或用 `kv_connector_module_path` 指向自己的类，类构造函数仍然是 `(vllm_config, role, kv_cache_config)`。
