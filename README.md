# LLM Inference Optimization

GLM-5.3-Flash、DeepSeek-V4 和 DeepSeek-V4.1 的 PCIe 多卡推理优化，包含部署脚本、并发实测与性能诊断工具。

主要优化：CUDA graph 覆盖、投机解码、多卡通信和 KV 缓存。项目采用 [MIT](LICENSE) 许可证。

## 模型与部署环境

| 模型 | GPU | 镜像 | 互联与通信 | 部署脚本 |
|---|---|---|---|---|
| [GLM-5.3-Flash-W4A16](https://modelscope.cn/models/JANGQ-AI/GLM-5.3-Flash-W4A16) | 4×A800 80GB，320GB | [lazymio/vllm-backport:v0.13.1-sm80](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm80)，Apptainer | PCIe，无 NVLink；组内 PIX、跨 NUMA SYS；P2P 可用，启用 CustomAllreduce | [GLM](deployment/start-glm53-final.sh) |
| [DeepSeek-V4-Flash-0731](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731) | 4×A800 80GB，320GB | [lazymio/vllm-backport:v0.13.1-sm80](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm80)，Docker | PCIe，PIX/PXB/SYS；禁用 NCCL P2P | [DeepSeek V4](deployment/start-dsv4-final.sh) |
| [DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)，节点 A | 8×A40 48GB，384GB | [lazymio/vllm-backport:v0.13.1-sm86](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm86)，Docker | PCIe，无 NVLink；PIX/PXB/SYS；禁用 NCCL P2P | [A40 节点 A](deployment/start-dsv41-a40-final.sh) |
| [DeepSeek-V4.1-Flash](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash)，节点 B | 8×A40 48GB，384GB | [lazymio/vllm-backport:v0.13.1-sm86](https://hub.docker.com/r/lazymio/vllm-backport/tags?name=v0.13.1-sm86) | PCIe，无 NVLink；P2P 正常启用 | [A40 节点 B](deployment/start-dsv41-a40-p2p-final.sh) |

两个 A40 节点使用相同推理配置，区别是 P2P 状态。A800 为 SM80，A40 为 SM86。禁用 P2P 的节点在真实引擎初始化中出现通信阻塞，因此保留 `NCCL_P2P_DISABLE=1`。

## GLM-5.3-Flash W4A16

### 配置

| 参数 | 值 |
|---|---|
| 并行与算子 | TP4，MARLIN WNA16 MoE，CustomAllreduce |
| 最大上下文 / 最大序列数 | 262144 / 16 |
| 投机解码 / batched tokens | MTP k=4 / 1152 |
| GPU 显存预算 | 0.92 |
| KV 缓存 | Prefix cache + LMCache，CPU L1 512GiB，chunk 1152 |
| CUDA graph | FULL_AND_PIECEWISE；`[1,2,4,5,8,10,16,20,25,32,40,48,56,64,72,80]` |

### 并发测试

同并发预热，512 token 输出，5 轮取 p50。吞吐单位为 token/s，TTFT 单位为秒。

| 请求并发 | 基线聚合 tok/s | 最终聚合 tok/s | 变化 | 最终 TTFT，s | 各轮吞吐范围 |
|---:|---:|---:|---:|---:|---|
| 1 | 201.9 | 225.7 | +11.8% | 0.487 | 223.5–226.6 |
| 2 | 338.4 | 372.8 | +10.2% | 0.590 | 353.9–375.0 |
| 4 | 502.8 | 531.3 | +5.7% | 0.792 | 525.3–540.6 |
| 8 | 719.8 | 703.8 | -2.2% | 1.064 | 648.8–729.6 |
| 16 | 977.7 | 1032.8 | +5.6% | 1.738 | 1024.5–1038.1 |

8 路另一次复测为 715.6 token/s、TTFT 1.128s。单路、2 路和 16 路收益较明确，8 路基本持平。

原始结果：[基线](results/glm-20261006/baseline-v2-prod-c1-2-4-8-16-o512.json)、[最终配置](results/glm-20261006/optv10-exactsize-c1-2-4-8-16-o512.json)、[8 路复测](results/glm-20261006/optv10-c8clean-c8-o512.json)。

## DeepSeek-V4-Flash

### 配置

| 参数 | 值 |
|---|---|
| 并行 / 投机解码 | TP4 + EP，DSpark k=5 |
| 最大上下文 / 最大序列数 | 393216 / 32 |
| batched tokens / GPU 显存预算 | 2176 / 0.92 |
| KV 缓存 | fp8_ds_mla；GPU KV 3,745,707 tokens；CPU L1 256GiB，chunk 256 |
| 通信 | 禁用 NCCL P2P、InfiniBand 和 CustomAllreduce |
| CUDA graph | `[1,2,4,8,16,24,32,40,48,56,64,80,96,120,144,168,192]` |
| 图池 | 约 1.23–1.25GiB/卡 |

### 并发测试

热态测试，512 token 输出，聚合吞吐取 p50。

| 请求并发 | 基线聚合 tok/s | 最终聚合 tok/s | 最终单路 tok/s，p50 | 最终 TTFT，s |
|---:|---:|---:|---:|---:|
| 1 | 224.1 | 220.5 | 220 | 0.35 |
| 4 | 535.8 | 537.5 | 137 | 0.57 |
| 6 | 194.2 | 652.6 | 110 | 0.41 |
| 8 | 255.7 | 795.0 | 103 | 0.46 |
| 16 | 481.0 | 1014.6 | 64.8 | 0.52 |
| 24 | — | 1126.3 | 47.3 | — |
| 32 | — | 1187.6 | 32.7 | — |

扩展图捕获范围后，6 路及以上的吞吐断崖消失，6 路吞吐提升约 236%，8 路约 211%，16 路约 111%。108K token 冷长上下文 TTFT 为 36.8s，chat、reasoning 和工具调用回归通过。

[部署脚本](deployment/start-dsv4-final.sh) · [CUDA graph 优化说明](docs/cuda-graphs.md)

## DeepSeek-V4.1-Flash：8×A40

### 配置

| 参数 | 值 |
|---|---|
| 并行 / 投机解码 | TP8 + EP，DSpark k=5，local argmax reduction |
| 最大上下文 / 最大序列数 | 262144 / 8 |
| batched tokens / GPU 显存预算 | 2048 / 0.95 |
| KV / Engram | fp8_ds_mla；CPU L1 128GiB，chunk 1024；Engram CPU offload |
| CUDA graph | FULL_AND_PIECEWISE；`[6,12,18,24,30,36,42,48]` |
| 通信 | 节点 A 禁用 P2P，节点 B 启用 P2P；均禁用 CustomAllreduce |
| GPU KV 容量 | 节点 A 936,141 tokens；节点 B 937,055 tokens |
| 图池 | 约 0.50GiB/卡 |

### 并发测试

512 token 输出，按 `usage.completion_tokens` 统计真实 token 数。下表展示两台机器的实测结果。

| 场景 | 节点 A：P2P 禁用 | 节点 B：P2P 启用 |
|---|---:|---:|
| c1，短 prompt，token/s | 175 | 212.7 |
| c4，短 prompt，token/s | 268 | 413.9 |
| c8，短 prompt，token/s | 385 | 583.3 |
| c8，6K 共享前缀，token/s | 436 | 485.8 |
| c8，30K 热文档，token/s | 248 | 367.9 |
| c8，短 prompt TTFT，s | 0.5–0.6 | 0.57–0.73 |
| 30K 二次访问 TTFT，s | 0.79 | 0.63–0.73 |
| c8，30K 热文档 TTFT，s | 2.1 | 1.66 |

节点 A 的 c8 三轮吞吐为 329/385/507 token/s；节点 B 复测范围约 487.7–604.7 token/s。测试期间存在其他请求负载。

节点 B 将最大序列数从 4 扩至 8 后，同样 8 路请求的聚合吞吐从 **340.7 提升至 583.3 token/s（+71.2%）**，TTFT 从 **2.3–2.5s 降至 0.57–0.73s**。

[节点 A 部署脚本](deployment/start-dsv41-a40-final.sh) · [节点 B 部署脚本](deployment/start-dsv41-a40-p2p-final.sh) · [runtime 启动方式](deployment/start-dsv41-runtime-final.sh)

## 部署

从上方 ModelScope 页面下载模型，将 `MODEL` 设置为下载位置。GLM 使用对应 SM80 镜像的 SIF 文件；DeepSeek 使用 Docker 镜像。

```bash
# GLM：同时启动 LMCache 和推理服务
MODEL=/models/GLM-5.3-Flash-W4A16 \
IMAGE=/images/vllm-backport-v0.13.1-sm80.sif \
bash deployment/start-glm53-final.sh

# DeepSeek V4
MODEL=/models/DeepSeek-V4-Flash-0731 bash deployment/start-dsv4-final.sh

# DeepSeek V4.1，P2P 禁用节点
MODEL=/models/DeepSeek-V4.1-Flash bash deployment/start-dsv41-a40-final.sh

# DeepSeek V4.1，P2P 正常节点
MODEL=/models/DeepSeek-V4.1-Flash bash deployment/start-dsv41-a40-p2p-final.sh
```

DeepSeek 部署先启动 [LMCache](deployment/start-lmcache.sh)：

```bash
# V4.1：128GiB / chunk 1024 / port 5556
IMAGE=lazymio/vllm-backport:v0.13.1-sm86 bash deployment/start-lmcache.sh

# V4：256GiB / chunk 256 / port 5555
IMAGE=lazymio/vllm-backport:v0.13.1-sm80 CACHE_PORT=5555 \
CACHE_CHUNK_SIZE=256 CACHE_SIZE_GB=256 CACHE_GPU_WORKERS=4 \
bash deployment/start-lmcache.sh
```

默认 API 监听 `127.0.0.1`；对外提供服务时设置 `API_HOST=0.0.0.0`。A40 重启时先等待旧 worker 退出、LMCache GPU IPC 映射释放，再启动新引擎。

## 基准与诊断

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 benchmarks/bench_decode.py --api http://127.0.0.1:8000 \
  --model YOUR_MODEL --cs 1,2,4,8,16 --out-tokens 512 \
  --warmup-rounds 2 --rounds 5 --cache-mode cold \
  --output local-results/cold.json
```

`--cache-mode shared-prefix` 测共享前缀；`BENCH_API_KEY` 提供鉴权。吞吐为输出 token 数除以请求总耗时，TTFT 从首个非空内容计时。更多测量细节见 [基准说明](docs/benchmarking.md)。

| 内容 | 入口 |
|---|---|
| CUDA graph 与投机解码 | [优化说明](docs/cuda-graphs.md) |
| PCIe、NCCL 与 CustomAllreduce | [通信说明](docs/communication.md)、[诊断脚本](diagnostics/) |
| KV 缓存与并发 | [缓存说明](docs/cache-and-concurrency.md) |
| 完整实验结果 | [results](results/README.md) |

```bash
python3 -m unittest discover -s tests -v
```
