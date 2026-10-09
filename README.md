# LLM Inference Optimization

记录 GLM-5.3-Flash、DeepSeek-V4-Flash 和 DeepSeek-V4.1-Flash 在 PCIe 多卡上的最终部署配置、优化过程与并发实测。项目脚本和文档采用 [MIT](LICENSE) 许可证；模型、引擎及镜像分别遵循其自身许可证。

## 部署环境总览

下表对应 2026-10-01～10-06 的最终配置。A40 两节点硬件、镜像与推理参数一致，区别是 NCCL P2P 状态；节点 B 的一致性由部署作者于 2026-10-09 确认。

| 部署 | GPU 与总标称显存 | 镜像 / 运行方式 | 卡间互联及 P2P 情况 | 最终部署脚本 |
|---|---|---|---|---|
| GLM-5.3-Flash W4A16 | 4×A800 80GB，320GB；SM80 | Apptainer：`vllm-backport-v0.13.1-sm80-local.sif` | PCIe，无 NVLink；0–1 / 2–3 为 PIX，跨 NUMA 为 SYS；P2P 可用，最终启用实验性 CustomAllreduce | [GLM 最终脚本](deployment/start-glm53-final.sh) |
| DeepSeek-V4-Flash | 4×A800 80GB，320GB；SM80 | Docker：`lazymio/vllm-backport:v0.13.1-sm80` | PCIe；0–1 为 PIX，2–3 为 PXB，跨 NUMA 为 SYS；CUDA peer copy 可用，但真实引擎通信初始化会挂，保留 `NCCL_P2P_DISABLE=1` | [DSv4 最终脚本](deployment/start-dsv4-final.sh) |
| DeepSeek-V4.1-Flash，A40 节点 A | 8×A40 48GB，384GB；SM86 | Docker：`lazymio/vllm-backport:v0.13.1-sm86` | 无 NVLink，PIX/PXB/SYS 混合 PCIe；八卡真实模型 P2P 初始化不稳定，`NCCL_P2P_DISABLE=1` | [A40 节点 A 最终脚本](deployment/start-dsv41-a40-final.sh) |
| DeepSeek-V4.1-Flash，A40 节点 B | 同样 8×A40 48GB，384GB；SM86 | 同一 SM86 镜像；历史运行在 K8s 容器内的 runtime/venv | P2P 健康，启用 P2P；其余推理配置与节点 A 一致 | [节点 B 同配置启动入口](deployment/start-dsv41-a40-p2p-final.sh)、[实际 runtime 脚本的参数化副本](deployment/start-dsv41-runtime-final.sh) |

A40 的 48GB 是产品标称容量，历史 `nvidia-smi` 显示可见约 46068MiB，部分引擎预算日志显示约 44.42GiB。总标称显存不能直接作为引擎可分配显存。这里记录已验证的历史拓扑和运行情况，不代表当前实时利用率。

### 镜像身份

- **GLM 实际 SIF 文件**：`vllm-backport-v0.13.1-sm80-local.sif`，2026-10-09 核对本地文件大小为 7,738,228,736 bytes，SHA-256 为 `5b9e3923f4ed04a9eedb1296062c3953d471a369950f95a58dff5e2e2a30253f`。该 hash 标识当前本地文件，历史测量时没有逐次记录 hash。
- **SM80 本地 OCI 归档**：tag 为 `lazymio/vllm-backport:v0.13.1-sm80`；归档 index digest `sha256:0fc33cd7d2be777bf6faaee9b02adf781dc6d0bc44b9a09ce4865089f67e9cba`，linux/amd64 manifest digest `sha256:8094fcbab905a04a480b327f2761255e3d17cd8d39470cac9d76450bbb567f7e`。归档可核验，但没有证据将每台机器、每轮结果全部绑定到此归档。
- **SM86 A40 镜像**：实际使用 tag 为 `lazymio/vllm-backport:v0.13.1-sm86`。旧文档给出的 `sha256:671aa3aadb9e536a89ecda894d93eb15870a4bd5c1d24b588ab5a06e2ca8ea8f` 同时是 SM80 归档的 Docker config ID，不能当作已核验的 SM86 registry manifest digest。当前不使用它拼接 `image@sha256:...`。
- SM80 归档的构建标签将源码 revision 标为 `unknown`；`v0.13.1` 是该 backport 的版本标记，不能替代源码 commit。未使用 `latest-sm80` 评测镜像作为最终部署镜像。

详细证据见 [镜像核对记录](deployment/image-evidence.json)。

## GLM：最终配置与并发测试

最终脚本日期为 2026-10-06，实际模型目录标识为 `GLM-5.3-Flash-W4A16-vllm`，served name 为 `glm-5.3-flash`。

| 配置 / 卡的运行情况 | 最终值 |
|---|---|
| 并行与算子 | TP4，MARLIN WNA16 MoE，实验性 CustomAllreduce |
| 最大上下文 / 最大序列数 | 262144 / 16 |
| MTP / batched tokens | k=4 / 1152；k=8 实验已回退 |
| GPU 显存预算 | 每卡 0.92 |
| KV 缓存 | prefix cache + LMCache，host L1 上限 512GiB，chunk 1152 |
| CUDA graph | FULL_AND_PIECEWISE；`[1,2,4,5,8,10,16,20,25,32,40,48,56,64,72,80]`，上限 80 |
| 早期覆盖修复的代价 | KV token 容量 1,874,276→1,864,724，约 -0.5%；不作为 10 月 6 日重新测得的容量 |

### 最终脚本对应的 exactsize 实验

同并发预热，512 token 输出，各档 5 轮取 p50。**这里统一使用最终脚本含 5/10 行 capture sizes 的 exactsize 实验，整张表来自同一份 JSON。** 吞吐为客户端聚合 token/s，TTFT 为各轮请求中位数的中位数。

| 请求并发 | 基线聚合 tok/s | 最终聚合 tok/s | 变化 | 最终 TTFT，s | 最终各轮吞吐范围 |
|---:|---:|---:|---:|---:|---|
| 1 | 201.9 | 225.7 | +11.8% | 0.487 | 223.5–226.6 |
| 2 | 338.4 | 372.8 | +10.2% | 0.590 | 353.9–375.0 |
| 4 | 502.8 | 531.3 | +5.7% | 0.792 | 525.3–540.6 |
| 8 | 719.8 | 703.8 | -2.2% | 1.064 | 648.8–729.6 |
| 16 | 977.7 | 1032.8 | +5.6% | 1.738 | 1024.5–1038.1 |

证据：[基线](results/glm-20261006/baseline-v2-prod-c1-2-4-8-16-o512.json)、[最终 exactsize](results/glm-20261006/optv10-exactsize-c1-2-4-8-16-o512.json)。c8 另一次干净复测为 **715.6 tok/s、TTFT 1.128s**，见 [c8clean](results/glm-20261006/optv10-c8clean-c8-o512.json)；该档没有稳定可测收益。

### 前一轮稳定配置的 final-confirm 结果

该轮同样是 CustomAllreduce + MTP k=4，但尚未添加最终脚本中的精确 5/10 行 capture sizes。它是原优化总结中 225.5/374.6/536.3/724.6 一组数字的来源，供比较，不与上表拼接。

| 请求并发 | 聚合 tok/s，p50 | TTFT，s |
|---:|---:|---:|
| 1 | 225.5 | 0.486 |
| 2 | 374.6 | 0.588 |
| 4 | 536.3 | 0.779 |
| 8 | 724.6 | 1.038 |
| 16 | 1029.7 | 1.760 |

[完整 final-confirm JSON](results/glm-20261006/optv5-final-confirm-c1-2-4-8-16-o512.json)。修复图覆盖断崖与此次通信增量优化属于不同阶段，不能合并成同一提升幅度。

## DSv4：最终配置与并发测试

模型目录标识 `DeepSeek-V4-Flash-0731`，served name 为 `dataspace-295b`，最终脚本日期 2026-10-01。

| 配置 / 卡的运行情况 | 最终值 |
|---|---|
| 并行 / 投机解码 | TP4 + EP；DSpark k=5；禁用 CustomAllreduce |
| 最大上下文 / 最大序列数 | 393216 / 32 |
| batched tokens / GPU 显存预算 | 2176 / 0.92 |
| KV 缓存 | fp8_ds_mla；GPU KV 3,745,707 tokens；LMCache host L1 256GiB，chunk 256 |
| 通信 | `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`、socket lo |
| CUDA graph | `[1,2,4,8,16,24,32,40,48,56,64,80,96,120,144,168,192]`，上限 192 |
| 图捕获代价 / 稳定性 | 图池约 1.23–1.25GiB/卡，捕获约 31s；报告记录 preemption=0 |

实测为热态、512 token 输出、聚合吞吐 p50；来自历史报告，本仓库没有远端逐轮原始文件。

| 请求并发 | 基线聚合 tok/s | 最终聚合 tok/s | 最终单路 tok/s，p50 | 最终 TTFT，s |
|---:|---:|---:|---:|---:|
| 1 | 224.1 | 220.5 | 220 | 0.35 |
| 4 | 535.8 | 537.5 | 137 | 0.57 |
| 6 | 194.2 | 652.6 | 110 | 0.41 |
| 8 | 255.7 | 795.0 | 103 | 0.46 |
| 16 | 481.0 | 1014.6 | 64.8 | 0.52 |
| 24 | 未测 | 1126.3 | 47.3 | 未测 |
| 32 | 未测 | 1187.6 | 32.7 | 未测 |

c8 首轮复测为 720，后续 5 轮 p50 795、spread 6.4%；取报告的后续复测结果。主要收益来自修复 c≥6 的图覆盖断崖，c1/c4 基本持平。108K token 冷长上下文 TTFT 36.8s，chat/reasoning/tool call 回归通过。

注意原报告的 `1..192` 是覆盖范围简写，实际脚本使用上表的**离散捕获尺寸**。[最终部署脚本](deployment/start-dsv4-final.sh)保留真实参数。

## DSv4.1：两台 8×A40 的最终配置与并发测试

两个节点使用相同模型、SM86 镜像及 8 路推理配置。**节点 A 禁用 P2P，节点 B P2P 正常启用**。跨节点速度差异包含互联、缓存状态和生产竞争因素，不是只改变单个 P2P 开关的隔离实验。

| 配置 / 卡的运行情况 | 两节点最终值或差异 |
|---|---|
| 并行 / 投机解码 | TP8 + EP；DSpark k=5，use_local_argmax_reduction=true；禁用 CustomAllreduce |
| 最大上下文 / 最大序列数 | 262144 / 8（原来是 4） |
| batched tokens / GPU 显存预算 | 2048 / 0.95 |
| KV / Engram | fp8_ds_mla；LMCache host L1 128GiB，chunk 1024，MP port 5556；Engram CPU offload |
| CUDA graph | FULL_AND_PIECEWISE；`[6,12,18,24,30,36,42,48]`，上限 48 |
| P2P | 节点 A：禁用；节点 B：启用 |
| GPU KV token 容量 | 节点 A：936,141；节点 B：937,055 |
| 图池 / 容量限制 | 约 0.50GiB/卡；约 936K KV tokens 不能容纳 8 条满 262K 上下文 |

节点 A 的 [Docker 启动脚本](deployment/start-dsv41-a40-final.sh)从已验证部署文档提取，并按 10 月 1 日最终报告更新为 8 路；节点 B 的 [P2P 启动入口](deployment/start-dsv41-a40-p2p-final.sh)共享该配置并开启 P2P。另附 [节点 B 历史实际 runtime 脚本](deployment/start-dsv41-runtime-final.sh)的参数化副本，它保留原 venv/runtime 启动方式。

### 最终 8 路配置：真实 token 口径

512 token 输出；聚合吞吐单位为 token/s。节点 A 测于 2026-10-01，节点 B 测于 2026-10-02。两节点都有生产流量混入，不能把下面每个数字视为隔离低噪声基准。

| 场景 | 节点 A：P2P 禁用 | 节点 B：P2P 启用 |
|---|---:|---:|
| c1，短 prompt | 175 | 212.7 |
| c2，短 prompt | 真 token 口径未单独记录 | 真 token 口径未单独记录 |
| c4，短 prompt | 268 | 413.9 |
| c8，短 prompt | 385，三轮 329/385/507 | 583.3，复测范围约 487.7–604.7 |
| c8，约 6K 共享前缀 | 436 | 485.8 |
| c8，30K 热文档 | 248 | 367.9 |
| c1 短 prompt TTFT | 0.36–0.46s | 原报告未单列最终该档 TTFT |
| c8 短 prompt TTFT | 后续约 0.5–0.6s；首轮约 2.0s | 0.57–0.73s |
| 30K 冷文档 TTFT | 17.4s | 1.05s，但缓存命中 28672 tokens，**不是真正冷算** |
| 30K 二次访问 TTFT | 0.79s | 0.63–0.73s |
| c8，30K 热文档 TTFT，p50 | 2.1s | 1.66s |

节点 A 早期测试将 SSE chunk 当成 token，31.8/71.9/103.9 等旧值**不作为真实 token/s 发布**。本表使用修正后的 usage.completion_tokens 结果。节点 A c8 短 prompt TTFT 来自早期首内容计时记录，token 计数修正不改变该时间定义。

节点 B 相同 c8 请求并发的 4 路→8 路对比：**340.7→583.3 token/s，约 +71.2%**；TTFT 从排队时 2.3–2.5s 降到 0.57–0.73s。节点 A 的旧 4 路与新 8 路没有完整真 token 口径的隔离 A/B，避免用旧 chunk 数构造提升幅度。

两节点最终回归包含 chat、reasoning、工具调用和 prefix cache。0.95 显存预算依赖启动次序：引擎退出后先确认旧 GPU IPC 映射释放，再启动新引擎；历史节点 A 的 LMCache 映射未释放时，vLLM 冷启动曾失败。

## 部署脚本的使用方式

脚本保留最终推理参数，模型路径、镜像路径、日志/编译缓存目录参数化；默认 API 监听 `127.0.0.1`，需要原部署的对外监听时设置 `API_HOST=0.0.0.0`。公开适配版本经过语法和命令参数检查，尚未重新启动 GPU 服务验证。

```bash
# GLM：实际 SIF + 四卡模型；脚本同时启动 LMCache
MODEL=/path/to/GLM-5.3-Flash-W4A16-vllm \
IMAGE=/path/to/vllm-backport-v0.13.1-sm80-local.sif \
bash deployment/start-glm53-final.sh

# DSv4：先确保本机 LMCache 5555 可用，chunk=256，host L1=256GiB
MODEL=/path/to/DeepSeek-V4-Flash-0731 bash deployment/start-dsv4-final.sh

# DSv4.1：先确保本机 LMCache 5556 可用，chunk=1024，host L1=128GiB
MODEL=/path/to/DeepSeek-V4.1-Flash bash deployment/start-dsv41-a40-final.sh

# 同样的八卡 A40，P2P 健康节点
MODEL=/path/to/DeepSeek-V4.1-Flash bash deployment/start-dsv41-a40-p2p-final.sh
```

DSv4/DSv4.1 原独立 LMCache 容器脚本未保存在本地；提供 [LMCache 配套模板](deployment/start-lmcache.sh)，它不是历史容器的逐字快照：

```bash
# DSv4.1 两节点，128GiB / chunk 1024 / 5556
IMAGE=lazymio/vllm-backport:v0.13.1-sm86 bash deployment/start-lmcache.sh

# DSv4，256GiB / chunk 256 / 5555
IMAGE=lazymio/vllm-backport:v0.13.1-sm80 CACHE_PORT=5555 \
CACHE_CHUNK_SIZE=256 CACHE_SIZE_GB=256 CACHE_GPU_WORKERS=4 \
bash deployment/start-lmcache.sh
```

模板创建新容器，不替换已有服务；已有部署按原运维脚本重启，先释放旧 IPC 映射。GLM 特定部署脚本默认开启最终配置的 CustomAllreduce 实验覆盖；通用 [包装器](patches/serve_custom_ar.py)仅在 `VLLM_FORCE_PCIE_CUSTOM_AR=1` 时覆盖平台判定，其他硬件先验证正确性和 graph replay。

## 基准使用与统计口径

HTTP 基准需要 Python 3.10+，支持 `/v1/completions`、stream usage 和 `ignore_eos` 的服务。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 benchmarks/bench_decode.py --api http://127.0.0.1:8000 \
  --model YOUR_MODEL --cs 1,2,4,8,16 --out-tokens 512 \
  --warmup-rounds 2 --rounds 5 --cache-mode cold \
  --output local-results/cold.json
```

`--cache-mode shared-prefix` 测量共享前缀，鉴权使用 `BENCH_API_KEY`。输出真实 token 数、首非空内容 TTFT、失败率、吞吐 p50/min/max 和缓存命中信息。

历史脚本的计时包含约 0.3 秒门控等待；新版脚本已经去掉。历史吞吐是 token 数除以请求总耗时，包含 prefill/TTFT，不是排除首字等待的纯 decode token/s。历史数据保留原值；新版脚本没有重跑以上硬件实验。DSv4/DSv4.1 表来自报告级记录，逐轮远端原始数据未收录；GLM JSON 和早期 infra 逐轮数据保留于 [results](results/README.md)。

## 方法与证据导航

| 主题 | 入口 |
|---|---|
| 投机解码 / CUDA graph 覆盖 | [方法与案例](docs/cuda-graphs.md) |
| PCIe / NCCL / CustomAllreduce | [通信说明](docs/communication.md)、[诊断脚本](diagnostics/) |
| 前缀缓存 / 并发 / 启动次序 | [缓存说明](docs/cache-and-concurrency.md) |
| TTFT / 真 token / 多轮测量 | [统计口径](docs/benchmarking.md)、[基准脚本](benchmarks/) |
| 数据来源和完整实验 | [结果说明](results/README.md)、[来源 hash 清单](results/source-manifest.json) |

```bash
python3 -m unittest discover -s tests -v
```

本地验证使用模拟 SSE 服务和命令替身，不加载模型。源码 commit、SM86 可拉取 digest、完整远端原始结果尚未齐全，按相同环境复现时需补录。镜像和模型权重不随项目分发。
