# LLM Inference Optimization

面向 PCIe 多 GPU 推理的实测记录、诊断脚本和部署经验，整理自 2026 年 9–10 月的 GLM、DSv4 和 DSv4.1 优化工作。

当前为发布前草稿。代码覆盖热态吞吐基准、共享前缀压测和通信诊断；模型相关配置来自特定 vLLM backport，不能直接视作上游版本支持承诺。

## 内容导航

| 主题 | 内容 | 入口 |
|---|---|---|
| 投机解码与 CUDA graph | 并发增加后吞吐断崖、verify 形状与捕获范围 | [案例说明](docs/cuda-graphs.md) |
| 多卡通信 | NCCL 微基准与真实服务的差异、CustomAllreduce 实验 | [通信说明](docs/communication.md)、[诊断脚本](diagnostics/) |
| 缓存与并发 | 前缀命中率、新 prefill 工作量、LMCache 启动次序 | [缓存说明](docs/cache-and-concurrency.md) |
| 基准方法 | TTFT、真实 token 计数、预热、多轮 p50 | [测量口径](docs/benchmarking.md)、[基准脚本](benchmarks/) |
| 历史结果 | 全部保留的 GLM 配置实验、DSv4/DSv4.1 汇总 | [结果说明](results/README.md) |

## 主要观测

| 实验 | 场景 | 变化 | 证据与限制 |
|---|---|---|---|
| GLM graph 覆盖修复 | 4 并发、512 token 输出 | 130.3 → 434.2 tok/s，约 3.33 倍 | 保留前后逐轮 JSON；修复后的一个样本仍有首轮尖刺 |
| DSv4 graph 覆盖修复 | 6 并发、512 token 输出 | 194.2 → 652.6 tok/s，约 3.36 倍 | 原始报告汇总，当前草稿未附远端逐轮原始数据 |
| GLM CustomAllreduce | 1 并发、512 token 输出 | 201.9 → 225.5 tok/s，约 +11.7% | 本地 JSON 摘要；统一使用 final-confirm 实验，不拼接最佳值 |
| DSv4.1 并发扩展 | 第二节点、8 并发 | 340.7 → 583.3 tok/s，约 +71.2% | 原报告真 token 口径，存在生产流量干扰 |

以上是特定硬件、模型、引擎与缓存状态下的历史观测。前两项主要修复异常退化，第三项是在较健康基线上的增量优化；不同实验不能合并成一个统一提升幅度。

## 快速使用

HTTP 基准需要 Python 3.10+ 和一个支持流式 `/v1/completions`、`stream_options.include_usage` 的兼容服务。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 benchmarks/bench_decode.py --api http://127.0.0.1:8000 \
  --model YOUR_MODEL --cs 1,2,4,8,16 --out-tokens 512 \
  --warmup-rounds 2 --rounds 5 --cache-mode cold \
  --output local-results/cold.json
```

运行同一共享前缀场景时，将 `--cache-mode` 改为 `shared-prefix`。需要鉴权的服务用 `BENCH_API_KEY` 环境变量。脚本使用 vLLM 扩展 `ignore_eos`，固定输出长度；不支持该参数的服务需要适配。

共享前缀的 chat 网关基准：

```bash
export BENCH_MODEL=YOUR_MODEL
# NEWAPI_KEY 从本地环境提供
python3 benchmarks/bench_shared_prefix.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --common-repeats 2000 --unique-lines 100 --levels 1,2,4,8
```

GPU 诊断需要在目标引擎环境内安装 PyTorch、CUDA/NCCL 和相应 vLLM backport，详见[通信说明](docs/communication.md)。HTTP 依赖文件不安装这些 GPU 包。

## 复现边界

- 历史环境包括 4×A800 80GB PCIe、8×A40 48GB PCIe；各节点 P2P 状况不同。
- GLM 历史报告标记引擎为 vLLM `v0.13.1 backport`，未保存足够信息将所有结果绑定到唯一源码 commit 或镜像 digest。该版本文字不能作为完整复现标识。
- 当前包含原始摘要和部分逐轮数据；trace、模型权重、镜像、生产日志和远端主机配置未收录。
- 草稿新增的统计修正尚未在目标 GPU 服务重跑。历史数据保留原测量口径，不会被改写成新版脚本测量结果。
- 依赖软件、模型和 backport 分别遵循各自许可证。当前草稿尚未授予本项目开源许可，正式发布时需补充项目许可证和权属确认。

## 本地验证

```bash
python3 -m unittest discover -s tests -v
```

测试通过本地模拟 SSE 服务检查统计逻辑，不启动模型服务。
