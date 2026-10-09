# PCIe 多卡通信实验

NCCL 算法/协议、P2P 和 CustomAllreduce 的效果依赖 GPU 拓扑、库版本以及真实模型执行路径。

## NCCL 探测

`diagnostics/probe_nccl.py` 改编前的本地脚本测试了 8KiB、64KiB、1MiB、64MiB all-reduce，逐轮重新填充张量并检查结果。

在没有生产负载的目标环境中运行：

```bash
timeout 120s torchrun --standalone --nproc_per_node=4 diagnostics/probe_nccl.py
```

该脚本使用当前环境中的 NCCL 配置，比较配置时必须保存各组环境变量。进程组的 60 秒超时未必能终止所有底层 CUDA 阻塞，因此示例另设外部超时。

历史某节点的原始 torch all-reduce 在 P2P 开启时通过，但真实 vLLM PyNcclCommunicator 初始化仍死锁；另一个 A800 节点在自动选择下正常。因此微基准通过只是必要观测之一，继续用真实引擎的初始化、图捕获、解码和重启验证。

## CustomAllreduce

本地 GLM 实验强制覆盖平台 `is_fully_connected` 判定，尝试在已测四卡 PCIe A800 上启用 CustomAllreduce。10 月 6 日的 final-confirm 摘要与基线对比：

| 并发 | 基线 tok/s | final-confirm tok/s | 变化 |
|---:|---:|---:|---:|
| 1 | 201.9 | 225.5 | +11.7% |
| 2 | 338.4 | 374.6 | +10.7% |
| 4 | 502.8 | 536.3 | +6.7% |
| 8 | 719.8 | 724.6 | +0.7% |
| 16 | 977.7 | 1029.7 | +5.3% |

另一轮 c8 干净复测为 715.6，低于 719.8 基线，说明 c8 收益不足以与噪声区分。原报告中的 1032.8 来自 exactsize 实验，当前表格统一选 final-confirm 的 1029.7。

`patches/serve_custom_ar.py` 是适配私有 API 的实验包装器。默认不覆盖判定；显式设置环境变量后才开启：

```bash
VLLM_FORCE_PCIE_CUSTOM_AR=1 python3 patches/serve_custom_ar.py \
  serve /path/to/model --tensor-parallel-size 4 OTHER_VERIFIED_ARGS
```

原始四卡诊断改编为 `diagnostics/probe_custom_ar.py`，包含小批次正确性、延迟和图内 replay 检查。它针对历史 backport 内部配置，移植到其他版本需适配；运行前使用独立环境，外加 `timeout`。草稿整理时未在 GPU 上重新执行这些探测。

## 已尝试但没有有效收益的方向

- 草稿深度 k=8：原报告记录接受率后半段崩塌，单路变慢；最终回到 k=4。
- CustomAllreduce 缓冲区 32MB：对应 JSON 数据显示无收益。
- 仅添加精确 5/10 行 capture sizes：收益落在波动内。
- 固定 NCCL SIMPLE/LL128：prefill 探测较默认配置更慢。
- Triton MoE 替换 MARLIN：该模型/硬件的原报告记录显著变慢。

原报告关于“+50% 物理不可达”的推导依赖带宽假设、测量范围与当前模型。草稿保留尝试结论，但不将有限实验覆盖解释为所有未来优化的上界。
