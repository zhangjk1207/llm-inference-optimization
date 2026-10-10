# 实验结果

## GLM：2026-10-10 长输入与 Agent 调度

[glm-20261010/README.md](glm-20261010/README.md) 记录已部署的 1024 prefill 上限、6～10 Agent 实测、长输入交错准入、缓存回读、边界、取消和 256K 检查。目录包含摘要及合成请求时序导出，保留诊断阶段记录。收益与冷长输入变慢的代价分别列出。

## GLM：2026-10-06 历史短输入基准

`glm-20261006/` 包含基线、CustomAllreduce、稳定配置确认、异步调度、32MB buffer、精确图捕获尺寸和 8 路复测，共七组 JSON 摘要。

README 的历史并发表对应 `optv10-exactsize-c1-2-4-8-16-o512.json`。其他配置结果可用于比较通信和图捕获策略。

## CUDA graph 覆盖

`infra-20261001/` 包含 GLM 旧捕获配置的 c3/c4 测试、扩展捕获后的并发测试及多轮复测，保留逐轮吞吐和 TTFT。

## 测量口径

GLM JSON 为多轮摘要，infra 文件为逐轮 JSON。DeepSeek V4/V4.1 的 README 表格来自实验报告汇总。token 计数、缓存状态与计时口径见 [基准说明](../docs/benchmarking.md)。

[source-manifest.json](source-manifest.json) 保存源文件校验值和脚本改编记录。
