# 历史结果及来源

## GLM：2026-10-06

`glm-20261006/` 保留本地已有的全部七份 JSON 摘要：baseline、customar、final-confirm、async-check、32MB buffer、exactsize 与 c8clean。没有改写数值或删掉较差的配置。目录中文件名继续使用历史命名，避免破坏来源对应。

README 的最终配置表使用 baseline 对 exactsize 的同一组并发，与最终脚本含 5/10 行捕获尺寸一致；另单列前一稳定配置的 final-confirm 完整表，不跨实验拼接最佳值。

## 基础设施：2026-10-01

`infra-20261001/` 保留 GLM c3/c4 旧图覆盖测试、修复后逐轮测试、后续三次生产复测和 canary 多并发记录。文件包含原始低速轮次，某些样本具有较大波动，不能统一称为低噪声严格热态结果。

## 报告级证据

DSv4/DSv4.1 及 9 月缓存场景的数字整理自本地 Markdown 报告。当前草稿没有复制报告里的内网访问方式、SSH 登录配置、生产回滚命令和远端路径，也没有将缺失的远端原始文件伪装为已经收录的数据。对应结果只能解释为历史报告级证据。

## 来源核对

`source-manifest.json` 对直接复制的文件保存原相对路径和 SHA-256。`adapted=true` 表示公开草稿有参数化或统计修正，记录的 hash 是整理前本地源文件的 hash。`bench_decode.py` 与 CustomAllreduce 包装器也由本地原脚本改编，来源记录在同一清单。

最终部署脚本也记录来源。A40 节点 A 的 Docker 脚本从本地已验证部署文档提取并更新至最终 8 路；节点 B 的 Docker 入口共享该配置，P2P 差异由部署作者确认；另保留节点 B 实际 runtime 脚本的参数化副本。独立 LMCache 启动器是按历史配置补充的模板。镜像核对证据在 [image-evidence.json](../deployment/image-evidence.json)。
