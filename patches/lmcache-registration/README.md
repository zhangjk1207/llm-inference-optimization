# LMCache 注册存活修复（2026-10-09，10-10 继续验证）

这是当前 vLLM backport v0.13.1 SM80 镜像内 LMCache 的 ZMQ/IPC 定向修复，不是任意 LMCache 版本通用补丁。相同镜像标签不保证包内容相同；修改文件的原始 SHA-256 见 `original-source-sha256.json`。

> **上游追踪（2026-10-13）**：本修复的现场报告已发至 [LMCache#4715](https://github.com/LMCache/LMCache/issues/4715#issuecomment-6136695036)（同域根因另见 #4795），并与在途 PR #4593 / #4687 / #4732 协调向 `dev` 分支移植；本目录修复的探测与重注册部分（第 2–4 条）对应上游明确推迟的 follow-up。
> Upstream: field report posted at [LMCache#4715](https://github.com/LMCache/LMCache/issues/4715#issuecomment-6136695036) (2026-10-13); the detection / re-registration part is being coordinated toward `dev` with in-flight PRs #4593 / #4687 / #4732.

故障表现为 worker 注册在空闲期间过期，旧 PING 只能确认服务器活着，不能发现注册已经丢失。修复在注册 ACK 后立即启动心跳，通过 PING_REGISTERED 检测注册，失败时重新注册并重试，恢复期间复用 IPC 上下文，并串行化关闭与重新注册。

`registration-liveness.patch` 是相对实测镜像原包的完整源码差异；`verify_registration.py` 是当时通过的 13 项针对性测试，需要镜像内完整依赖。gRPC 和其他传输模式未验证。原包及其原生动态库不随本仓库分发；适用的第三方许可继续有效。

## 准备和部署

1. 从待使用的镜像导出完整 `lmcache` 包到独立的新目录。核对本目录列出的原始源码校验值；不匹配时先审查版本差异，不要强制打补丁。
2. 对新包目录先 dry-run，再应用补丁。下例路径替换为本机实际绝对路径：

   ```bash
   patch --dry-run --batch --fuzz=0 -p2 -d /staging/lmcache -i /repo/patches/lmcache-registration/registration-liveness.patch
   patch --batch --fuzz=0 -p2 -d /staging/lmcache -i /repo/patches/lmcache-registration/registration-liveness.patch
   ```

3. 用同一镜像绑定新包，运行 `verify_registration.py`。测试用实际类和模拟传输资源，不替代真实 GPU 回归。
4. 停止原服务，设置 `LMCACHE_PACKAGE=/staging/lmcache`，启动本仓库 GLM 脚本。脚本将同一包只读绑定到缓存服务和推理进程的 `/usr/local/lib/python3.12/dist-packages/lmcache`；其他 Python 路径需适配。
5. 核对四个 worker 注册、缓存写入、真实 CPU 回读、空闲存活、取消和重启。部署包运行期间不得覆盖或重建，包括其中的 `.so`。

如果镜像已经包含等价且验证过的修复，可以不设置 `LMCACHE_PACKAGE`；未设置时脚本使用镜像内原包，**不会自动修复注册问题**。10-10 的实测配置使用了上述修复，不能将未修复的镜像视为等价环境。

GLM 启动脚本会监控两个子进程，任一退出就终止服务并返回非零；自动重启需要外部进程管理器配置。脚本本身不会安装 systemd 单元，也不提供开机启动。

## 验证范围

10-09 的 13 项针对性测试通过；10-10 的实机验证确认四个 worker 注册正常，CPU 回读分别传输了 4608、10368 和 16128 tokens，并通过对应位置的标记检索。CPU 强制回读使用诊断干预，不作为普通请求时延 A/B。详见 [Agent 场景报告](../../results/glm-20261010/README.md)。
