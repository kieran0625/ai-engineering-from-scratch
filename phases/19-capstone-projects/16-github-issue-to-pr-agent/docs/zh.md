# 综合项目 16 — GitHub Issue 转 PR 自主代理

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 均采用了相同的 2026 年产品形态：标记一个 Issue，生成一个 PR。在云端沙箱中运行代理，验证测试通过，并发布附带执行说明的、可供审查的 PR。难点在于自动复现仓库的构建环境、防止凭证泄露、执行单仓库预算限制，以及确保代理无法强制推送（force-push）。本综合项目将构建自托管版本，并在成本与通过率方面与托管替代方案进行对比。

**类型：** 综合项目
**语言：** Python（代理）、TypeScript（GitHub App）、YAML（Actions）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（代理）、阶段 15（自主）、阶段 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P17
**耗时：** 30 小时

## 问题描述

异步云端编码代理是一个与交互式编码代理（综合项目 01）独立的产品类别。其交互体验基于 GitHub Label。当你为 Issue 添加标签 `@agent fix this` 时，工作节点会在云端沙箱中启动，克隆仓库、运行测试、编辑文件、验证结果，并打开一个 PR，PR 正文中包含代理的执行说明。整个过程无需交互式循环，也无需终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 均收敛于这一模式。

工程挑战具体明确：环境复现（代理必须从零构建仓库，不能使用缓存的开发镜像）、不稳定测试（必须重新运行或隔离）、凭证作用域控制（使用最小化细粒度权限的 GitHub App）、按仓库每日预算执行，以及禁止强制推送策略。本综合项目将衡量其在通过率、成本和安全性方面与托管替代方案的对比表现。

## 概念设计

触发器为 GitHub Webhook（Issue 标签或 PR 评论）。调度器将任务入队至 ECS Fargate 或 Lambda。工作节点将仓库拉取到 Daytona 或 E2B 沙箱中，并根据仓库信息（语言、框架）推断生成通用的 Dockerfile。代理运行 mini-swe-agent 或 SWE-agent v2 循环，调用 Claude Opus 4.7 或 GPT-5.4-Codex。迭代过程包括：读取代码、提出修复方案、应用补丁、运行测试。

验证是关键门禁步骤。PR 打开前，沙箱内的完整 CI 必须全部通过。系统会计算覆盖率变化量；若下降超过阈值，PR 仍会打开，但会被打上标签 `needs-review`。代理会将执行说明作为 PR 描述发布，并创建一个 `@agent` 讨论线程，供审查者 @提及以进行后续跟进。

安全性通过两个不同的 GitHub 层面进行作用域控制：App 提供短期有效的安装令牌，具备 `workflows: read` 及狭窄的仓库内容/PR 作用域；分支保护规则（而非 App 权限）强制执行“禁止直接写入 `main`”和“禁止强制推送”——该 App 永远不会被加入绕过列表。GitHub App 原生并不支持对 `.github/workflows` 的路径级只读访问，因此代理的文件编辑白名单必须在 Worker 端强制执行此限制。按仓库每日预算上限由调度器强制执行（例如：每个仓库每天最多 5 个 PR，每个 PR 最高 $20）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器：带细粒度令牌的 GitHub App；通过 Lambda 或 Fly.io 接收 Webhook
- 工作节点：ECS Fargate 任务（或 GitHub Actions 自托管 Runner）
- 沙箱：每个任务独立的 Daytona devcontainer 或 E2B 沙箱
- 代理循环：mini-swe-agent 基线或基于 Claude Opus 4.7 / GPT-5.4-Codex 的 SWE-agent v2
- 检索：tree-sitter 仓库映射 + ripgrep
- 验证：沙箱内完整 CI + 覆盖率变化门禁
- 可观测性：Langfuse，PR 正文链接每个 PR 的追踪归档
- 预算：按仓库每日美元上限；每个仓库每天最大 PR 数

## 构建指南

1. **GitHub App。** 细粒度安装令牌：issues 读写、pull_requests 写入、contents 读写、workflows 读取。分支保护规则（唯一能实现此功能的层面）强制执行“禁止直接推送到 `main`”和“禁止强制推送”；该 App 不在绕过列表中。由于 GitHub App 权限不支持路径作用域，Worker 会通过检查提议的 Diff 来强制执行“禁止写入 `.github/workflows` 目录下”的白名单校验。

2. **Webhook 接收器。** Lambda 函数接收 Issue 标签/PR 评论 Webhook。按标签 `@agent fix this` 进行过滤。将任务入队至 SQS。

3. **调度器。** 从 SQS 弹出任务。强制执行按仓库每日预算。启动一个 ECS Fargate 任务，传入仓库 URL、Issue 正文和全新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。若不存在 Dockerfile，则动态生成。

5. **代理循环。** 使用 Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter 仓库映射、read_file、edit_file、run_tests、git。硬性限制：$20 成本、30 分钟实际耗时（wall-clock）、30 次代理交互轮次。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率变化量。若 CI 失败：停止运行，不打开 PR。若覆盖率下降超过 2%：打开 PR 并打上 `needs-review` 标签。

7. **发布 PR。** 推送代理分支。通过 GitHub API 打开 PR，包含：标题、执行说明、Diff 摘要、追踪 URL、成本、交互轮次。

8. **凭证清理。** Worker 使用短期有效的 GitHub App 安装令牌运行。日志在归档前会清除敏感信息。

9. **评估。** 使用 30 个不同难度的内部种子 Issue。衡量通过率、PR 质量（Diff 大小、风格规范、覆盖率）、成本、延迟。在同一组 Issue 上与 Cursor Background Agents 和 AWS Remote SWE Agents 进行对比。

## 使用指南

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付指南

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云端工作节点，能够将已标记的 Issue 转化为成本可控、凭证作用域受限的、可供审查的 PR。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 30 个 Issue 的通过率 | 端到端成功（CI 全绿 + 覆盖率达标） |
| 20 | PR 质量 | Diff 大小、覆盖率变化量、风格规范符合度 |
| 20 | 每个已解决 Issue 的成本与延迟 | 每个 PR 的美元成本与实际耗时 |
| 20 | 安全性 | 作用域令牌、按仓库预算、禁止强制推送、凭证清理 |
| 15 | 运维体验 | 执行说明注释、重试入口、@提及跟进 |
| **100** | | |

## 练习

1. 添加“修复不稳定测试”模式：标签 `@agent stabilize-flake TestX` 会在沙箱内将测试运行 50 次，并提出最小化修改以使其稳定。
2. 在三个共享 Issue 上对比与 Cursor Background Agents 的成本。报告各工具在哪些场景下表现更优。
3. 实现预算仪表盘：按仓库每日成本、按用户成本。异常时发出告警。
4. 构建“试运行”模式：在不运行 CI 的情况下打开草稿 PR，以便审查者低成本地查看计划。
5. 添加保留策略：超过 7 天未合并的 PR 分支将自动删除。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | “作用域机器人身份” | 具备细粒度权限 + 短期有效安装令牌的 App |
| 异步云端代理 | “后台代理” | 在云端沙箱中运行的非交互式工作节点，而非终端 |
| 环境推断 | “Dockerfile 合成” | 检测语言与包管理器，若缺失则生成 Dockerfile |
| 验证 | “沙箱内 CI” | 在打开 PR 前于工作节点内运行完整测试套件 |
| 覆盖率变化量 | “覆盖率保持” | 从基础分支到代理分支的测试覆盖率百分比变化 |
| 按仓库预算 | “每日上限” | 在调度器端执行的美元金额与 PR 数量上限 |
| 执行说明 | “PR 正文解释” | 代理对变更内容及原因的总结；需包含在 PR 正文中 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) —— 权威的异步云端代理参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) —— CLI 参考文档
- [Cursor Background Agents](https://docs.cursor.com/background-agent) —— 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) —— 托管竞品
- [Google Jules](https://jules.google) —— Google 的托管版本
- [Factory Droids](https://www.factory.ai) —— 其他商业参考
- [GitHub App documentation](https://docs.github.com/en/apps) —— 作用域机器人身份
- [Daytona cloud sandboxes](https://daytona.io) —— 参考沙箱
