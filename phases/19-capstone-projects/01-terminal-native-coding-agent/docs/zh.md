# 综合项目 01 —— 终端原生编码智能体

> 到 2026 年，编码智能体的形态已经定型。一个 TUI 控制台、一个有状态的计划器、一个沙盒化的工具层、一个规划-执行-观察-恢复的循环。从 50 英尺外看，Claude Code、Cursor 3 和 OpenCode 看起来都一样。本综合项目要求你从头到尾构建一个——输入 CLI，输出 Pull Request——并在 SWE-bench Pro 上将其与 mini-swe-agent 和 Live-SWE-agent 进行对比测量。你将学到难点不在于模型调用，而在于工具循环、沙盒环境以及 50 轮运行中的成本上限。

**类型：** 综合项目
**语言：** TypeScript / Bun（控制台运行时），Python（评估脚本）
**前置知识：** 第 11 阶段（LLM 工程）、第 13 阶段（工具与协议）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**预计耗时：** 35 小时

## 问题背景

编码智能体在 2026 年成为主导性的 AI 应用类别。Claude Code（Anthropic）、带 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（11.2 万星）、Factory Droids 以及 Google Jules 均采用了同一架构的变体：一个终端控制台、一个受权限控制的工具层、一个沙盒环境，以及围绕前沿模型构建的“规划-执行-观察”循环。技术前沿虽然狭窄——Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到了 79.2% 的得分——但工程实践的范畴却非常广泛。大多数故障模式并非模型本身的错误。它们通常是工具循环不稳定、上下文污染、失控的 Token 消耗以及破坏性的文件系统操作。

你无法仅从外部去推演这些智能体的行为。你必须亲自构建一个，亲眼看着它在第 47 轮因为 ripgrep 返回了 8MB 的匹配结果而崩溃，然后重新构建截断层。这正是本综合项目的核心目的。

## 概念设计

控制台包含四个交互面。**计划（Plan）**维护一个类似 TodoWrite 的状态对象，模型会在每一轮中重写它。**执行（Act）**分发工具调用（读取、编辑、运行、搜索、git）。**观察（Observe）**捕获 stdout / stderr / 退出码，进行截断，并将摘要反馈回去。**恢复（Recover）**处理工具错误，且不撑爆上下文窗口或陷入无限循环。2026 年的形态还增加了一项功能：**钩子（Hooks）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` —— 这些是可配置的扩展点，操作员可在此注入策略、遥测数据和护栏机制。

沙盒环境采用 E2B 或 Daytona。每个任务都在一个全新的 devcontainer 中运行，并挂载了一个可读写的 git worktree。控制台绝不会触碰宿主机文件系统。无论成功还是失败，worktree 都会被销毁。成本控制通过三层机制强制执行：每轮 Token 上限、每次会话的美元预算，以及硬性轮次限制（通常为 50 轮）。可观测性层基于带有 GenAI 语义规范的 OpenTelemetry spans，数据发送至自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 控制台运行时：Bun 1.2 + Ink 5（终端内 React）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最困难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙盒：E2B 沙盒（JS SDK）或 Daytona devcontainer
- 代码搜索：ripgrep 子进程，17 种语言的 tree-sitter 解析器（预编译）
- 隔离性：每个任务使用 `git worktree add`，成功/失败后清理
- 评估控制台：SWE-bench Pro（已验证子集）+ Terminal-Bench 2.0 + 你自己准备的 30 个任务保留集
- 可观测性：OpenTelemetry SDK，配合 `gen_ai.*` 语义规范 → 自托管 Langfuse
- PR 发布：GitHub App，使用细粒度 Token，作用域仅限于目标仓库

## 构建指南

1. **TUI 与命令循环。** 使用 Ink 搭建 Bun 项目脚手架。接收 `agent run <repo> "<task>"`。打印分屏视图：顶部为计划窗格，中部为工具调用流，底部为 Token 预算。添加 Ctrl-C 取消功能，退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义一个类型化的 TodoWrite 模式（包含 pending / in_progress / done 项及备注）。模型在每一轮中以工具调用的形式重写完整状态 —— 不要允许其增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后可以恢复。

3. **工具层。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时机制）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使控制台与传输协议解耦。每个工具返回截断后的输出（单次调用上限 4k Token）。

4. **沙盒封装。** 每个任务启动一个 E2B 沙盒。`git worktree add -b agent/$TASK_ID` 一个新分支。所有工具调用均在沙盒内执行。宿主机文件系统不可访问。

5. **钩子。** 实现全部八种 2026 钩子类型。接入至少四个用户自定义钩子：(a) `PreToolUse` 破坏性命令守卫，用于拦截 worktree 外的 `rm -rf`；(b) `PostToolUse` Token 记账；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终追踪包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个 Issue 子集。针对每个 Issue 运行你的控制台。与 mini-swe-agent（最小基线）在 pass@1、每任务轮次和每任务美元成本上进行对比。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截止阈值：50 轮、200k 上下文、每任务 $5。`PreCompact` 钩子在达到 150k Token 时将较早的轮次摘要为前置状态块，从而为新观察腾出空间，同时不丢失计划。

8. **PR 发布。** 成功后，最后一步是 `git push` + 一次 GitHub API 调用，以打开一个 PR，并在正文中包含计划和 diff 摘要。

## 使用方式

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付标准

交付技能位于 `outputs/skill-terminal-coding-agent.md`。给定仓库路径和任务描述，它将在沙盒中运行完整的“规划-执行-观察”循环，并返回 PR URL 及追踪包。本综合项目的评分标准如下：

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 对比基线 | 你的控制台 vs mini-swe-agent 在 30 个匹配的 Python 任务上的表现 |
| 20 | 架构清晰度 | 计划/执行/观察分离、钩子接口、工具模式 —— 对照 Live-SWE-agent 布局进行审查 |
| 20 | 安全性 | 沙盒逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | 追踪完整性（100% 工具调用均有 span）、每轮 Token 记账 |
| 15 | 开发者体验 | 冷启动 < 2秒、崩溃恢复可继续计划、Ctrl-C 能干净地取消正在进行的工具调用 |
| **100** | | |

## 练习

1. 将底层模型从 Claude Sonnet 4.7 替换为 vLLM 部署的 Qwen3-Coder-30B。对比 pass@1 和每任务美元成本。报告开源模型表现不佳的具体环节。

2. 添加一个 `reviewer` 子智能体，在发布 PR 前读取 diff，并可请求修订循环。测量虚假审查是否会导致 SWE-bench 通过率低于单智能体基线（提示：通常确实会下降）。

3. 对沙盒进行压力测试：编写一个尝试 `curl` 外部 URL 的任务，以及一个向 worktree 外部写入的任务。确认两者均被 PreToolUse 钩子拦截。记录这些尝试。

4. 使用较小模型（Haiku 4.5）实现 `PreCompact` 摘要功能。测量在 3 倍压缩率下计划保真度损失了多少。

5. 将 MCP StreamableHTTP 传输替换为 stdio。基准测试冷启动时间和单次调用延迟。为纯本地使用场景选出优胜方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Harness | “智能体循环” | 围绕模型编写的代码，负责分发工具、维护计划状态并强制执行预算 |
| Hook | “智能体事件监听器” | 由用户编写的脚本，在控制台定义的八个生命周期事件中之一触发运行 |
| Worktree | “Git 沙盒” | 位于独立路径的链接 Git 检出目录；可独立销毁而不影响主克隆仓库 |
| TodoWrite | “计划状态” | 模型在每一轮中重写的类型化列表，包含待办/进行中/已完成项 |
| StreamableHTTP | “MCP 传输协议” | 2026 MCP 修订版：具有双向流功能的长连接 HTTP 连接；取代了 SSE |
| Token ceiling | “上下文预算” | 输入+输出 Token 的单轮或单会话上限；触发压缩或终止 |
| pass@1 | “单次尝试通过率” | 首次运行即解决的 SWE-bench 任务比例，不允许重试或窥探测试集 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) —— Anthropic 提供的参考控制台实现
- [Cursor 3 更新日志](https://cursor.com/changelog) —— Agent Tabs 和 Composer 2 的产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) —— 用于 SWE-bench 控制台对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) —— 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) —— 开源控制台，11.2 万星
- [SWE-bench Pro 排行榜](https://www.swebench.com) —— 本综合项目所针对的评估基准
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 工具调用与 Token 使用的 Span 模式
