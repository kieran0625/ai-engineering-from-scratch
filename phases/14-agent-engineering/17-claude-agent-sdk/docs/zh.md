# Claude Agent SDK: Subagents and Session Store

> Claude Agent SDK 是 Claude Code harness 的库形式。内置工具、用于上下文隔离的子代理（subagents）、钩子（hooks）、W3C 追踪传播、与 TypeScript 版本一致的会话存储（session store）。Claude Managed Agents 是用于长期异步任务的托管替代方案。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（Agent 循环），第 14 阶段 · 10（技能库）
**耗时：** 约 75 分钟

## Learning Objectives

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness 形态）之间的区别。
- 描述子代理（subagents）——并行化与上下文隔离——以及何时使用它们。
- 说出 Python SDK 的会话存储接口（`append`, `load`, `list_sessions`, `delete`, `list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个基于标准库的 harness，包含内置工具、具有隔离上下文的子代理生成、生命周期钩子和会话存储。

## The Problem

原始的 LLM API 仅提供一次往返调用。生产级 Agent 需要工具执行、MCP 服务器、生命周期钩子、子代理生成、会话持久化和追踪传播。Claude Agent SDK 以库的形式提供这种形态——即 Claude Code 所使用的同一套 harness，现已开放供自定义 Agent 使用。

## The Concept

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`)**。原始 Messages API。你自行掌控循环、工具和状态。
- **Agent SDK (`claude-agent-sdk`)**。内置工具执行、MCP 连接、钩子、子代理生成、会话存储。将 Claude Code 循环封装为库。

### Built-in tools

该 SDK 开箱即用自带 10 多种工具：文件读写、Shell、grep、glob、网页抓取等。自定义工具通过标准的 tool-schema 接口进行注册。

### Subagents

Anthropic 文档中记载了两个用途：

1. **Parallelization**。并发运行独立任务。“为这 20 个模块中的每一个查找测试文件”就是 20 个并行的子代理任务。
2. **Context isolation**。子代理使用独立的上下文窗口；只有结果会返回给编排器（orchestrator）。从而保护编排器的预算不被耗尽。

Python SDK 近期新增功能：`list_subagents()`、`get_subagent_messages()`，用于读取子代理的运行记录（transcripts）。

### Session store

与 TypeScript 版本保持协议一致：

- `append(session_id, message)` — 添加一轮对话。
- `load(session_id)` — 恢复会话。
- `list_sessions()` — 枚举列表。
- `delete(session_id)` — 支持向子代理会话级联操作。
- `list_subkeys(session_id)` — 列出子代理键名。

`--session-mirror`（CLI 参数）会在流式传输时将运行记录镜像到外部文件中，便于调试。

### Hooks

可注册的生命周期钩子：

- `PreToolUse`, `PostToolUse` — 拦截或审计工具调用。
- `SessionStart`, `SessionEnd` — 初始化和清理。
- `UserPromptSubmit` — 在模型看到用户输入前对其进行干预。
- `PreCompact` — 在上下文压缩前运行。
- `Stop` — Agent 退出时清理资源。
- `Notification` — 侧信道告警。

钩子是 pro-workflow（第 14 阶段课程参考）及类似系统添加横切关注点行为的方式。

### W3C trace context

调用方活跃的 OTel span 会通过 W3C trace context 头信息传播到 CLI 子进程中。整个多进程追踪将在你的后端显示为单一追踪链路。

### Claude Managed Agents

托管替代方案（beta 请求头 `managed-agents-2026-04-01`）。适用于长期异步任务，内置提示词缓存和自动压缩。以部分控制权换取托管基础设施。

### Where this pattern goes wrong

- **Subagent over-spawn**。为 100 个微小任务生成 100 个子代理。开销会占据主导。请改用批处理。
- **Hook creep**。每个团队都添加钩子；启动时间急剧膨胀。建议每季度审查一次钩子。
- **Session bloat**。会话不断累积；体积持续增长。请使用 `list_sessions` 配合过期策略。

## Build It

`code/main.py` 在标准库中实现了该 SDK 形态：

- `Tool`, `ToolRegistry` 内置 `read_file`, `write_file`, `list_dir`。
- `Subagent` — 私有上下文、隔离运行、返回结果。
- `SessionStore` — append, load, list, delete, list_subkeys。
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`。
- 演示示例：主 Agent 并行生成 3 个子代理（各自隔离），聚合结果并持久化会话。

运行方式：

```
python3 code/main.py
```

追踪日志展示了子代理的上下文隔离（编排器上下文大小保持有界）、钩子执行与会话持久化。

## Use It

- 若产品以 Claude 为核心且希望采用 Claude Code 的 harness 形态，请使用 **Claude Agent SDK**。
- 若需托管的长期异步任务，请使用 **Claude Managed Agents**。
- 若需 OpenAI 生态的对标方案，请参考 **OpenAI Agents SDK**（第 16 课）。
- 若更倾向于图状状态机架构，可使用 **LangGraph + 自定义工具**。

## Ship It

`outputs/skill-claude-agent-scaffold.md` 用于脚手架搭建一个包含子代理、钩子、会话存储、MCP 服务器挂载及 W3C 追踪传播的 Claude Agent SDK 应用。

## Exercises

1. 添加一个子代理生成器，将 20 个任务分批为每组 5 个并行子代理。测量编排器上下文大小与“每任务一个”方案的对比。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用进行限流（每个会话每分钟 5 次）。追踪其行为表现。
3. 接入 `list_subkeys` 以渲染子代理树。深层嵌套的表现是什么样的？
4. 将该示例代码移植到真实的 `claude-agent-sdk` Python 包中。工具注册方面会发生什么变化？
5. 阅读 Claude Managed Agents 文档。在什么情况下你会从自托管切换到托管模式？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent SDK | "作为库的 Claude Code" | Harness 形态：工具、MCP、钩子、子代理、会话存储 |
| Subagent | "子 Agent" | 独立上下文、独立预算；结果向上汇报 |
| Session store | "会话数据库" | 持久化、加载、列出、删除轮次，支持子代理级联 |
| Hook | "生命周期回调" | 工具前后、会话、提示词提交、压缩、停止时的回调 |
| W3C trace context | "跨进程追踪" | 父级 span 传播至 CLI 子进程 |
| Managed Agents | "托管 harness" | Anthropic 托管的长期异步任务 |
| `--session-mirror` | "运行记录镜像" | 流式传输时将会话轮次写入外部文件 |
| MCP server | "工具表面" | 附加到 Agent 的外部工具/资源源 |

## Further Reading

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产环境模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对标方案
