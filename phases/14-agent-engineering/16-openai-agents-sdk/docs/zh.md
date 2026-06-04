# OpenAI Agents SDK：Handoffs、Guardrails、Tracing

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体框架。包含五个核心原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 是名为 `transfer_to_<agent>` 的工具。Guardrail 会在输入或输出时触发。Tracing 默认开启。

**类型：** 学习 + 实践
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（Agent Loop）、Phase 14 · 06（Tool Use）
**耗时：** 约 75 分钟

## Learning Objectives

- 说出 OpenAI Agents SDK 的五个核心原语。
- 解释 Handoff：为何将其建模为工具、模型看到的命名格式是什么，以及上下文如何传递。
- 区分输入 Guardrail、输出 Guardrail 和工具 Guardrail；解释 `run_in_parallel` 与 Blocking 模式的区别。
- 实现一个支持 Handoff + Guardrail + Span 风格追踪的标准库运行时。

## The Problem

无法清晰委派任务的智能体会最终把所有内容塞进同一个 Prompt 中。没有 Guardrail 的智能体会泄露 PII、产生违反策略的输出，或陷入无限循环。OpenAI 的 SDK 将使多智能体工作变得可控的三个核心原语进行了标准化定义。

## The Concept

### Five primitives

1. **Agent。** LLM + 指令 + 工具 + Handoff。
2. **Handoff。** 向另一个智能体委派任务。在模型侧表现为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（仅限首个智能体）、输出（仅限末个智能体）或工具调用（每个函数工具）进行验证。
4. **Session。** 跨轮次的自动对话历史记录。
5. **Tracing。** 为 LLM 生成、工具调用、Handoff、Guardrail 提供内置的 Span。

### Handoffs as tools

模型在其工具列表中会看到 `transfer_to_billing_agent`。调用该工具会指示运行时执行以下操作：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 功能进行折叠）。
2. 使用其指令初始化目标智能体。
3. 使用目标智能体继续运行。

这是 Supervisor Pattern（第 13 课 / 第 28 课）的产品化实现。

### Guardrails

三种类型：

- **输入 Guardrail。** 在首个智能体的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围请求。
- **输出 Guardrail。** 在末个智能体的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具 Guardrail。** 针对每个函数工具运行。验证参数、检查权限、审计执行过程。

模式：

- **Parallel**（默认）。Guardrail LLM 与主 LLM 并行运行。降低长尾延迟。若被触发，主 LLM 的工作将被丢弃（造成 Token 浪费）。
- **Blocking**（`run_in_parallel=False`）。Guardrail LLM 先运行。若被触发，则不会在主调用上浪费 Token。

Tripwire 会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每次 LLM 生成、工具调用、Handoff 和 Guardrail 都会生成一个 Span。`OPENAI_AGENTS_DISABLE_TRACING=1` 用于关闭追踪。`add_trace_processor(processor)` 可将 Span 分流至您自己的后端，与 OpenAI 的后端同时接收。

### Sessions

`Session` 将对话历史记录存储在后端（SQLite、Redis 或自定义存储）中。`Runner.run(agent, input, session=session)` 会自动加载并追加记录。

### Where this pattern goes wrong

- **Handoff 漂移。** 智能体 A 委派给智能体 B，智能体 B 又委派回智能体 A。需添加跳数计数器。
- **Guardrail 绕过。** 工具 Guardrail 仅在函数工具上触发；内置工具（文件读取器、网页抓取）需要单独的策略。
- **过度追踪。** Span 中包含敏感内容。需配合 OTel GenAI 内容捕获规则（第 23 课）使用——外部存储，通过 ID 引用。

## Build It

`code/main.py` 在标准库中实现了该 SDK 的结构：

- `Agent`、`FunctionTool`、`Handoff`（作为具有转移语义的函数工具）。
- `Runner`，包含输入/输出/工具 Guardrail、Handoff 分发逻辑和跳数计数器。
- 一个简单的 Span 发射器，用于展示追踪结构。
- 一个分诊智能体，根据用户查询委派给计费或支持模块；其中一个输入会触发 Guardrail。

运行方式：

```
python3 code/main.py
```

追踪结果展示了两次成功的 Handoff、一次输入 Guardrail 触发，以及一个与真实 SDK 输出结构一致的 Span 树。

## Use It

- **OpenAI Agents SDK**：适用于以 OpenAI 为核心的产品。
- **Claude Agent SDK**（第 17 课）：适用于以 Claude 为核心的产品。
- **LangGraph**（第 13 课）：当您需要显式状态管理和持久化恢复时使用。
- **Custom**：当您需要精确控制（如语音交互、多提供商集成、联邦部署）时使用。

## Ship It

`outputs/skill-agents-sdk-scaffold.md` 用于脚手架搭建一个 Agents SDK 应用，包含分诊智能体、Handoff、输入/输出/工具 Guardrail、Session 存储和追踪处理器。

## Exercises

1. 添加 Handoff 跳数计数器：超过 N 次转交后拒绝服务。追踪其行为。
2. 将 `nest_handoff_history` 实现为可选配置——在转交前将历史消息折叠为一条摘要。
3. 编写一个 Blocking 模式的输出 Guardrail。对比会被触发的提示词与正常通过的提示词之间的延迟差异。
4. 将 `add_trace_processor` 接入 JSON 日志记录器。它每个 Span 输出的数据结构是什么样的？
5. 阅读 SDK 文档。将您的标准库玩具项目移植到 `openai-agents-python`。您最初建模时有哪些错误？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "LLM + instructions" | Agent type in the SDK; owns tools and handoffs |
| Handoff | "Transfer" | Tool the model calls to delegate to another agent |
| Guardrail | "Policy check" | Validation on input / output / tool invocation |
| Tripwire | "Guardrail trip" | Exception raised when guardrail rejects |
| Session | "History store" | Conversation memory persisted between runs |
| Tracing | "Spans" | Built-in observability over LLM + tool + handoff + guardrail |
| Blocking guardrail | "Sequential check" | Guardrail runs first; no token waste on trip |
| Parallel guardrail | "Concurrent check" | Guardrail runs alongside; lower latency, wastes tokens on trip |

## Further Reading

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — primitives, handoffs, guardrails, tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude-flavored counterpart
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when to reach for handoffs at all
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the standard Agents SDK spans map to
