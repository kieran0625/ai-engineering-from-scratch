# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个智能体调用了五个工具、三个 MCP 服务器和两个子智能体。你需要在整个流程中保持单一 Trace。OpenTelemetry GenAI 语义规范（v1.37 及以上版本的稳定属性）是 2026 年的行业标准，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 均原生支持。本课将列出所需的属性，梳理 Span 层级结构（智能体 → LLM → 工具），并提供一个基于标准库的 Span 发射器，可无缝接入任意 OTel 导出器。

**类型：** 构建
**语言：** Python（标准库、OTel Span 发射器）
**前置条件：** 阶段 13 · 07（MCP 服务器）、阶段 13 · 08（MCP 客户端）
**耗时：** 约 75 分钟

## 学习目标

- 说出 LLM Span 和工具执行 Span 所需的 OTel GenAI 属性。
- 构建覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发的 Trace 层级。
- 决定哪些内容需要捕获（可选开启）与哪些需要脱敏（默认设置）。
- 向本地收集器（Jaeger、Langfuse）发射 Span，且无需重写工具代码。

## 问题背景

2026年2月的一次调试记录：用户反馈“我的智能体有时响应需要 30 秒，有时只需 3 秒。”没有 Trace。日志仅显示了 LLM 调用，未显示工具分发、MCP 服务器往返交互或子智能体。你只能猜测。最终发现：某个 MCP 服务器在冷启动时偶尔会挂起。

如果没有端到端追踪，你将无法定位此问题。OTel GenAI 能解决它。

这些规范由 OpenTelemetry semantic-conventions 工作组于 2025-2026 年敲定。它们定义了稳定的属性名称，确保 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 Span。一次插桩；即可对接任意后端。

## 核心概念

### Span 层级结构

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

所有内容嵌套在同一个 Trace ID 下。Span ID 用于链接父子关系。

### 必需属性

根据 2025-2026 版语义规范：

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — 用于关联的提供商响应 ID。

对于工具 Span：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 具体的调用 ID。
- `gen_ai.tool.description` — 工具描述（可选）。

对于智能体 Span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Span 类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL` 用于智能体自身的循环步骤和工具执行。

### 可选内容捕获

默认情况下，Span 仅携带指标和计时数据——不包含提示词或补全内容。大负载数据和隐私信息（PII）默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 及特定的内容捕获环境变量以包含内容。在生产环境启用前请仔细审查。

### Span 事件

可以将令牌级事件作为 Span 事件添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

Span 内的事件按时间顺序排列，以便进行详细回放。

### 导出器

OTel Span 可导出至：

- **Jaeger / Tempo。** 开源，支持本地部署。
- **Langfuse。** 专为 LLM 可观测性设计；可视化令牌使用情况。
- **Arize Phoenix。** 结合评估与追踪功能。
- **Datadog。** 商业软件；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；查询友好。

它们均使用 OTLP（网络传输格式）。你的代码无需关心底层差异。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，需将 W3C traceparent 头注入请求中。Streamable HTTP 支持标准头。Stdio 本身不携带 HTTP 头；该规范的 2026 路线图讨论了在 JSON-RPC 调用中添加 `_meta.traceparent` 字段。

在该功能上线前：手动在每个请求的 `_meta` 中包含 traceparent。服务器将记录 Trace ID。

### 指标

除 Span 外，GenAI 语义规范还定义了以下指标：

- `gen_ai.client.token.usage` — 直方图。
- `gen_ai.client.operation.duration` — 直方图。
- `gen_ai.tool.execution.duration` — 直方图。

适用于不需要逐次调用细节的仪表盘。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它封装了流行框架（LangGraph、Pydantic AI、CrewAI）以自动发射 OTel Span。如果你的技术栈使用了受支持的框架，这将非常有用；否则请使用手动插桩。

## 实践应用

`code/main.py` 将符合 OTel 格式的 Span 发射到 stdout（采用类似 OTLP-JSON 的格式），适用于一个调用 LLM、分发两个工具并执行一次 MCP 往返交互的智能体。此处不使用真实的导出器——课程重点在于 Span 结构和属性集。可将输出粘贴到兼容 OTLP 的查看器中，或直接阅读。

关注要点：

- Trace ID 在所有 Span 间共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；其中一个场景通过环境变量开启。

## 交付使用

本课将生成 `outputs/skill-otel-genai-instrumentation.md`。给定智能体代码库后，该技能将生成一份插桩计划：标明何处添加 Span、填充哪些属性以及目标导出器。

## 练习

1. 运行 `code/main.py`。统计 Span 数量，并区分哪些是 CLIENT 类型，哪些是 INTERNAL 类型。

2. 开启内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意其对隐私信息（PII）的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，并将其作为每次调用的直方图样本发射。

4. 将父级智能体 Span 中的 traceparent 传播至 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器是否能识别相同的 Trace ID。

5. 阅读 OTel GenAI 语义规范文档。找出规范中列出但本课代码未发射的一个属性，并将其补充进去。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OTel | “OpenTelemetry” | 追踪、指标、日志的开放标准 |
| GenAI semconv | “GenAI 语义规范” | 用于 LLM / 工具 / 智能体 Span 的稳定属性名 |
| `gen_ai.*` | “属性命名空间” | 所有 GenAI 属性共享此前缀 |
| Span | “带计时的操作” | 具有开始、结束和属性的工作单元 |
| Trace | “跨 Span 的祖先关系” | 共享同一 Trace ID 的 Span 树 |
| SpanKind | “CLIENT / SERVER / INTERNAL” | 指示 Span 方向的提示 |
| OTLP | “OpenTelemetry 行协议” | 导出器的网络传输格式 |
| Opt-in content | “提示词/补全捕获” | 默认关闭；可通过环境变量启用 |
| traceparent | “W3C 头” | 在服务间传播追踪上下文 |
| Exporter | “特定后端的数据发送器” | 将 Span 发送至 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI Span、指标和事件的权威规范
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 Span 属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级别的 `invoke_agent` Span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — 托管于 GitHub 的事实来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产环境集成指南
