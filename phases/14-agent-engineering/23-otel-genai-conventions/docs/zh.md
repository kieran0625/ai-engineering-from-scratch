# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（于 2024 年 4 月启动）定义了代理遥测的标准模式。各厂商在 Span 名称、属性和内容捕获规则上趋于统一，使得代理追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同的含义。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 24（可观测性平台）
**耗时：** 约 60 分钟

## 学习目标

- 说出 GenAI Span 的分类：模型/客户端、代理、工具。
- 区分 `invoke_agent` CLIENT 与 INTERNAL Span，并说明各自的适用场景。
- 列出顶层 GenAI 属性：提供商名称、请求模型、数据源 ID。
- 解释内容捕获契约：可选开启、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用推荐。

## 问题所在

每个厂商都发明自己的 Span 名称。运维团队最终不得不为每个框架单独构建仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一个全生态系共同遵循的标准来解决此问题。

## 核心概念

### Span 分类

1. **模型/客户端 Span。** 覆盖原始的 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器生成。
2. **代理 Span。** `create_agent`（代理构造时）和 `invoke_agent`（代理运行时）。
3. **工具 Span。** 每次工具调用对应一个；通过父子关系与代理 Span 关联。

### 代理 Span 命名

- Span 名称：若已命名则为 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- Span 类型：
  - **CLIENT** —— 用于远程代理服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** —— 用于进程内代理框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` —— `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` —— 模型 ID。
- `gen_ai.response.model` —— 解析后的模型（可能因路由策略与请求模型不同）。
- `gen_ai.agent.name` —— 代理标识符。
- `gen_ai.operation.name` —— `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` —— 用于 RAG：指明查询了哪个语料库或存储。

Anthropic、Azure AI Inference、AWS Bedrock 和 OpenAI 均有特定的技术约定。

### 内容捕获

默认规则：插桩默认不应捕获输入/输出。需通过以下方式显式启用：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产环境模式：将内容存储在外部（如 S3 或日志存储），在 Span 中记录引用（指针 ID，而非文本内容）。这是 Lesson 27 中将内容投毒防御机制集成到可观测性中的做法。

### 稳定性

截至 2026 年 3 月，大多数约定仍处于实验阶段。可通过以下方式选择加入稳定预览版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生支持将 GenAI 属性映射到其 LLM 可观测性模式中。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 常见误区

- **在 Span 中捕获完整提示词。** 包含 PII、密钥和客户数据的追踪信息会被运维人员读取。应将其存储在外部。
- **缺少 `gen_ai.provider.name`。** 当缺乏归属信息时，多提供商仪表盘将无法正常工作。
- **Span 缺少父级链接。** 导致孤立工具 Span。务必始终传播上下文。
- **未设置稳定性选择加入。** 在后端升级时，你的属性可能会被重命名。

## 动手实践

`code/main.py` 实现了一个符合 GenAI 约定的标准库 Span 发射器：

- `Span`，带有 GenAI 属性模式。
- `Tracer`，带有 `start_span` 和嵌套上下文。
- 脚本化代理运行流程，会生成：`create_agent`、`invoke_agent`（INTERNAL）、每个工具的 Span，以及用于 LLM 调用的 `chat` Span。
- 一种内容捕获模式，将提示词存储在外部并在 Span 中记录 ID。

运行方式：

```
python3 code/main.py
```

输出：包含所有必需 GenAI 属性的 Span 树，以及显示可选内容引用的“外部存储”。

## 使用方式

- **Datadog LLM 可观测性**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（Lesson 24）—— 自动为生态系添加插桩。
- **Jaeger / Honeycomb / Grafana Tempo** —— 原始 OTel 追踪；基于 GenAI 属性构建仪表盘。
- **自托管** —— 运行带有 GenAI 处理器的 OTel Collector。

## 交付部署

`outputs/skill-otel-genai.md` 将 OTel GenAI Span 集成到现有代理中，采用默认的内容捕获配置和外部引用存储。

## 练习

1. 使用 `invoke_agent`（INTERNAL）及每个工具的 Span 为你的 Lesson 01 ReAct 循环添加插桩。发送至 Jaeger 实例。
2. 以“仅引用”模式添加内容捕获：将提示词存入 SQLite，Span 属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其集成到你的 Lesson 09 Mem0 搜索中。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性未被 Collector 重命名。
5. 构建仪表盘：仅基于 GenAI 属性分析“哪些工具错误与哪些模型相关”。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI SIG | “OpenTelemetry GenAI 小组” | 负责定义模式的 OTel 工作组 |
| invoke_agent | “代理 Span” | 表示代理运行的 Span 名称 |
| CLIENT span | “远程调用” | 指向远程代理服务的调用 Span |
| INTERNAL span | “进程内” | 进程内代理运行的 Span |
| gen_ai.provider.name | “提供商” | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | “RAG 来源” | 检索命中对应的语料库/存储 |
| Content capture | “提示词日志” | 消息的可选捕获；生产环境中应存至外部 |
| Stability opt-in | “预览模式” | 用于锁定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 规范文档
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 默认生成 GenAI Span
- [AutoGen v0.4 (微软研究院)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 内置 OTel Span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) —— W3C 追踪上下文传播
