# 函数调用深度解析 —— OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年收敛于相同的工具调用循环，随后在其他方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 ID 关联机制。本课程将三者并列对比，以便你在移植代码时，确保在一处提供商上运行的代码不会在其他地方出错。

**类型：** 构建
**语言：** Python（标准库、模式转换器）
**前置条件：** 第 13 阶段 · 01（工具接口）
**耗时：** 约 75 分钟

## 学习目标

- 阐述 OpenAI、Anthropic 和 Gemini 函数调用载荷在结构上的三个差异（声明、调用、结果）。
- 将一个工具声明翻译为所有三种提供商格式，并预测严格模式约束会在何处产生差异。
- 在各提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解各提供商的硬性限制（工具数量、模式深度、参数长度），以及违反限制时各自发出的错误特征。

## 问题所在

函数调用请求的结构因提供商而异。以下是来自 2026 年生产栈的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你必须解析的 JSON 字符串。严格模式（`strict: true`）通过受限解码强制执行模式合规性。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应返回为 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`。`input` 已经是解析后的状态（对象而非字符串）。你用一条包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息进行回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 形式到达，其中 `id` 在 Gemini 3 及更高版本中是唯一的，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 进行回复。

相同的循环。不同的字段名、不同的嵌套层级、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写天气代理的团队，仅为了适配基础通信逻辑，就需要花费两天时间移植到 Anthropic，再花一天移植到 Gemini。

本课程将构建一个转换器，将这三种格式统一为一个标准的工具声明，并在边缘层进行路由。第 13 阶段 · 17 会将该模式泛化为一个 LLM 网关。

## 核心概念

### 通用结构

每个提供商都需要以下五个要素：

1. **工具列表。** 每个工具的名称、描述和输入模式。
2. **工具选择。** 强制指定工具、禁止使用工具，或让模型自行决定。
3. **调用输出。** 命名工具和参数的结构化输出。
4. **调用 ID。** 将响应与正确的调用关联（对并行调用很重要）。
5. **结果注入。** 将结果与调用关联起来的消息或块。

### 逐字段结构差异

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 模式字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | JSON 字符串化 | 已解析对象 | 已解析对象 |
| ID 格式 | `call_...`（由 OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | role 为 `tool`，含 `tool_call_id` | 含 `tool_result`、`tool_use_id` 的 `user` | 含匹配 `id` 的 `functionResponse` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式 | `strict: true` | 模式即契约（始终强制执行） | 请求级别的 `responseSchema` |

### 实际会遇到的限制

- **OpenAI。** 每次请求最多 128 个工具。模式深度为 5。参数字符串长度 <= 8192 字节。严格模式要求不包含 `$ref`，不允许 `oneOf`/`anyOf`/`allOf` 存在重叠，且 `required` 中列出的每个属性都必须明确声明。
- **Anthropic。** 每次请求最多 64 个工具。模式深度理论上无界，但实际限制为 10。没有严格模式标志；模式即契约，模型通常会遵守。
- **Gemini。** 每次请求最多 64 个函数。模式类型为 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有差异）。自 Gemini 3 起，并行调用使用唯一 ID。

### `tool_choice` 行为

三方均支持三种模式，仅命名不同。

- **Auto（自动）。** 模型自行选择工具或文本。默认值。
- **Required / Any（必须/任意）。** 模型必须至少调用一个工具。
- **None（无）。** 模型不得调用工具。

此外，每个提供商还有各自独有的模式：

- **OpenAI。** 按名称强制指定特定工具。
- **Anthropic。** 按名称强制指定特定工具；`disable_parallel_tool_use` 标志用于区分单次与多次调用。
- **Gemini。** 无论模型意图如何，`mode: "VALIDATED"` 都会将每条响应路由至模式验证器。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）会在单条助手消息中发出多个调用。你将它们全部执行，并用一条批处理工具角色消息回复，其中包含针对每个 `tool_call_id` 的一条记录。Anthropic 历史上仅支持单次调用；`disable_parallel_tool_use: false`（Claude 3.5 起为默认）启用了多调用支持。Gemini 2 允许并行调用但未提供稳定 ID；Gemini 3 引入了 UUID，使得乱序响应也能清晰关联。

### 流式传输

三者均支持流式工具调用。网络传输格式有所不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你将其累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 块开始/块增量/块停止事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，以便多个并行调用可以交错进行。

第 13 阶段 · 03 将深入探讨并行与流式重组。本课程专注于声明结构与单次调用形态。

### 错误与修复

无效参数错误的表现形式也各不相同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你需注入错误消息并重试调用。
- **OpenAI（严格）。** 验证发生在解码期间；不可能出现无效 JSON，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；模式仅供参考。需在服务端进行验证。
- **Gemini。** OpenAPI 3.0 特性：对象字段上的 `enum` 会被静默忽略；需自行验证。

### 转换器模式

代码中的标准工具声明如下所示（形状由你定义）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小型函数将其转换为三种提供商格式。`code/main.py` 中的测试框架正是如此操作，随后将伪造的工具调用通过每种提供商的响应格式进行往返测试。无需网络连接——本课程教授的是数据结构，而非 HTTP 协议。

生产团队通常将此转换器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。第 13 阶段 · 17 将提供一个网关，在任何一种提供商前端暴露出 OpenAI 风格的 API。

## 实践应用

`code/main.py` 定义了一个标准的 `Tool` 数据类，以及三个用于生成 OpenAI、Anthropic 和 Gemini 声明 JSON 的转换器。随后，它将手工构造的各格式提供商响应解析为同一个标准调用对象，证明其底层语义完全一致。运行它并将三个声明并排对比。

重点关注：

- 三个声明块仅在信封结构和字段名称上有所不同。
- 三个响应块在调用所在位置上的差异（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 单个 `canonical_call()` 函数可从所有三种响应形状中提取 `{id, name, args}`。

## 交付成果

本课程将产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某一提供商的函数调用集成，该技能将生成一份可移植性审计报告：指出其依赖哪些提供商限制、哪些字段需要重命名，以及移植到其他提供商时会发生哪些破坏。

## 练习

1. 运行 `code/main.py`，验证三个提供商的声明 JSON 是否都序列化了相同的底层 `Tool` 对象。修改标准工具以添加枚举参数，并确认只有 Gemini 转换器需要处理 OpenAPI 特性。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，用于提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生不支持此功能；请注意这种不对称性。

3. 实现 `tool_choice` 转换：将标准 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供商格式。然后映射 `mode="any"` 和 `mode="none"`。对照课程中的差异表进行检查。

4. 从三个提供商中任选其一，通读其函数调用指南。找出其模式规范中另外两者不支持的一个字段。候选项：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写测试用例：一个参数违反声明模式的工具调用。将其通过每个提供商的验证器运行（课程 01 中的标准库验证器可作为代理），并记录触发的错误。文档说明出于严格性考虑，你会在生产环境中选择哪个提供商。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|------------------------|------------------------|
| Function calling | “工具使用” | 提供商级的结构化工具调用输出 API |
| Tool declaration | “工具规范” | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | “强制/禁止” | 自动/必须/无/指定名称模式 |
| Strict mode | “模式强制执行” | 限制解码以匹配模式的 OpenAI 标志 |
| `tool_use` block | “Anthropic 调用形态” | 包含 id、name、input 的内联内容块 |
| `functionCall` part | “Gemini 调用形态” | 包含 name、args 和 id 的 `parts[]` 条目 |
| Arguments-as-string | “JSON 字符串化” | OpenAI 将参数作为 JSON 字符串返回，而非对象 |
| Parallel tool calls | “单轮扇出” | 单条助手消息中包含多个工具调用 |
| Refusal | “模型拒绝” | 仅严格模式下出现的拒绝块，而非调用 |
| OpenAPI 3.0 subset | “Gemini 模式特性” | Gemini 使用类 JSON Schema 方言，存在细微差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) —— 包含严格模式和并行调用的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) —— `tool_use` 与 `tool_result` 块的语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) —— 并行调用、唯一 ID 及 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) —— Gemini 的企业级接口
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) —— 严格模式下的模式强制执行详情
