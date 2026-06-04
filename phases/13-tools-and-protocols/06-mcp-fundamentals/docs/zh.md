# MCP 基础 — 原语、生命周期与 JSON-RPC 基础

> 在 MCP 出现之前，每一项集成都是孤立的。模型上下文协议（Model Context Protocol）最初由 Anthropic 于 2024 年 11 月推出，现由 Linux 基金会旗下的 Agentic AI Foundation 维护，它标准化了发现与调用机制，使得任何客户端都能与任何服务端通信。2025-11-25 版规范定义了六种原语（服务端三种、客户端三种）、三阶段生命周期以及基于 JSON-RPC 2.0 的传输格式。掌握这些核心概念后，本阶段其余关于 MCP 的内容将变得易于理解。

**类型：** 学习
**语言：** Python（标准库、JSON-RPC 解析器）
**前置要求：** 第 13 阶段 · 01 至 05（工具接口与函数调用）
**预计时间：** 约 45 分钟

## 学习目标

- 说出所有六种 MCP 原语的名称（服务端的 tools、resources、prompts；客户端的 roots、sampling、elicitation），并各举一个使用场景。
- 梳理三阶段生命周期（initialize、operation、shutdown），并说明每个阶段由谁发送哪些消息。
- 解析并生成 JSON-RPC 2.0 的请求、响应和通知信封结构。
- 解释 `initialize` 处的能力协商机制是什么，以及缺少它会引发什么问题。

## 问题背景

在 MCP 出现之前，每个使用工具的 Agent 都有自己专属的协议。Cursor 有一套类似 MCP 但互不兼容的工具系统。Claude Desktop 自带另一套不同的系统。VS Code 的 Copilot 扩展又有第三套。一个团队开发了一个“Postgres 查询”工具，却不得不为三个不同的主机 API 分别编写三次。复用该工具只能靠复制粘贴代码。

这导致一次性集成方案如寒武纪大爆发般涌现，同时生态系统的演进速度遇到了瓶颈。

MCP 通过标准化传输格式解决了这一问题。同一个 MCP 服务端可以无缝接入所有 MCP 客户端：截至 2026 年 4 月，已有 Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf 等 300 多个客户端支持。SDK 月下载量达 1.1 亿次，公开服务端超过 10,000 个。Linux 基金会于 2025 年 12 月在新的 Agentic AI Foundation 下接管了该项目。

本阶段使用的规范版本为 **2025-11-25**。该版本新增了异步任务（SEP-1686）、URL 模式提示输入（SEP-1036）、带工具调用的采样（SEP-1577）、增量范围授权（SEP-835）以及 OAuth 2.1 资源指示符语义。第 13 阶段 · 09 至 16 节将详细讲解这些扩展功能。本课仅聚焦基础内容。

## 核心概念

### 三种服务端原语

1. **Tools。** 可调用的操作。遵循第 13 阶段 · 01 中的四步循环。
2. **Resources。** 暴露的数据。只读内容，通过 URI 寻址：`file:///path`、`db://query/...` 及自定义协议。
3. **Prompts。** 可复用的模板。在主界面中以斜杠命令形式呈现；服务端提供模板，客户端填充参数。

### 三种客户端原语

4. **Roots。** 服务端允许访问的 URI 集合。由客户端声明，服务端必须遵守。
5. **Sampling。** 服务端请求客户端的模型执行补全操作。这使得服务端托管的 Agent 循环无需服务端 API 密钥即可运行。
6. **Elicitation。** 服务端在交互过程中向客户端用户请求结构化输入。支持表单或 URL（SEP-1036）。

MCP 中的每项能力都严格归属于这六种之一。第 13 阶段 · 10 至 14 节将对每种能力进行深度讲解。

### 传输格式：JSON-RPC 2.0

每条消息都是一个包含以下字段的 JSON 对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` —— 无 `id`，不期望收到响应。

基础规范包含约 15 种方法，按原语分组。其中重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务端到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize。**

客户端发送 `initialize`，附带其 `capabilities` 和 `clientInfo`。服务端回复自己的 `capabilities`、`serverInfo` 以及所支持的规范版本。客户端消化完响应后发送 `notifications/initialized`。此后，双方均可根据协商好的能力发送请求。

**阶段 2：operation。**

双向通信。客户端调用 `tools/list` 进行发现，随后调用 `tools/call` 进行调用。若服务端声明了相应能力，可能会发送 `sampling/createMessage`。当服务端工具集发生变更时，可能发送 `notifications/tools/list_changed`。当用户更改根目录范围时，客户端可能发送 `notifications/roots/list_changed`。

**阶段 3：shutdown。**

任一方关闭传输通道。MCP 中没有结构化的关闭方法；传输层（stdio 或 Streamable HTTP，见第 13 阶段 · 09）负责传递连接结束信号。

### 能力协商

`capabilities` 是 `initialize` 握手过程中的契约。服务端示例如下：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务端声明它可以发送 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自身能力表示同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端未声明 `sampling`，服务端绝不应调用 `sampling/createMessage`。反之亦然：如果服务端未声明 `resources.subscribe`，客户端也不应尝试订阅。

这正是防止生态碎片化的关键。不支持采样的客户端依然是合法的 MCP 客户端；不调用 `sampling` 的服务端依然是合法的 MCP 服务端。它们只是不会共同使用该特性而已。

### 结构化内容与错误形态

`tools/call` 返回一个包含类型化块的 `content` 数组：`text`、`image`、`resource`。第 13 阶段 · 14 节在此列表中增加了 MCP Apps（`ui://` 交互式 UI）。

错误处理沿用 JSON-RPC 错误码。规范新增的定义包括：`-32002` “Resource not found”、`-32603` “Internal error”，以及作为 `error.data` 的 MCP 专属错误数据。

### 客户端能力 vs 工具调用细节

常见的混淆点：`capabilities.tools` 指的是客户端是否支持 tool-list-changed 通知。客户端是否会实际调用特定工具是由其底层模型决定的运行时选择，而非能力标志位。能力标志位是规范层面的契约，而模型的决策是正交独立的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010 年发布）是一种轻量级的双向协议。REST 则是客户端发起的。MCP 需要服务端主动发起的消息（如采样、通知），因此具有对称请求/响应结构的 JSON-RPC 是天然的选择。此外，JSON-RPC 能够干净地构建在 stdio 和 WebSocket/Streamable HTTP 之上，而无需重新发明 HTTP 的请求结构。

## 动手实践

`code/main.py` 提供了一个极简的 JSON-RPC 2.0 解析器与生成器，随后手动演示 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，并打印每条消息。不涉及真实传输层；仅展示消息结构。请对照“Further Reading”中链接的规范文档来验证每个信封结构。

重点关注以下内容：

- `initialize` 双向声明能力；响应中包含 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回 `tools` 数组；每条记录包含 `name`、`description`、`inputSchema`。
- `tools/call` 使用了 `params.name` 和 `params.arguments`。
- 响应 `content` 是一个 `{type, text}` 块数组。

## 交付成果

本课的最终产出是 `outputs/skill-mcp-handshake-tracer.md`。给定一份类 pcap 格式的 MCP 客户端-服务端交互日志，该技能会为每条消息标注其所属的原语、生命周期阶段以及依赖的能力标志。

## 练习

1. 运行 `code/main.py`。找出发生能力协商的代码行，并描述如果服务端未声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息结构为：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长耗时 `tools/call` 执行期间生成该消息，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范全文（约 80 页）。找出绝大多数服务端都不需要的一个能力标志位。提示：它与资源订阅有关。

4. 在纸上草绘一个假设的“定时任务（cron job）”功能应归属哪种原语。（提示：服务端希望客户端在预定时间调用它。目前六种原语均不完全契合。）MCP 的 2026 路线图已为此准备了草案 SEP。

5. 解析 GitHub 上一个开源 MCP 服务端的一次会话日志。统计请求、响应和通知消息的数量。计算生命周期流量与运行期流量的占比。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP | “Model Context Protocol” | 用于模型到工具发现与调用的开放协议 |
| Server primitive | “服务端暴露的内容” | tools（操作）、resources（数据）、prompts（模板） |
| Client primitive | “客户端允许服务端使用的功能” | roots（作用域）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | “传输格式” | 对称的请求/响应/通知信封结构 |
| `initialize` handshake | “能力协商” | 首对消息；服务端与客户端声明各自支持的功能 |
| `tools/list` | “发现” | 客户端向服务端询问当前工具集 |
| `tools/call` | “调用” | 客户端请求服务端执行带参数的工具 |
| `notifications/*_changed` | “变更事件” | 服务端通知客户端其原语列表已更新 |
| Content block | “类型化结果” | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | “Spec Evolution Proposal” | 命名草案提案（例如用于异步任务的 SEP-1686） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 官方规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) —— 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) —— 2024 年 11 月发布博文
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) —— 一周年回顾与 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) —— SEP-1686、1036、1577、835 及 1724 摘要
