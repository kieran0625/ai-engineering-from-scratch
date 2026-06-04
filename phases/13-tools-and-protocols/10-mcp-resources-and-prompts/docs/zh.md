# MCP 资源与提示词 —— 超越工具的上下文暴露

> 工具占据了 MCP 90% 的注意力。另外两个服务器原语解决的是不同的问题。资源用于暴露可读数据；提示词以斜杠命令的形式暴露可复用模板。许多服务器应当使用资源来替代将读取操作包装在工具中，并使用提示词来替代在客户端提示词中硬编码工作流。本课将阐明选型原则，并逐步讲解 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（标准库、资源与提示词处理器）
**前置条件：** 第 13 阶段 · 07（MCP 服务器）
**耗时：** 约 45 分钟

## 学习目标

- 针对特定领域，决定将某项能力暴露为工具、资源还是提示词。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 使用参数模板实现 `prompts/list` 和 `prompts/get`。
- 识别宿主何时将提示词作为斜杠命令展示，而非自动注入的上下文。

## 问题所在

一个朴素的笔记应用 MCP 服务器将所有功能都暴露为工具：`notes_read`、`notes_list`、`notes_search`。这会将每次数据访问都包装成由模型驱动的工具调用。后果如下：

- 对于每一个可能受益于上下文的查询，模型都必须自行决定是否调用 `notes_read`。
- 只读内容无法被订阅，也无法流式传输到宿主的侧边栏面板。
- 客户端 UI（如 Claude Desktop 的资源附件面板、Cursor 的“包含文件”选择器）无法直接展示这些数据。

正确的拆分方式：将数据暴露为资源，将修改或计算型操作暴露为工具，将可复用的多步工作流暴露为提示词。每个原语都有其对应的用户交互特性和访问模式。

## 核心概念

### 工具、资源与提示词 —— 选型原则

| 能力 | 原语 |
|------------|-----------|
| 用户希望搜索、过滤或转换数据 | 工具 (tool) |
| 用户希望宿主将此数据作为上下文包含进来 | 资源 (resource) |
| 用户希望拥有一个可重复运行的模板化工作流 | 提示词 (prompt) |

指导原则：如果模型在每次相关查询时都能从中受益而调用它，则应设计为工具。如果用户能从中受益并将其附加到对话中，则应设计为资源。如果整个多步工作流是用户希望复用的基本单位，则应设计为提示词。

### 资源 (Resources)

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的对象：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义协议）
- `memory://session-2026-04-22/recent`（服务器专属）

`contents[]` 同时支持文本和二进制格式。二进制数据使用 `blob` 作为 Base64 编码字符串，并附带一个 `mimeType`。

### 资源订阅

在能力声明中注册 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源发生变化时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

使用场景：一个资源对应磁盘文件的笔记服务器；文件监听器触发更新通知；当文件在宿主外部被编辑时，Claude Desktop 会重新将该文件拉取到上下文中。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 允许你暴露一个带参数的 URI 模式：`notes://{id}`，并将 `id` 作为补全目标。客户端可以在资源选择器中对 ID 进行自动补全。

### 提示词 (Prompts)

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示词是一个模板，填充后会生成一个消息列表，供宿主提供给其模型。例如，一个 `code_review` 提示词接收一个 `file_path` 参数，并返回一个包含三条消息的序列：一条系统消息、一条包含文件正文的用户消息，以及一条带有推理模板的助手开场消息。

### 宿主与提示词

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将提示词暴露为斜杠命令。用户输入 `/code_review` 并从表单中选择参数。服务器的提示词充当了“用户快捷指令”与“发送给模型的完整提示词”之间的契约。

并非所有客户端都支持提示词——请检查能力协商结果。如果一个服务器声明了提示词能力，但客户端不支持，那么该客户端根本不会看到这些斜杠命令。

### “列表变更”通知

资源和提示词在集合发生变动时都会发出 `notifications/list_changed`。一个刚刚导入了 20 条新笔记的服务器会发出 `notifications/resources/list_changed`；客户端随后重新调用 `resources/list` 以获取新增内容。

### 内容类型约定

文本格式：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制格式：`image/png`、`application/pdf`，以及 `blob` 字段。
MCP 应用（第 14 课）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### 动态资源

资源 URI 不一定对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端能够按 URI 进行缓存，则该 URI 必须保持稳定。如果计算是一次性的，URI 应包含时间戳或随机数（nonce），以防止客户端缓存过期失效。

### 订阅与轮询

支持订阅的客户端通过 `notifications/resources/updated` 接收服务器推送。尚未订阅的客户端或不支持该功能的宿主则通过重新读取来进行轮询。两者均符合规范。服务器的能力声明会告知客户端它支持哪种方式。

订阅的成本：服务器需要维护每会话的状态（谁订阅了什么）。需保持订阅集合有界；断开的客户端应设置超时机制。

### 提示词与系统提示词

MCP 中的提示词并非系统提示词。宿主的系统提示词（其自身的运行指令）与 MCP 提示词（由服务器提供、由用户触发的模板）是并列存在的。表现良好的客户端绝不会让服务器提示词覆盖其自身的系统提示词；而是将它们分层叠加。

## 动手实践

`code/main.py` 在第 07 课的笔记服务器基础上增加了以下内容：

- 逐条笔记的资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个渲染为三消息模板的 `review_note` 提示词。
- 一个文件监听器模拟程序，在笔记被修改时发出 `notifications/resources/updated`。
- 一个始终返回最新五条笔记的 `notes://recent` 动态资源。

运行演示以查看完整流程。

## 交付成果

本课将产出 `outputs/skill-primitive-splitter.md`。给定一个拟议的 MCP 服务器，该技能会将每项能力分类为工具/资源/提示词，并提供理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，并验证 `notifications/resources/updated` 事件是否触发。

2. 添加一个 `resources/list_changed` 发射器：当创建新笔记时，发送该通知以便客户端重新发现。

3. 为 GitHub MCP 服务器设计三个提示词：`summarize_pr`、`triage_issue`、`release_notes`。每个提示词均需包含参数 Schema。提示词主体应可直接运行，无需进一步修改。

4. 选取第 07 课服务器中的一个现有工具，判断它应保留为工具，还是拆分为“资源+工具”对。用一句话说明理由。

5. 阅读规范中的 `server/resources` 和 `server/prompts` 章节。找出 `resources/read` 中那个很少被填充但规范支持的一个字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 资源 (Resource) | “暴露的数据” | 宿主可读取的、可通过 URI 寻址的内容 |
| 资源 URI | “数据指针” | 带协议前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | “监听变更” | 客户端主动选择的、针对特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | “资源已变更” | 向客户端发出的信号，表明已订阅的资源有新内容 |
| 资源模板 | “参数化 URI” | 带有宿主选择器补全提示的 URI 模式 |
| 提示词 (Prompt) | “斜杠命令模板” | 带有参数槽位的命名多消息模板 |
| 提示词参数 | “模板输入” | 宿主在渲染前收集的强类型参数 |
| `prompts/get` | “渲染模板” | 服务器返回填充后的消息列表 |
| 内容块 (Content Block) | “类型化数据块” | `{type: text \| image \| resource \| ui_resource}` |
| 斜杠命令交互 | “用户快捷方式” | 宿主将提示词展示为以 `/` 开头的命令 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) —— 资源 URI、订阅与模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) —— 提示词模板与斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) —— 完整的 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) —— 完整的 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) —— 社区指南，对官方文档的扩展说明
