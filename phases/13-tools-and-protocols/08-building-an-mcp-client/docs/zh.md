# 构建 MCP 客户端 —— 发现、调用与会话管理

> 大多数 MCP 内容都侧重于服务器教程，而对客户端则一带而过。客户端代码才是复杂编排的核心所在：进程启动、能力协商、跨多个服务器的工具列表合并、采样回调、重连以及命名空间冲突解决。本课程将构建一个多服务器客户端，将三个不同的 MCP 服务器整合到一个扁平的工具命名空间中供模型使用。

**类型：** 构建
**语言：** Python（标准库、多服务器 MCP 客户端）
**前置课程：** Phase 13 · 07（构建 MCP 服务器）
**耗时：** 约 75 分钟

## 学习目标

- 将 MCP 服务器作为子进程启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每个服务器的会话状态（能力声明、工具列表、最近一次的通知 ID）。
- 将多个服务器的工具列表合并到一个命名空间中，并处理冲突。
- 将工具调用路由到拥有该工具的服务器，并重组响应。

## 问题背景

真实的 Agent 宿主环境（如 Claude Desktop、Cursor、Goose、Gemini CLI）会同时加载多个 MCP 服务器。用户可能同时运行文件系统服务器、Postgres 服务器和 GitHub 服务器。客户端的职责是：

1. 启动每个服务器。
2. 独立与每个服务器进行握手。
3. 对每个服务器调用 `tools/list` 并展平结果。
4. 当模型发出 `notes_search` 时，在合并的命名空间中查找并将其路由到正确的服务器。
5. 非阻塞地处理来自任何服务器的通知（`tools/list_changed`）。
6. 在传输层故障时进行重连。

手动实现所有这些逻辑是将“玩具项目”与“可用服务”区分开来的关键。官方 SDK 封装了这些细节，但底层的心智模型必须由你自己掌握。

## 核心概念

### 子进程启动

使用 `subprocess.Popen` 配合 `stdin=PIPE, stdout=PIPE, stderr=PIPE` 来启动进程。设置 `bufsize=1` 并使用文本模式进行逐行读取。每个服务器对应一个进程；客户端为每个服务器维护一个 `Popen` 句柄。

### 每服务器会话状态

每个服务器的 `Session` 对象包含：

- `process` —— Popen 句柄。
- `capabilities` —— 服务器在 `initialize` 时声明的内容。
- `tools` —— 上一次 `tools/list` 的结果。
- `pending` —— 从 request id 到等待响应的 promise/future 的映射。

请求本质上是异步的；在向服务器 A 发送 `tools/call` 时，即使服务器 B 正在处理调用，也不应造成阻塞。可以使用带队列的线程或 asyncio。

### 合并命名空间

当客户端看到聚合后的工具列表时，名称可能会发生冲突。两个服务器可能都暴露了 `search`。客户端有三种处理方式：

1. **按服务器名称加前缀。** `notes/search`、`files/search`。清晰但不够美观。
2. **静默先到先得。** 后加载服务器的 `search` 会覆盖先前的。有风险；会隐藏冲突。
3. **拒绝冲突。** 拒绝加载第二个服务器；通知用户。对于安全敏感的宿主环境最为稳妥。

Claude Desktop 采用按服务器加前缀的方式。Cursor 采用带明确错误提示的冲突拒绝策略。VS Code MCP 也采用了按服务器加前缀的方式。

### 路由机制

合并之后，调度表将 `tool_name -> session` 映射起来。模型通过名称发出调用；客户端找到对应的会话，并向该服务器的 stdin 写入 `tools/call` 消息，然后等待响应。

### 采样回调

如果服务器在 `initialize` 时声明了 `sampling` 能力，它可能会发送 `sampling/createMessage`，要求客户端运行其 LLM。客户端必须：

1. 在该样本解析完成之前阻止对该服务器的进一步请求，或者如果实现支持并发则进行流水线处理。
2. 调用其 LLM 提供商。
3. 将响应发回给服务器。

第 11 课将完整讲解采样流程。本课仅为保持完整性而提供桩代码（stub）。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果资源正在使用中则重新读取。通知不应产生响应——不要尝试对其进行确认（ack）。

一个常见的客户端 Bug：当通知停留在流中时，在 `tools/call` 上阻塞了读取循环。应使用后台读取线程将每条消息推送到队列中；主线程负责出队并分发。

### 重连机制

传输层可能发生故障：服务器崩溃、操作系统杀死了进程、stdio 管道断裂。客户端会在 stdout 上检测到 EOF 并将该会话视为已终止。可选方案：

- 静默重启服务器并重新握手。适用于纯只读服务器。
- 向用户暴露故障信息。适用于具有用户可见会话的状态保持型服务器。

Phase 13 · 09 将涵盖 Streamable HTTP 的重连语义；stdio 的实现更为简单。

### 保活与会话 ID

Streamable HTTP 使用 `Mcp-Session-Id` 头。Stdio 没有会话 ID——进程身份即代表会话。保活 ping 是可选的；stdio 管道在空闲状态下不会断开。

## 使用示例

`code/main.py` 会将三个模拟的 MCP 服务器作为子进程启动，与每个服务器握手，合并它们的工具列表，并将工具调用路由到正确的服务器。“服务器”实际上是运行玩具响应器（无真实 LLM）的其他 Python 进程。运行它以查看：

- 三次初始化，各自拥有独立的能力集。
- 三个 `tools/list` 结果合并为一个包含 7 个工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀避免的冲突。

重点关注：

- `Session` 数据类清晰地维护了每服务器的状态。
- 后台读取线程在不阻塞主线程的情况下从 stdout 逐行出队。
- 调度表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个服务器声明了相同的名称时，后加载的服务器会被加上前缀并重命名。

## 交付成果

本课将产出 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP 服务器列表（名称、命令、参数），该技能模块会生成一个运行框架，用于启动它们、合并工具列表，并提供带有冲突解决功能的路由函数。

## 练习

1. 运行 `code/main.py` 并观察服务器启动日志。使用 SIGTERM 终止其中一个模拟服务器进程，观察客户端如何检测 EOF 并将该会话标记为已终止。

2. 实现命名空间前缀化。当两个服务器暴露 `search` 时，将第二个重命名为 `<server>/search`。更新调度表并验证工具调用是否正确路由。

3. 为服务器重启添加类似连接池的退避策略：连续失败时采用指数退避，上限设为 30 秒，在三次失败后向用户发送通知。

4. 草拟一个支持 100 个并发 MCP 服务器的客户端。什么数据结构可以替代简单的调度字典？（提示：使用前缀树进行前缀命名空间管理，并增加每个服务器的工具数量指标。）

5. 将客户端移植到官方 MCP Python SDK。该 SDK 封装了 `stdio_client` 和 `ClientSession`。代码量应从约 200 行缩减至约 40 行，同时保留多服务器路由功能。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP client | “Agent 宿主” | 启动服务器并编排工具调用的进程 |
| Session | “每服务器状态” | 能力声明、工具列表及待处理请求的记账管理 |
| Merged namespace | “单一工具列表” | 所有活跃服务器中扁平化的工具名称集合 |
| Namespace collision | “两个服务器同名工具” | 客户端必须选择加前缀、拒绝或先到先得来处理重复项 |
| Routing | “这个调用归谁？” | 根据工具名称分发到所属服务器 |
| Background reader | “非阻塞 stdout” | 将服务器 stdout 排空至队列的线程或任务 |
| Sampling callback | “LLM 即服务” | 处理服务器 `sampling/createMessage` 的客户端处理器 |
| `notifications/*_changed` | “基础状态变更” | 指示客户端必须重新发现或重新读取的信号 |
| Reconnection policy | “服务器宕机时” | 传输层故障时的重启语义 |
| Stdio session | “进程即会话” | 无会话 ID；子进程的生存期即为会话周期 |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) —— 标准的客户端行为规范
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) —— 使用 Python SDK 的 Hello World 客户端教程
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) —— 参考 `ClientSession` 与 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) —— TypeScript 并行实现
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) —— VS Code 如何在单个编辑器宿主中多路复用多个 MCP 服务器
