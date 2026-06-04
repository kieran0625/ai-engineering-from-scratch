# 构建 MCP 服务器 —— Python + TypeScript SDK

> 大多数 MCP 教程仅展示基于 stdio 的 Hello World 示例。一个生产级的服务器会同时提供 Tools、Resources 和 Prompts，处理能力协商，抛出结构化错误，并在不同 SDK 间保持一致的行为。本课程将端到端地构建一个笔记服务器：涵盖标准库实现的 stdio 传输、JSON-RPC 分发、三大服务器原语，以及一种纯函数风格——当你准备进阶时，可无缝迁移至 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 实战构建
**语言：** Python（标准库实现，stdio MCP 服务器）
**前置要求：** 第 13 阶段 · 06（MCP 基础）
**预计时间：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个分发循环，从 stdin 读取 JSON-RPC 消息并将响应写入 stdout。
- 根据 JSON-RPC 2.0 规范及 MCP 的附加代码，输出结构化的错误响应。
- 在不重写工具逻辑的前提下，将标准库实现进阶迁移至 FastMCP（Python SDK）或 TypeScript SDK。

## 问题背景

在使用远程传输（第 13 阶段 · 09）或认证层（第 13 阶段 · 16）之前，你需要一个干净的本地服务器。本地即指 stdio：服务器由客户端作为子进程启动，消息通过换行符分隔的 stdin/stdout 进行流式传输。

2025-11-25 版规范规定，stdio 消息需编码为带有显式 `\n` 分隔符的 JSON 对象。此处不使用 SSE；SSE 是旧的远程模式，将于 2026 年中旬被移除（Atlassian 的 Rovo MCP 服务器已于 2026 年 6 月 30 日弃用它；Keboola 于 2026 年 4 月 1 日弃用）。对于 stdio，每行一个 JSON 对象即为完整的网络传输格式。

笔记服务器是一个很好的载体，因为它能覆盖所有三种服务器原语。Tools 负责变更操作（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 交付模板（`review_note`）。本课程的架构设计可泛化至任何领域。

## 核心概念

### 消息分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何非 JSON-RPC 信封的内容。调试日志应输出到 stderr。
- 每个请求都必须匹配一个携带相同 `id` 的响应。
- 通知类请求不得回复。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

仅声明你支持的功能。客户端依赖能力集来限制可用功能。

### 实现 `tools/list` 与 `tools/call`

`tools/list` 返回 `{tools: [...]}`，其中每项包含 `name`、`description` 和 `inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块具有明确类型。最常见的是：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误分为两种形态。协议级错误（未知方法、参数错误）属于 JSON-RPC 错误。工具级错误（调用合法但执行失败）以 `{content: [...], isError: true}` 形式返回。这使得模型能在其上下文中感知到该失败。

### 实现 Resources

Resources 设计上为只读。`resources/list` 返回清单；`resources/read` 返回具体内容。URI 可以是 `file://...`、`http://...`，或类似 `notes://` 的自定义方案。

当以 Resource 而非 Tool 暴露数据时：

- 模型不会“调用”它；客户端可根据用户请求将其注入上下文。
- 订阅机制允许服务器在资源变更时推送更新（第 13 阶段 · 10）。
- 第 13 阶段 · 14 通过 `ui://` 对此进行了扩展，用于交互式资源。

### 实现 Prompts

Prompts 是带有命名参数的模板。宿主应用会将其作为斜杠命令呈现。一个 `review_note` 提示词可能接受 `note_id` 参数，并生成多轮对话模板，供客户端传递给其模型。

### stdio 传输的注意事项

- 换行符分隔的 JSON。无长度前缀帧结构。
- 不要缓冲。每次写入后执行 `sys.stdout.flush()`。
- 生命周期由客户端控制。当 stdin 关闭（EOF）时，应优雅退出。
- 不要静默处理 SIGPIPE；记录日志后退出。

### 注解（Annotations）

每个工具均可携带 `annotations` 来描述安全属性：

- `readOnlyHint: true` —— 纯读取操作，可安全重试。
- `destructiveHint: true` —— 不可逆的副作用；客户端应进行确认。
- `idempotentHint: true` —— 相同输入产生相同输出。
- `openWorldHint: true` —— 与外部系统交互。

客户端利用这些注解来决定用户体验（确认对话框、状态指示器）和路由策略（第 13 阶段 · 17）。

### 进阶迁移路径

`code/main.py` 中的标准库服务器代码约 180 行。FastMCP（Python）将相同的逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 具有等效的架构。当你准备好时，可直接无缝替换；核心概念（能力声明、消息分发、内容块）完全一致。

## 使用方式

`code/main.py` 是一个完整的基于 stdio 的标准库笔记 MCP 服务器。它处理三个工具（`notes_list`、`notes_search`、`notes_create`）的 `initialize`、`tools/list`、`tools/call`，每个笔记的 `resources/list` 和 `resources/read`，以及一个 `review_note` 提示词。你可以通过管道传递 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

重点关注：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器均返回内容块列表，而非裸字符串。
- 当执行器抛出异常时，会设置 `isError: true`。

## 交付成果

本课程将产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个业务领域（如笔记、工单、文件、数据库），该技能脚手架将为你搭建一个具备合理 Tools/Resources/Prompts 划分及 SDK 进阶路径的 MCP 服务器。

## 练习

1. 运行 `code/main.py`，并使用手动构造的 JSON-RPC 消息驱动它。测试 `notes_create`，随后通过 `resources/read` 检索新创建的笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证客户端是否会弹出确认对话框（这需要真实的宿主环境；Claude Desktop 可用）。

3. 实现 `resources/subscribe`，使服务器在笔记修改时推送 `notifications/resources/updated`。添加一个保活任务。

4. 将服务器移植到 FastMCP。Python 文件代码量应缩减至 80 行以内。网络传输行为必须完全一致；请使用相同的 JSON-RPC 测试套件进行验证。

5. 阅读规范中的 `server/tools` 章节，找出本课程服务器中未实现的一个工具定义字段。（提示：有好几个；任选一个并补充实现。）

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MCP server | “暴露工具的东西” | 通过 stdio 或 HTTP 通信 MCP JSON-RPC 的进程 |
| stdio transport | “子进程模型” | 由客户端启动服务器；通过 stdin/stdout 通信 |
| Dispatcher | “方法路由器” | 将 JSON-RPC 方法名映射到处理函数的字典/映射表 |
| Content block | “工具结果片段” | 工具响应 `content` 数组中的类型化元素 |
| `isError` | “工具级失败” | 标识工具执行失败；区别于 JSON-RPC 协议错误 |
| Annotations | “安全提示” | readOnly / destructive / idempotent / openWorld 标志位 |
| FastMCP | “Python SDK” | 基于装饰器的高层框架，封装于 MCP 协议之上 |
| Resource URI | “可寻址数据” | `file://`、`db://` 或标识资源的自定义方案 |
| Prompt template | “斜杠命令简介” | 服务器提供的模板，含供宿主 UI 填充的参数槽位 |
| Capability declaration | “功能开关” | 在 `initialize` 中按原语声明的标志位 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) —— 官方参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) —— 并行 TypeScript 实现
- [FastMCP — server framework](https://gofastmcp.com/) —— 面向 MCP 服务器的装饰器风格 Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) —— 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) —— tools/* 消息的完整参考文档
