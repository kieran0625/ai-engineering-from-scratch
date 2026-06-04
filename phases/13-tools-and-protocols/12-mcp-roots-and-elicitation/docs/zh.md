# Roots 与 Elicitation —— 作用域限定与运行中用户输入

> 硬编码路径在用户打开不同项目时会立即失效。预填充的工具参数在用户描述不充分时也会出错。Roots 将服务器的作用域限制在用户控制的一组 URI 内；Elicitation 会在工具调用中途暂停，通过表单或 URL 向用户请求结构化输入。这是两个客户端原语，用于修复两种常见的 MCP 故障模式。SEP-1036（URL-mode Elicitation，2025-11-25）在 2026 年上半年仍处于实验阶段——依赖它之前请检查 SDK 版本。

**类型：** 构建
**语言：** Python（标准库，roots + elicitation 演示）
**前置条件：** Phase 13 · 07（MCP 服务器）
**耗时：** 约 45 分钟

## 学习目标

- 声明 `roots` 并响应 `notifications/roots/list_changed`。
- 将服务器的文件操作限制在已声明的 root 集合内的 URI。
- 使用 `elicitation/create` 在工具调用中途向用户请求确认或结构化输入。
- 在 form-mode 和 URL-mode Elicitation 之间做出选择（后者为实验性功能；需注意漂移风险）。

## 问题所在

笔记类 MCP 服务器在生产环境中会遇到的两种具体故障。

**路径假设失效。** 服务器代码基于 `~/notes` 编写。另一台机器上的用户将笔记存放在 `~/Documents/Notes`，此时发起的工具调用会静默失败（找不到文件），或者更糟，写入了错误的位置。

**缺失用户本应知晓的参数。** 用户要求“删除旧的 TPS 报告笔记”。模型调用了 `notes_delete(title: "TPS report")`，但存在 2023、2024 和 2025 年的三个匹配笔记。工具无法猜测。返回“歧义”错误令人恼火；若强行对三者全部执行则会导致灾难性后果。

Roots 解决第一个问题：客户端在 `initialize` 处声明服务器可访问的 URI 集合。Elicitation 解决第二个问题：服务器暂停工具调用，并发送 `elicitation/create` 让用户从中选择。

## 核心概念

### Roots

客户端在 `initialize` 处声明一个 root 列表：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

随后服务器可调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

服务器必须将 Roots 视为边界：任何超出 root 集合的文件读取或写入操作均会被拒绝。这并非由客户端强制实施（服务器仍是用户信任的代码），但符合规范的服务器会遵守此约定。

当用户添加或删除 root 时，客户端会发送 `notifications/roots/list_changed`。服务器重新调用 `roots/list` 并更新其边界。

### 为什么 Roots 是客户端原语

Roots 由客户端声明，因为它们代表用户的授权模型。用户告诉 Claude Desktop “允许这个笔记服务器访问这两个目录”。服务器无权扩大该作用域。

### Elicitation：默认的 Form Mode

`elicitation/create` 接收一个表单 Schema 以及自然语言提示词：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

客户端渲染表单，收集用户答案后返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三种可能的动作：`accept`（用户填写完毕）、`decline`（用户关闭了表单）、`cancel`（用户中止了整个工具调用）。

表单 Schema 是扁平的——v1 不支持嵌套对象。SDK 通常会拒绝比单层结构更复杂的定义。

### Elicitation：URL Mode（SEP-1036，实验性）

2025-11-25 新增功能。服务器不再发送 Schema，而是发送一个 URL：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

客户端在浏览器中打开该 URL，等待完成，并在用户返回后返回结果。适用于 OAuth 流程、支付授权和文档签署等表单无法满足的场景。

漂移风险说明：SEP-1036 的响应结构仍在调整中；部分 SDK 返回回调 URL，另一些返回完成令牌。在生产环境使用 URL Mode 前，请务必阅读您所用 SDK 的发行说明。

### 何时适合使用 Elicitation

- 破坏性操作前的用户确认（destructive hint + elicitation）。
- 消除歧义（从 N 个匹配项中选择其一）。
- 首次运行配置（API 密钥、目录、偏好设置）。
- OAuth 风格流程（URL mode）。

### 何时不适合使用 Elicitation

- 填补模型本可以用自然语言询问的工具必填参数。应使用普通的重新提示（re-prompt），而非 elicitation 对话框。
- 高频调用。Elicitation 会中断对话流；切勿在循环内部触发它。
- 服务器事后能够验证的任何内容。应先验证，返回错误，让模型用文本向用户提问。

### 人机协同（Human-in-the-loop）桥梁

Elicitation 与 sampling 结合共同实现了 MCP 的“人机协同（human-in-the-loop）”模型。服务器的 Agent 循环可以因用户输入（elicitation）或模型推理（sampling）而暂停。Phase 13 · 11 讲解了 sampling；本课讲解 elicitation。将它们结合使用即可实现完整的循环内控制。

## 实践应用

`code/main.py` 为笔记服务器扩展了以下功能：

- `roots/list` 响应：服务器在收到 root-list-changed 通知后会重新查询。
- 一个 `notes_delete` 工具：当多个笔记匹配时，使用 `elicitation/create` 进行消歧。
- 一个 `notes_setup` 工具：使用 URL-mode elicitation 打开首次运行配置页面（模拟）。
- 边界检查：拒绝在已声明 roots 之外的 URI 上执行操作。

演示包含三种场景：正常路径（单个匹配）、消歧（三个匹配，触发 elicitation）、越权写入（out-of-root-write，被拒绝）。

## 交付成果

本课将产出 `outputs/skill-elicitation-form-designer.md`。针对可能需要用户确认或消歧的工具，掌握设计 elicitation 表单 Schema 和消息模板的技能。

## 练习

1. 运行 `code/main.py`。触发消歧路径；确认模拟的用户答案能正确路由回工具。

2. 添加一个新工具 `notes_archive`，每次都需要 elicitation 确认（附带 destructive hint）。检查用户体验：这与模型用文本重新询问相比有何优劣？

3. 为首次运行的 OAuth 流程实现 URL-mode elicitation。注意漂移风险，并添加 SDK 版本守卫。

4. 扩展 `roots/list` 的处理逻辑：当收到通知时，服务器应原子性地重新读取并扫描可能已超出作用域的打开文件句柄。

5. 阅读 GitHub 上关于 SEP-1036 的 Issue 讨论帖。找出一个影响服务器如何处理 URL-mode 回调的未决问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Root | “授权边界” | 客户端允许服务器访问的 URI |
| `roots/list` | “服务器请求作用域” | 客户端返回当前的 root 集合 |
| `notifications/roots/list_changed` | “用户更改了作用域” | 客户端通知 root 集合已发生变更 |
| Elicitation | “调用中途询问用户” | 由服务器发起的结构化用户输入请求 |
| `elicitation/create` | “该方法” | 用于 elicitation 请求的 JSON-RPC 方法 |
| Form mode | “基于 Schema 的表单” | 在客户端 UI 中渲染为表单的扁平 JSON Schema |
| URL mode | “浏览器重定向” | SEP-1036 实验性功能；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | “用户响应结果” | 服务器需处理的三个分支 |
| Disambiguation | “二选一/多选一” | 工具面对 N 个候选项时的常见 elicitation 用例 |
| Flat form | “仅顶层属性” | Elicitation Schema 不支持嵌套 |

## 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) —— 权威的 roots 参考文档
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) —— 权威的 elicitation 参考文档
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) —— 2025-11-25 新增功能详解
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) —— URL-mode elicitation 提案（实验性，存在漂移风险）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) —— 用户体验详解
