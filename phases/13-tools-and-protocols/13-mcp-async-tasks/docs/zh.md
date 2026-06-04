# 异步任务（SEP-1686）—— 针对长耗时工作的“立即调用，稍后获取”

> 真实的 Agent 工作通常需要几分钟到几小时：CI 运行、深度研究综合、批量导出等。同步工具调用会导致连接断开、超时或阻塞 UI。SEP-1686 于 2025-11-25 合并，引入了 Tasks 基础概念：任何请求都可以被增强为任务，结果可以稍后获取或通过状态通知流式传输。规范漂移风险提示：Tasks 在 2026 年上半年仍处于实验阶段；SDK 接口仍在围绕该规范进行设计。

**类型：** 构建
**语言：** Python（标准库、异步任务状态机）
**前置条件：** 第 13 阶段 · 07（MCP 服务器）、第 13 阶段 · 09（传输层）
**预计时间：** 约 75 分钟

## 学习目标

- 识别何时将工具从同步升级为任务增强型（服务端工作超过 30 秒）。
- 熟悉任务生命周期：`working` → `input_required` → `completed` / `failed` / `cancelled`。
- 持久化任务状态，确保崩溃时不会丢失进行中的工作。
- 正确轮询 `tasks/status` 并获取 `tasks/result`。

## 问题所在

一个 `generate_report` 工具会运行多分钟的提取流水线。在同步模型下的选项有：

1. 保持连接打开三分钟。远程传输层会丢弃它；客户端超时；UI 冻结。
2. 立即返回占位符；要求客户端轮询自定义端点。破坏了 MCP 的统一性。
3. 发后即忘；无结果返回。

这些方案都不理想。SEP-1686 提供了第四种方案：任务增强。任何请求（通常是 `tools/call`）都可以标记为任务。服务器会立即返回任务 ID。客户端轮询 `tasks/status`，完成后获取 `tasks/result`。服务端状态可经受重启考验。

## 核心概念

### 任务增强

通过设置 `params._meta.task.required: true`（或 `optional: true`，由服务器决定），请求即变为任务。服务器会立即响应以下内容：

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl` 是服务器保留状态的承诺；超过 ttl 后，任务结果将被丢弃。

### 按工具可选启用

工具注解可声明任务支持情况：

- `taskSupport: "forbidden"` —— 此工具始终同步运行。适用于快速工具。
- `taskSupport: "optional"` —— 客户端可请求任务增强。
- `taskSupport: "required"` —— 客户端必须使用任务增强。

一个 `generate_report` 工具应设为 `required`。一个 `notes_search` 工具应设为 `forbidden`。

### 状态

```
working  -> input_required -> working  (loop via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

状态机为仅追加模式：一旦进入 `completed`、`failed` 或 `cancelled`，任务即达到终态。

### 方法

- `tasks/status {taskId}` —— 返回当前状态和进度提示。
- `tasks/result {taskId}` —— 若未完成则阻塞或返回 404。
- `tasks/cancel {taskId}` —— 幂等操作；终态会被忽略。
- `tasks/list` —— 可选；枚举活跃及最近完成的任务。

### 流式状态变更

当服务器支持时，客户端可订阅状态通知：

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

采用流式传输而非轮询的客户端能获得更好的用户体验。轮询作为最小功能集始终受支持。

### 持久化状态

规范要求声明了任务支持的服务器必须持久化状态。崩溃不应导致 ttl 内的已完成结果丢失。存储方案涵盖 SQLite、Redis 到文件系统。本课的实验环境使用文件系统。

### 取消语义

`tasks/cancel` 是幂等的。如果任务正在执行中，服务器会尝试停止（检查执行器协作式取消）。如果已是终态，则该请求为空操作。

### 崩溃恢复

当服务器进程重启时：

1. 加载所有持久化的任务状态。
2. 将已死进程对应的 `working` 任务标记为 `failed`，错误信息为 `CRASH_RECOVERY`。
3. 在 ttl 内保留 `completed` / `failed` / `cancelled`。

### 异步任务与采样结合

任务本身可以调用 `sampling/createMessage`。这正是长耗时研究任务的运作方式：服务器的任务线程按需对客户端模型进行采样，同时客户端 UI 将任务显示为 `working` 并定期更新进度。

### 为何处于实验阶段

SEP-1686 已于 2025-11-25 发布，但更广泛的路线图指出了三个待解决问题：持久化订阅原语、子任务（父子任务关系）以及结果 TTL 标准化。预计该规范将在 2026 年持续演进。生产代码应将 Tasks 视为仅在常见场景下稳定，并为未来子任务相关的 SDK 变更做好防护。

## 实践应用

`code/main.py` 实现了一个持久的任务存储（基于文件系统）和一个在后台线程运行的 `generate_report` 工具。客户端调用该工具后立即获得任务 ID，在工作者更新进度期间轮询 `tasks/status`，完成后获取 `tasks/result`。取消功能可用；通过终止工作者线程并重新加载状态来模拟崩溃恢复。

重点关注：

- 持久化到 `/tmp/lesson-13-tasks/<id>.json` 的任务状态 JSON。
- 工作者线程更新 `progress` 字段；轮询可见其逐步推进。
- 客户端发起取消时会设置事件标志；工作者检查后提前退出。
- “崩溃”时重新加载状态会将进行中的任务标记为 `failed`，错误为 `CRASH_RECOVERY`。

## 交付成果

本课将产出 `outputs/skill-task-store-designer.md`。面对长耗时工具（研究、构建、导出），你需要设计任务存储（状态结构、ttl、持久性），选择合适的 taskSupport 标志，并规划进度通知机制。

## 练习

1. 运行 `code/main.py`。启动一个 `generate_report` 任务，轮询状态，然后获取结果。

2. 在运行中途添加一次 `tasks/cancel` 调用。验证工作者是否响应，且状态变为 `cancelled`。

3. 模拟崩溃恢复：终止工作者线程，重启加载器，观察 `CRASH_RECOVERY` 故障模式。

4. 将存储扩展至 SQLite。持久性优势相同；查询选项更加丰富（例如列出会话 X 的所有任务）。

5. 阅读 MCP 2026 年路线图文章。找出最有可能在未来一年影响 SDK API 设计的 Tasks 相关未决问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Task | “长耗时工具调用” | 带有 `_meta.task` 以实现异步执行的请求 |
| SEP-1686 | “Tasks 规范” | 于 2025-11-25 引入 Tasks 的规范演进提案 |
| `_meta.task` | “任务信封” | 包含 ID、状态、ttl 的逐请求元数据 |
| taskSupport | “工具标志” | 每个工具的 `forbidden` / `optional` / `required` |
| `tasks/status` | “轮询方法” | 获取当前状态及可选进度提示 |
| `tasks/result` | “获取结果” | 返回已完成的有效载荷，若未完成则返回 404 |
| `tasks/cancel` | “停止它” | 幂等取消请求 |
| ttl | “保留预算” | 服务器承诺保留任务状态的毫秒数 |
| `notifications/tasks/updated` | “状态推送” | 服务器发起的状态变更事件 |
| Durable store | “防崩溃状态” | 文件系统 / SQLite / Redis 持久化层 |

## 延伸阅读

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) —— 原始提案与完整讨论
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) —— 带原理的设计详解
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) —— 机制与状态机
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) —— SDK 级任务实现模式
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— 未决问题与 2026 年优先事项（含子任务）
