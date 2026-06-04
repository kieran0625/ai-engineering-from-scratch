# MCP 传输协议 — stdio 与 Streamable HTTP 及 SSE 迁移

> stdio 仅适用于本地环境，无法用于远程。Streamable HTTP（2025-03-26 版本）是远程传输的标准。旧的 HTTP+SSE 传输协议已弃用，并将于 2026 年中旬移除。选错传输协议意味着后续需要迁移成本；选对则能构建出支持远程托管、具备会话连续性及防 DNS 重绑定保护的 MCP 服务器。

**类型：** 学习
**语言：** Python（标准库、Streamable HTTP 端点骨架）
**前置知识：** 阶段 13 · 07、08（MCP 服务器与客户端）
**耗时：** 约 45 分钟

## 学习目标

- 根据部署形态（本地与远程、单进程与集群）在 stdio 和 Streamable HTTP 之间做出选择。
- 实现 Streamable HTTP 单端点模式：使用 POST 发送请求，使用 GET 建立会话流。
- 强制执行 `Origin` 验证与会话 ID 语义，以防御 DNS 重绑定攻击。
- 在 2026 年中旬的移除期限前，将遗留的 HTTP+SSE 服务器迁移至 Streamable HTTP。

## 问题背景

MCP 的第一个远程传输协议（2024-11）是 HTTP+SSE：包含两个端点，一个用于客户端的 POST 请求，另一个用于服务端到客户端流的 Server-Sent-Events 通道。它确实能工作，但也很笨拙：每个会话需要两个端点，某些 CDN 前置缓存会失效，且严重依赖长连接 SSE，而部分 WAF 会激进地终止这些连接。

2025-03-26 规范将其替换为 Streamable HTTP：一个端点，POST 用于客户端请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头。此后构建或迁移的所有服务器均采用 Streamable HTTP。旧的 SSE 模式正在被弃用——Atlassian Rovo 于 2026 年 6 月 30 日移除它；Keboola 于 2026 年 4 月 1 日移除；大多数剩余的企业级服务器将在 2026 年底前完成移除。

对于本地服务器，stdio 依然重要。Claude Desktop、VS Code 以及所有类 IDE 客户端均通过 stdio 启动服务器。正确的思维模型是：stdio 用于“本机”，Streamable HTTP 用于“网络”。两者互不交叉。

## 核心概念

### stdio

- 子进程传输。客户端启动服务器，通过 stdin/stdout 进行通信。
- 每行一个 JSON 对象。换行符分隔。
- 无会话 ID；进程身份即代表会话。
- 无需认证（子进程继承父进程的信任边界）。
- 绝不要用于远程服务器——若需通过 SSH 或 socat 隧道转发，此时应直接使用 Streamable HTTP。

### Streamable HTTP

单一端点 `/mcp`（或任意路径）。支持三种 HTTP 方法：

- **POST /mcp**。客户端发送 JSON-RPC 消息。服务器回复单个 JSON 响应，或包含一个或多个响应的 SSE 流（适用于批量响应及与该请求相关的通知）。
- **GET /mcp**。客户端打开长连接 SSE 通道。服务器利用该通道向客户端发起请求（采样、通知、信息收集）。
- **DELETE /mcp**。客户端显式终止会话。

会话由服务器在首次响应中设置的 `Mcp-Session-Id` 头标识，客户端需在后续每次请求中回显该头。会话 ID 必须是密码学安全的随机数（128 位以上）；出于安全考虑，拒绝客户端自定义的 ID。

### 单端点 vs 双端点

旧规范中的双端点模式在 2026 年仍可调用——规范将其声明为“向后兼容”。但所有新服务器都应采用单端点模式。官方 SDK 均生成单端点代码；仅在访问未迁移的远程服务器时才使用旧模式。

### `Origin` 验证与 DNS 重绑定防护

浏览器目前并非 MCP 客户端，但攻击者可构造网页诱使浏览器向 `localhost:1234/mcp` 发起 POST 请求——这正是用户本地 MCP 服务器的监听地址。如果服务器不检查 `Origin`，浏览器的同源策略将无法提供保护，因为 `Origin: http://evil.com` 属于合法的跨域请求。

2025-11-25 规范要求服务器拒绝 `Origin` 不在白名单中的请求。白名单通常包含 MCP 客户端主机（`https://claude.ai`、`vscode-webview://*`）以及用于本地 UI 的 localhost 变体。

### 会话 ID 生命周期

1. 客户端发送不带 `Mcp-Session-Id` 的首次请求。
2. 服务器分配随机 ID，并在响应头中设置 `Mcp-Session-Id`。
3. 客户端在所有后续请求及用于流的 `GET /mcp` 中回显该头。
4. 服务器可撤销会话；客户端在后续请求中将收到 404，必须重新初始化。
5. 客户端可显式 DELETE 会话以实现优雅关闭。

### 保活与重连

SSE 连接可能会断开。客户端通过携带相同 `Mcp-Session-Id` 重新发起 GET 请求来重建连接。服务器必须缓存断线期间丢失的事件（在一定合理时间窗口内），并通过客户端回显的 `last-event-id` 头进行重放。

阶段 13 · 13 介绍了 Tasks，它能让长时间运行的任务即使在完整会话重连后也能继续存活。

### 向后兼容探测

希望同时支持新旧服务器的客户端探测流程：

1. 向 `/mcp` 发起 POST 请求。
2. 若响应为 `200 OK` 且包含 JSON 或 SSE，则为 Streamable HTTP。
3. 若响应为 `200 OK`，包含 `Content-Type: text/event-stream` 以及指向次要端点的 `Location` 头，则为遗留的 HTTP+SSE；请遵循 `Location` 指引。

### Cloudflare、ngrok 与托管服务

2026 年的生产级远程 MCP 服务器通常运行在 Cloudflare Workers（配合其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 环境中。关键点在于：你的托管环境必须支持 SSE GET 所需的长连接 HTTP。Vercel 免费层限制为 10 秒，不适用。Cloudflare Workers 支持无限时长的流。

### 网关聚合

当你使用网关（阶段 13 · 17）代理多个 MCP 服务器时，网关表现为一个单一的 Streamable HTTP 端点，负责重写会话 ID 并向上游复用流量。工具在网关层合并；客户端看到的是单一逻辑服务器。

### 传输故障模式

- **stdio SIGPIPE**。子进程在写入中途死亡会触发 SIGPIPE；服务器应干净退出。客户端应检测 EOF 并将会话标记为死亡。
- **HTTP 502 / 504**。Cloudflare、nginx 及其他代理在遇到上游故障时会返回这些状态码。Streamable HTTP 客户端应在短暂退避后重试一次。
- **SSE 连接断开**。TCP RST、代理超时或客户端网络变更会导致流关闭。客户端携带 `Mcp-Session-Id` 及可选的 `last-event-id` 进行重连以恢复。
- **会话撤销**。服务器使会话 ID 失效；客户端下次请求将收到 404。客户端必须重新握手。
- **时钟偏差**。客户端与服务端的 Resource-TTL 计算可能出现分歧。客户端应将服务端的时间戳视为权威来源。

### 何时绕过 Streamable HTTP

部分企业在内部网络中通过 gRPC 或消息队列传输协议部署 MCP 服务器。这属于非标准做法——MCP 规范并未正式定义这些协议。网关可以向 MCP 客户端暴露 Streamable HTTP 接口，同时在内部使用 gRPC。保持外部接口符合规范即可，翻译工作由网关负责。

## 实践应用

`code/main.py` 使用 `http.server`（标准库）实现了最小的 Streamable HTTP 端点。它在 `/mcp` 上处理 POST、GET 和 DELETE，在首次响应中设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非白名单来源的请求。处理器复用了第 07 课笔记服务器的分发逻辑。

重点关注：

- POST 处理器读取 JSON-RPC 请求体，进行分发，并写入 JSON 响应（单响应变体；SSE 变体在结构上类似）。
- `Origin` 检查会拒绝默认的 `http://evil.example` 探测，但接受 `http://localhost`。
- 会话 ID 为随机生成的 128 位十六进制字符串；服务器在内存中维护每个会话的状态。

## 交付成果

本课产出 `outputs/skill-mcp-transport-migrator.md`。针对基于 HTTP+SSE（遗留版）的 MCP 服务器，该技能将生成一份迁移至 Streamable HTTP 的计划，包含会话 ID 连续性、Origin 检查及向后兼容探测支持。

## 练习

1. 运行 `code/main.py`。从 `curl` 向 `initialize` 发起 POST 请求，并观察 `Mcp-Session-Id` 响应头。再次发起 POST 请求并回显该头，验证会话连续性。

2. 添加一个开启 SSE 流的 GET 处理器。每五秒发送一个 `notifications/progress` 事件。通过携带相同会话 ID 重新发起 GET 请求进行重连，并确认服务器接受该请求。

3. 实现 `last-event-id` 的重放逻辑。重连时，重放自该 ID 以来生成的所有事件。

4. 扩展 `Origin` 验证逻辑以支持通配符模式（`https://*.example.com`），并确认其接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册中心选取一个遗留的 HTTP+SSE 服务器（有多个可用），草拟迁移方案：说明端点处理、会话 ID 生成及头部语义方面的变更。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| stdio 传输 | “本地子进程” | 基于 stdin/stdout 的 JSON-RPC，换行符分隔 |
| Streamable HTTP | “远程传输协议” | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | “遗留模式” | 双端点模型，将于 2026 年中旬移除 |
| `Mcp-Session-Id` | “会话头” | 服务器分配的随机 ID，在每次后续请求中回显 |
| `Origin` 白名单 | “防 DNS 重绑定” | 拒绝 Origin 未经批准的请求 |
| 单端点 | “单一 URL” | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | “SSE 重放” | 用于恢复断开流且不丢失事件的头部 |
| 向后兼容探测 | “新旧版本检测” | 客户端通过响应形状检查自动选择传输协议 |
| 长连接 HTTP | “SSE 流式传输” | 服务器在单个 TCP 连接上持续推送数分钟或数小时的事件 |
| 会话撤销 | “强制重新初始化” | 服务器使会话 ID 失效；客户端必须重新握手 |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 与 Streamable HTTP 的权威参考
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的版本修订
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — 托管于 Workers 的 Streamable HTTP 模式
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 不同部署形态下的对比
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体的迁移期限示例
