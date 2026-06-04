# MCP 网关与注册表 —— 企业控制平面

> 企业不能让每位开发者随意安装随机的 MCP 服务器。网关集中管理认证、RBAC、审计、限流、缓存和工具投毒检测，然后将合并后的工具表面暴露为单个 MCP 端点。官方 MCP 注册表（由 Anthropic、GitHub、PulseMCP 和 Microsoft 联合维护，经过命名空间验证）是权威的上游源。本课将说明网关的定位，演示一个最小化实现，并梳理 2026 年的厂商生态。

**类型：** 学习
**语言：** Python（标准库，最小化网关）
**前置知识：** 阶段 13 · 15（工具投毒），阶段 13 · 16（OAuth 2.1）
**预计时间：** 约 45 分钟

## 学习目标

- 说明 MCP 网关的位置（位于 MCP 客户端与多个后端 MCP 服务器之间）。
- 实现网关的五大职责：认证、RBAC、审计、限流、策略。
- 在网关层实施固定工具哈希清单。
- 区分官方 MCP 注册表与元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题背景

一家财富 500 强企业拥有 30 个已批准的 MCP 服务器、5000 名开发者、合规与审计要求，以及希望实施集中式策略的安全团队。允许每位开发者在各自的 IDE 中随意安装任意服务器是不可行的。

网关模式如下：

1. 网关作为单个 Streamable HTTP 端点运行，开发者通过该端点进行连接。
2. 网关保管每个后端 MCP 服务器的凭据。
3. 所有开发者请求均通过网关自身的 OAuth 进行认证和范围限定。
4. 网关将调用路由至后端服务器，并应用安全策略。
5. 所有调用均记录日志以供审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway —— 均在 2025-2026 年间推出了网关或网关功能模块。

与此同时，官方 MCP 注册表作为权威上游源推出：提供经过筛选、命名空间验证、采用反向 DNS 命名的服务器，供网关拉取。元注册表（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则聚合来自多个来源的服务器。

## 核心概念

### 网关的五大职责

1. **认证（Auth）**：使用 OAuth 2.1 识别开发者身份；映射到用户角色。
2. **RBAC**：基于用户的策略：决定可访问哪些服务器、哪些工具、哪些作用域。
3. **审计（Audit）**：记录每次调用的操作者、内容、时间及结果。
4. **限流（Rate limit）**：按用户/工具/服务器设置上限，防止滥用。
5. **策略（Policy）**：拒绝被投毒的描述，强制执行“双重规则”，脱敏个人身份信息（PII）。

### 网关作为单一端点

对开发者而言，网关看起来就像单个 MCP 服务器。内部它会将请求路由到 N 个后端。会话 ID（阶段 13 · 09）在边界处会被重写。

### 凭据保管

开发者永远看不到后端令牌。网关负责保管它们（或代理给负责保管的身份提供商）。拥有 `notes:read` 权限的开发者可以通过网关的后端凭据间接访问 notes MCP 服务器——但这仅在绑定该间接访问权限的策略下才被允许。

### 网关层的工具哈希固定

网关维护一份已批准工具描述的清单（SHA256 哈希值）。在发现阶段，它会获取每个后端的 `tools/list`，将哈希值与清单进行比对，并移除任何描述发生变动的工具。这是将阶段 13 · 15 中的防“跑路”机制集中应用于网关层。

### 代码化策略

高级网关使用 OPA/Rego、Kyverno 或 Styra 来表达策略。诸如“用户 `alice` 仅可在组织 `acme` 下的仓库中调用 `github.open_pr`”的规则以声明式方式编码。简单网关则使用手写 Python 代码。两种形式均可行。

### 会话感知路由

当用户的会话包含多种服务器时，网关会进行多路复用：开发者的单个 MCP 会话会持有 N 个后端会话，每个服务器对应一个。来自任何后端的通知都会通过网关路由至开发者的会话。

### 命名空间合并

网关会合并所有后端的工具命名空间，通常在冲突时添加前缀。`github.open_pr`、`notes.search`。这使得路由变得明确无误。

### 注册表

- **官方 MCP 注册表（`registry.modelcontextprotocol.io`）**：在 Anthropic、GitHub、PulseMCP 和 Microsoft 的共同维护下推出。经过命名空间验证（反向 DNS：`io.github.user/server`）。已预先过滤以保证基础质量。
- **Glama**：以搜索为核心的元注册表，聚合多个来源。
- **MCPMarket**：偏向商业的目录，包含供应商列表。
- **MCP.so**：社区目录；开放提交。
- **Smithery**：采用包管理器风格的安装流程。
- **LobeHub**：集成在其 LobeChat 应用 UI 中的注册表。

企业级网关默认从官方注册表拉取数据，允许管理员从元注册表中添加精选内容，并拒绝任何未固定的项目。

### 反向 DNS 命名

官方注册表强制要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间可防止域名抢注，并使信任委托更加清晰。

### 厂商调研（2026年4月）

| 厂商 | 优势 |
|--------|----------|
| Cloudflare MCP Portals | 边缘托管；集成 OAuth；提供免费层级 |
| Kong AI Gateway | 原生支持 K8s；细粒度策略；日志输出至 OpenTelemetry |
| IBM ContextForge | 企业级 IAM；合规支持；审计导出 |
| TrueFoundry | 偏向 DevOps；指标优先 |
| MintMCP | 面向开发者平台 |
| Envoy AI Gateway | 开源；可自定义过滤器 |

阶段 17（生产基础设施）将更深入地探讨网关运维。

## 实践指南

`code/main.py` 提供了一个约 150 行的最小化网关实现：通过伪造的 Bearer Token 对用户进行认证，维护基于用户的 RBAC 策略，将请求路由至两个后端 MCP 服务器，将每次调用写入审计日志，执行限流，并拒绝任何描述哈希与固定清单不匹配的后端工具。

重点关注：

- `RBAC` 字典以 `user_id` 为键，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是一个仅追加的事件列表。
- 限流机制采用基于用户的令牌桶算法。
- 固定清单是一个包含 `server::tool -> hash` 的字典。

## 交付成果

本课将生成 `outputs/skill-gateway-bootstrap.md`。给定企业 MCP 计划（用户、后端、合规要求），该技能将生成一份网关配置规范。

## 练习

1. 运行 `code/main.py`。分别以允许的用户、禁止的用户身份发起调用，然后模拟超出限流的突发流量。验证这三种流程。

2. 添加一项策略，在返回结果给客户端之前对 PII 进行脱敏。使用简单的正则表达式匹配类似 SSN 格式的字符串；注意其局限性（如无法覆盖邮箱、电话号码）。

3. 扩展审计日志以输出 OpenTelemetry GenAI spans。阶段 13 · 20 详细说明了具体属性。

4. 为一个拥有 50 名开发者的团队设计 RBAC 策略，后端包括 notes、github、postgres、jira、slack。谁在每个后端上只有只读权限？谁有写入权限？

5. 通读 Cloudflare 的企业 MCP 博文。找出 Cloudflare 提供的一项本标准库网关所不具备的功能。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Gateway | “MCP 代理” | 位于客户端与后端之间的中心化服务器 |
| Credential vaulting | “后端令牌保留在服务端” | 开发者永远无法看到上游令牌 |
| Session-aware routing | “多后端会话” | 网关为每个开发者会话多路复用 N 个后端会话 |
| Tool-hash pinning | “已批准清单” | 每个已批准工具描述的 SHA256 哈希值；集中拦截恶意篡改 |
| RBAC | “基于用户的策略” | 针对工具和服务器基于角色的访问控制 |
| Policy-as-code | “声明式规则” | 在网关层强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| Audit log | “谁、做了什么、何时” | 用于合规的仅追加事件日志 |
| Rate limit | “基于用户的令牌桶” | 每分钟上限以防止滥用 |
| Official MCP Registry | “权威上游” | `registry.modelcontextprotocol.io`，经过命名空间验证 |
| Reverse-DNS naming | “注册表命名空间” | `io.github.user/server` 约定 |

## 延伸阅读

- [官方 MCP 注册表](https://registry.modelcontextprotocol.io/) —— 权威上游源，经过命名空间验证
- [Cloudflare — 企业级 MCP](https://blog.cloudflare.com/enterprise-mcp/) —— 结合 OAuth 与策略的网关模式
- [agentic-community — MCP 网关注册表](https://github.com/agentic-community/mcp-gateway-registry) —— 开源参考网关
- [TrueFoundry — 什么是 MCP 网关？](https://www.truefoundry.com/blog/what-is-mcp-gateway) —— 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) —— IBM 推出的企业级网关
