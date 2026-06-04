# MCP 安全 II —— OAuth 2.1、资源指示器与增量作用域

> 远程 MCP 服务器需要授权，而不仅仅是身份验证。2025-11-25 版规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）保持一致。SEP-835 增加了增量作用域同意机制，并在收到 403 WWW-Authenticate 响应时触发升级授权。本课程将升级流程实现为状态机，以便你观察每一步跳转。

**类型：** 构建
**语言：** Python（标准库、OAuth 状态机模拟器）
**前置要求：** 阶段 13 · 09（传输层）、阶段 13 · 15（安全 I）
**预计时间：** 约 75 分钟

## 学习目标

- 区分资源服务器与授权服务器的职责。
- 掌握受 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆副手攻击。
- 实现升级授权：服务器返回 403 并附带 WWW-Authenticate 头要求更高权限的作用域；客户端重新提示用户同意并重试请求。

## 问题背景

早期的 MCP（2025 年之前）在远程服务器上采用临时的 API 密钥甚至无认证的方式。2025-11-25 版规范通过完整的 OAuth 2.1 配置档案填补了这一空白。

三个实际应用场景：

- **普通远程服务器。** 用户安装访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。采用带 PKCE 的 OAuth 2.1 是最合适的方案。
- **作用域升级。** 已授予 `notes:read` 的笔记服务器后续可能因特定操作需要 `notes:write`。无需重新执行完整流程，升级授权（SEP-835）会直接请求额外作用域。
- **防止混淆副手攻击。** 客户端持有仅对服务器 A 有效的令牌。若服务器 A 恶意尝试将该令牌出示给服务器 B，资源指示器（RFC 8707）可将令牌绑定至其预期受众。

OAuth 2.1 并非新技术。新变化在于 MCP 的配置档案：规定了特定的必需流程（仅限授权码 + PKCE；默认不使用隐式流或客户端凭证），强制要求在每次令牌请求中使用资源指示器，并发布受保护资源元数据以便客户端知晓目标地址。

## 核心概念

### 角色

- **客户端。** MCP 客户端（如 Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（如笔记服务、GitHub、Postgres 等）。
- **授权服务器。** 负责颁发令牌。可以是与资源服务器相同的服务，也可以是独立的身份提供商（IdP，如 Auth0、Keycloak、Cognito）。

在 MCP 的配置档案中，资源服务器和授权服务器可以是同一主机，但应通过 URL 进行区分。

### 授权码 + PKCE

流程如下：

1. 客户端生成 `code_verifier`（随机值）和 `code_challenge`（SHA256 哈希值）。
2. 客户端将用户重定向至 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意授权。授权服务器重定向至 `redirect_uri?code=...`。
4. 客户端向 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...` 发起 POST 请求。
5. 授权服务器将验证器的哈希值与存储的挑战值进行比对，验证通过后颁发访问令牌。
6. 客户端在每次向资源服务器发起请求时使用该令牌：`Authorization: Bearer ...`。

PKCE 可防止授权码拦截攻击。资源指示器可防止令牌在其他地方被非法使用。

### 受保护资源元数据（RFC 9728）

资源服务器发布一份 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少了配置负担——客户端仅需知道资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数将令牌的预期受众固定下来。颁发的令牌中包含 `aud: "https://notes.example.com"`。其他接收此令牌的 MCP 服务器会检查 `aud` 并予以拒绝。

### 作用域模型

作用域是空格分隔的字符串。常见的 MCP 约定包括：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理功能（请谨慎使用）
- `profile:read` 用于身份标识

作用域选择应遵循最小权限原则：按需申请当前所需权限，需要更多时再进行升级。

### 升级授权（SEP-835）

用户最初授予了 `notes:read`。随后用户要求代理删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端捕获到 `insufficient_scope` 错误，通过同意弹窗提示用户授予额外作用域，为其执行一次轻量级 OAuth 流程，然后使用新令牌重试请求。

### 令牌受众验证

每次请求：服务器检查 `token.aud == self.resource_url`。若不匹配则返回 401。此举可阻止跨服务器复用令牌。

### 短期令牌与轮换

访问令牌应为短期有效（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 禁止令牌透传

采样服务器（阶段 13 · 11）严禁将客户端令牌透传给其他服务。采样请求即为边界。

### 防止混淆副手攻击

令牌绑定至 `aud`。客户端绑定至 `client_id`。每次请求均需同时验证两者。规范明确禁止了 MCP 远程工具生态系统中曾普遍存在的“透传令牌”旧模式。

### 客户端 ID 发现

每个 MCP 客户端都在固定 URL 发布其元数据。授权服务器可获取该元数据文档以发现重定向 URI 和联系方式。这消除了手动注册客户端的需求。

### 网关与 OAuth

阶段 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，发给客户端的令牌由网关颁发，且上游令牌绝不会流出网关。这翻转了信任模型——用户只需向网关认证一次；网关代为处理与 N 个服务器的授权交互。

## 实践应用

`code/main.py` 以状态机形式模拟完整的 OAuth 2.1 升级流程。它实现了：

- PKCE 代码验证器/挑战值生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- 针对 `insufficient_scope` 的升级授权。

本课程不启动 HTTP 服务器；状态机在内存中运行，便于你追踪每一步跳转。阶段 13 · 17 的网关课程将将其接入实际的传输层。

## 交付成果

本课程将产出 `outputs/skill-oauth-scope-planner.md`。给定一个带有工具的远程 MCP 服务器，你将设计作用域集合、绑定规则以及升级策略。

## 练习

1. 运行 `code/main.py`。追踪双作用域升级流程。注意升级过程中哪些步骤会重复执行。

2. 添加刷新令牌轮换机制：每次刷新都颁发新的刷新令牌并使旧令牌失效。模拟窃取刷新令牌在轮换后被使用的情况，并确认其失败。

3. 使用标准库 `http.server` 将受保护资源元数据端点实现为真实的 HTTP 响应。参考第 09 课中的 `/mcp` 端点进行镜像实现。

4. 为 GitHub MCP 服务器设计作用域层级：读取仓库、写入 PR、审批 PR、合并 PR、管理员。在各层级之间使用升级授权。

5. 阅读 RFC 8707 和 RFC 9728。找出 RFC 9728 中 MCP 用法与示例不同的一个字段。（提示：它与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OAuth 2.1 | “现代 OAuth” | 整合后的 RFC，强制要求 PKCE 并禁止隐式流 |
| PKCE | “持有证明” | 代码验证器与挑战值结合，用于防御授权码拦截 |
| 资源指示器 | “令牌受众” | RFC 8707 中的 `resource` 参数，将令牌绑定至单一服务器 |
| 受保护资源元数据 | “发现文档” | RFC 9728 定义的 `.well-known/oauth-protected-resource` |
| 升级授权 | “增量同意” | SEP-835 流程，支持按需添加作用域 |
| `insufficient_scope` | “403 配合 WWW-Authenticate” | 服务器信号，提示用户重新同意更大范围的作用域 |
| 混淆副手 | “跨服务令牌复用” | 攻击者利用可信持有者不当转发令牌的行为 |
| 短期令牌 | “访问令牌 TTL” | 快速过期的 Bearer 令牌；通过刷新令牌续期 |
| 作用域层级 | “最小权限栈” | 分级作用域集合，各层级间通过升级授权衔接 |
| 客户端 ID 元数据 | “客户端发现文档” | 客户端发布自身 OAuth 元数据的固定 URL |

## 延伸阅读

- [MCP — 授权规范](https://modelcontextprotocol.io/specification/draft/basic/authorization) —— MCP 官方 OAuth 配置档案
- [den.dev — MCP 十一月授权规范](https://den.dev/blog/mcp-november-authorization-spec/) —— 2025-11-25 变更详解
- [RFC 8707 — OAuth 2.0 资源指示器](https://datatracker.ietf.org/doc/html/rfc8707) —— 负责令牌受众绑定的 RFC
- [RFC 9728 — OAuth 2.0 受保护资源元数据](https://datatracker.ietf.org/doc/html/rfc9728) —— 负责发现文档的 RFC
- [Aembit — MCP OAuth 2.1、PKCE 与 AI 授权的未来](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) —— 升级流程实战指南
