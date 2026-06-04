# MCP 生产环境认证 — 客户端注册、JWKS 刷新与受众绑定令牌

> 第 16 课在内存中实现了 OAuth 2.1 状态机。到 2026 年，你交付给真实组织的每个 MCP 服务器都将置于生产环境认证之后：支持无限扩展客户端规模的客户端注册机制（优先使用客户端 ID 元数据文档，动态客户端注册作为向后兼容的回退方案）、授权服务器元数据发现（RFC 8414 *或* OpenID Connect Discovery）、不会导致凌晨 3 点令牌验证中断的 JWKS 缓存刷新机制，以及拒绝跨资源重放的受众绑定令牌。本课通过三个角色——授权服务器、资源服务器（即 MCP 服务器）和客户端——对完整交互面进行建模，以便你追踪从发现到验证工具调用的每一步。
>
> **规范说明（2025-11-25）：** 2025 年 11 月的 MCP 认证规范将动态客户端注册从 `SHOULD` 降级为 `MAY`，并将**客户端 ID 元数据文档（CIMD）**设为推荐的默认注册机制。本课按规范的优先级顺序教授两者，代码中保留 DCR 是因为它在单个进程中完全自包含。

**类型：** 构建
**语言：** Python (stdlib)
**前置要求：** 阶段 13 · 16（OAuth 2.1 状态机），阶段 13 · 17（网关）
**时间：** 约 90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证其契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可完成注册。
- 按计划缓存和刷新 JWKS 密钥，确保签名验证能够安全度过密钥轮换期。
- 使用 RFC 8707 资源指示符将令牌绑定到单个 MCP 资源，并拒绝混淆副手（confused-deputy）重用。
- 清晰分离三个角色——授权服务器、资源服务器、客户端——使每个角色仅执行属于它的检查。
- 阅读身份提供商（IdP）能力矩阵，并在 IdP 无法满足 MCP 认证配置文件时拒绝部署。

## 问题所在

第 16 课的模拟器在内存中运行 OAuth 2.1。生产环境存在三个纯内存模拟器无法察觉的操作缺口。

第一个缺口是注册（enrollment）。一个真实的组织运行着数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会手动为每个 Cursor 用户注册为 OAuth 客户端。2025-11-25 规范给出了客户端解决此问题的优先级顺序：如果有预注册的 `client_id` 则使用它，否则使用**客户端 ID 元数据文档**（客户端使用其控制的 HTTPS URL 标识自身，授权服务器*拉取*该元数据），否则回退到**RFC 7591 动态客户端注册**（客户端*推送*一个 `POST /register` 并立即获得一个 `client_id`），最后才提示用户。CIMD 是推荐的默认值，因为它完全消除了逐服务器注册的需求，同时保留了基于 DNS 的信任模型；DCR 则为了向后兼容而保留。两者都从授权服务器的元数据中发现其入口点：CIMD 对应 `client_id_metadata_document_supported`，DCR 对应 `registration_endpoint`。

第二个缺口是密钥轮换。JWT 验证依赖于授权服务器的签名密钥，这些密钥以 JSON Web Key Set (JWKS) 形式发布。授权服务器按计划轮换这些密钥（通常每小时一次，事故响应期间可能更快）。如果在启动时只获取一次 JWKS，MCP 服务器在轮换窗口之前验证正常，但窗口过后直到重启前所有请求都会失败。生产环境将 JWKS 配置为带刷新任务的缓存值，该任务会在旧密钥过期前覆盖缓存，同时在缓存未命中时提供备用获取逻辑，以应对收到由比缓存中更新密钥签名的令牌的情况。

第三个缺口是受众绑定。第 16 课引入了 RFC 8707 资源指示符。在生产环境中，该指示符成为每次请求的硬性声明检查。MCP 服务器将 `token.aud` 与其自身的规范资源 URL 进行比较，不匹配则返回 HTTP 401。这是防止上游 MCP 服务器（或持有本应属于另一台服务器的令牌的恶意客户端）在同一信任网格中对另一台服务器重放该令牌的唯一防御手段。

本课将每个缺口映射到交互面的具体部分。元数据文档是一个 HTTP 端点。JWKS 缓存刷新是一个定时任务加键值缓存。JWT 验证是资源服务器在分发任何工具调用前运行的例程。保持三个角色分离，每个角色仅强制执行其负责的检查：授权服务器负责颁发和轮换密钥，资源服务器负责缓存和验证，客户端负责发现和注册。

## 核心概念

### RFC 8414 — OAuth 授权服务器元数据

位于 `/.well-known/oauth-authorization-server` 的文档描述了客户端所需的一切信息：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

获得 MCP 资源 URL 的客户端会链式执行发现流程：`oauth-protected-resource`（来自 RFC 9728，即资源服务器的文档）指明发行者，然后 `oauth-authorization-server`（本文档）指明所有端点。客户端绝不硬编码授权 URL。

在信任 IdP 用于 MCP 之前，你需要验证的契约如下：

- `code_challenge_methods_supported` 必须包含 `S256`（基于 RFC 7636 的 PKCE）。规范明确指出：如果此字段**缺失**，表示授权服务器不支持 PKCE，客户端**必须**拒绝继续。
- `grant_types_supported` 必须包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 必须至少提供一种注册路径：`client_id_metadata_document_supported: true`（CIMD，首选）**或** `registration_endpoint`（RFC 7591 DCR，回退）。满足其一即可符合契约；你不再强制要求 DCR。
- `response_types_supported` 对于 OAuth 2.1 必须精确等于 `["code"]`。

如果 `S256` 缺失，MCP 服务器将拒绝针对此 IdP 部署——PKCE 没有降级模式。如果*既未*提供注册路径，你又没有预注册的 `client_id`，你也无法完成注册；此时是部署清单有误，而非代码问题。

### RFC 9728（回顾）— 受保护资源元数据

第 16 课已涵盖 RFC 9728。生产环境的差异在于：此文档是客户端查找*此* MCP 服务器所信任的授权服务器的唯一位置。单个 MCP 服务器可接受来自多个 IdP 的令牌（例如一个用于员工，一个用于合作伙伴）。RFC 9728 声明了该集合；RFC 8414 记录每个 IdP 的支持情况。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### 客户端 ID 元数据文档（推荐的默认值）

CIMD 将注册从*推送*反转为*拉取*。客户端不使用授权服务器来生成 `client_id`，而是将其控制的 HTTPS URL **直接用作**其 `client_id`。该 URL 解析为一个 JSON 元数据文档；授权服务器在 OAuth 流程中按需拉取。信任根植于 DNS：如果服务器运营商信任 `app.example.com`，它就信任从 `https://app.example.com/client.json` 提供的客户端。无需注册往返交互，无需耗尽 `client_id` 命名空间，无需维护需同步的逐服务器状态。

客户端托管的元数据文档如下：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值**必须**等于其提供的 URL（授权服务器会验证此项；不匹配将被拒绝）。授权服务器通过其 RFC 8414 元数据中的 `client_id_metadata_document_supported: true` 宣告对此的支持。

规范明确指出的两个安全事实：

- **SSRF（服务端请求伪造）。** 授权服务器会拉取攻击者提供的 URL。它必须防范 SSRF（禁止拉取内部/管理端点）。
- **localhost 冒充。** 仅靠 CIMD 无法阻止本地攻击者声称合法客户端的元数据 URL 并绑定任意 `localhost` 重定向 URI。授权服务器**必须**在同意界面清晰显示重定向 URI 的主机名，并**建议**对仅使用 `localhost` 的重定向发出警告。

由于 CIMD 不需要服务器端状态，因此不像 DCR 那样需要搭建注册中心。客户端侧是只读的：将你的元数据文档托管在静态 HTTPS 端点上，让授权服务器拉取即可。

### RFC 7591 — 动态客户端注册（回退 / 向后兼容）

DCR 现已成为 `MAY`，保留它是为了与 2025-11-25 之前的部署及尚未支持 CIMD 的 IdP 向后兼容。若无它（且无 CIMD 或预注册），每个 MCP 客户端（Cursor、Claude Desktop、自定义代理）都需要与 IdP 管理员进行带外交换。有了 DCR，客户端只需发送 POST 请求：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认值。它们仅获得 `client_id`——无需暴露 `client_secret`。PKCE 提供了公共客户端所需的持有证明。

三个生产环境陷阱：

- 注册端点必须按源 IP 进行速率限制。否则，恶意行为者可以脚本化生成数百万次虚假注册并耗尽 `client_id` 命名空间。在注册中心处理请求前运行速率限制检查。
- `software_statement`（为客户端担保的签名 JWT）是某些企业 IdP 的要求。本课程的模拟示例跳过了它；生产环境会接入验证步骤，拒绝除 localhost 重定向 URI 之外的任何未签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。泄露此令牌意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）— 资源指示符

第 16 课确立了其结构。生产环境规则：每次令牌请求都必须包含 `resource=<canonical-mcp-url>`，且 MCP 服务器必须在每次调用时验证 `token.aud` 是否与其自身的规范资源 URL 匹配。规范 URI 是服务器的*最具体*标识符：它使用小写方案和主机，不包含片段，且惯例上不包含尾部斜杠。路径组件**不得**按规则剥离——规范在需要用它标识单个 MCP 服务器时会保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 均为有效的规范 URI。为每台服务器选择一个，并将 `aud` 精确固定为该值。（本课程模拟示例为简洁起见使用裸主机受众如 `https://notes.example.com`；在实际部署中，若在同一源下共托管多个 MCP 服务器，则通过路径区分它们。）

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制性的。本课程的授权码流始终携带 `code_challenge` 和 `code_verifier`。服务器会拒绝任何缺少验证器或验证器哈希值与存储的挑战不匹配的令牌请求。

### MCP 规范 2025-11-25 认证配置文件

MCP 规范（2025-11-25）对 MCP 服务器的认证层必须执行的操作规定得非常精确：

- 实现 RFC 9728 受保护资源元数据，并通过 401 响应上的 `WWW-Authenticate: Bearer resource_metadata="..."` 头**或**已知 URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 使该头变为可选，并提供已知 URI 回退）。元数据 `authorization_servers` 字段**必须**至少命名一个服务器。
- 仅在**每次**请求的 `Authorization: Bearer ...` 中接受令牌——绝不在查询字符串中，也绝不在会话开始时仅做一次验证。
- 按请求验证 `aud`、`iss`、`exp` 及必需的作用域。服务器**必须**验证令牌是否专门为其颁发（受众）；缺失或不匹配的 `aud` 将被拒绝，绝不能视为通配符。
- 在 401/403 响应中，返回携带 `WWW-Authenticate: Bearer` 的 `error=...`，包含 `resource_metadata="<PRM-URL>"` 参数（元数据文档的 URL，*非*裸资源），并在 `insufficient_scope`（403）上附带 `scope="..."`。注意：该参数是 `resource_metadata`，一个发现指针——挑战中不存在 `resource` 参数。
- 授权服务器发现接受**任一** RFC 8414 OAuth 元数据**或** OpenID Connect Discovery 1.0；客户端必须按优先级顺序尝试两者的已知后缀。
- 防**混合攻击（mix-up attacks）**的责任在客户端（而非服务器）：客户端在重定向前记录预期的 `issuer`，并在兑换代码前验证 `iss` 授权响应参数（RFC 9207）。仅靠 PKCE 无法阻止混合攻击，因为客户端会将其 `code_verifier` 交给它被引导到的任何令牌端点。

OAuth 2.1 草案是底层基础；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是交互表面；MCP 规范是配置文件。

### IdP 能力矩阵

并非所有 IdP 都支持完整的 MCP 配置文件。下表记录了截至 2025-11-25 规范的事实性能力声明。它是一个*部署门禁*，而非推荐意见。

CIMD 随 2025-11-25 规范发布，其底层的 OAuth 草案直至 2025 年 10 月才被采纳，因此厂商支持仍在陆续到位——请将下方的“CIMD”视为“当前现状，请在你的租户中核实”，而非永久性陈述。

| IdP 类别 | AS 元数据 (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 资源指示符 | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|---|
| 自托管 (Keycloak) | 是 | 发展中 | 是 | 是 (24.x 起) | 是 | 本课 MCP 配置文件的参考 IdP；完整的 DCR 端到端路径，CIMD 跟踪新规范。 |
| 企业 SSO (Microsoft Entra ID) | 是 | 发展中 | 是 (高级版) | 是 | 是 | DCR 可用性因租户版本而异；部署前请在目标租户中核实。 |
| 企业 SSO (Okta) | 是 | 发展中 | 是 (Okta CIC / Auth0) | 是 | 是 | DCR 在 Auth0（现 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP (通用) | 视情况而定 | 否 | 极少 | 极少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；不提供自助注册。仅用作身份源，在其上层叠加你自己的 MCP 感知授权服务器。 |
| 自定义/自研 | 视情况而定 | 视情况而定 | 视情况而定 | 视情况而定 | 视情况而定 | 如果你自行开发，请交付完整配置文件并优先使用 CIMD。跳过 PKCE 或受众绑定会破坏 MCP 认证契约。 |

部署清单拒绝规则：如果所选 IdP 未在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器将拒绝启动——PKCE 没有降级模式。注册是较软的门禁：你需要*一个*可行的路径（预注册的 `client_id`、`client_id_metadata_document_supported: true` 或 `registration_endpoint`）。仅缺少 DCR 不再触发拒绝，因为 CIMD 或预注册可以覆盖它。

### JWKS 刷新模式（在 AS 轮换，在资源服务器刷新）

保持两个动词分离，因为混淆它们是实际的生产环境 Bug：

- **轮换（Rotate）**是*授权服务器*所做的操作：生成新的签名密钥，在 JWKS 中发布它，稍后退役旧密钥。资源服务器不参与此过程也无法执行此操作——它不持有 IdP 的私钥。
- **刷新（Refresh）**是*资源服务器*所做的操作：重新 `GET` 已发布的 JWKS 到其缓存中。这是资源服务器唯一执行的 JWKS 操作。

生产环境故障模式是缓存过期。通过定时刷新任务加键值缓存来解决。资源服务器运行一个任务（cron、定时器或运行时提供的任何机制），按固定间隔拉取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取。如果令牌的 `kid` 在缓存中缺失，则触发**一次**同步刷新作为回退，然后重新检查。这同时处理两种情况：定时刷新，以及密钥重叠窗口（由全新密钥签名的令牌在下次定时刷新前到达）。

回退**必须是重新拉取，绝不能是轮换**。如果你将缓存未命中路径连接到轮换并生成新密钥，会导致两个问题：(1) 生成新密钥会产生一个*仍然*与令牌不匹配的 `kid`，导致查找依然失败；(2) 向随机 `kid` 值喷洒令牌的攻击者会迫使系统进行无限制的密钥创建——自毁式 DoS。重新拉取是可幂等的，因此错误的 `kid` 最多只浪费一次拉取。

缓存结构如下：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时拥有两个密钥是稳态。授权服务器在退役前一个密钥（`k_2026_03`）之前引入下一个密钥（`k_2026_04`），因此用旧密钥颁发的令牌在过期前仍然有效。缓存保存并集；验证器通过 `kid` 进行选择。

### 验证例程

MCP 服务器在分发任何工具调用前运行验证。其结构 `code/main.py` 如下：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存中解析签名密钥（未命中时刷新一次），验证签名，然后检查 `iss` 是否在白名单中、`aud` 是否与此服务器的规范资源匹配、`exp` 以及必需的作用域——首次失败时返回 `WWW-Authenticate` 挑战。将其保持在资源服务器上的单一例程中，意味着每个入口点（每次工具调用、每种传输）都经过相同的检查；不存在绕过验证直接访问工具的路径。

### 受众重放演练（访问令牌权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）均向同一授权服务器注册。服务器 A 被攻破。攻击者获取用户的笔记令牌并将其重放到服务器 B。

服务器 B 的验证器：

1. 解码 JWT，通过 `kid` 获取 JWKS，验证签名。
2. 检查 `iss` 是否与其受保护资源元数据中的 `authorization_servers` 匹配。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——令牌的 `aud` 为 `https://notes.example.com`。）
4. 返回 401 并附带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众声明是协议层防御此攻击的唯一手段。为性能而跳过它是常见的生产环境失误；验证器必须在每次请求时运行，而不仅仅在会话开始时。规范将此称为**访问令牌权限限制**：MCP 服务器 `MUST` 拒绝任何未在受众中提及自身的令牌。

> **命名说明。** 规范将术语*混淆副手（confused deputy）*保留给一个相关但不同的问题：充当第三方 API OAuth **代理**的 MCP 服务器，使用静态客户端 ID，在未获得每客户端用户同意的情况下转发令牌。受众绑定修复了上述重放问题；混淆副手的修复需要每客户端同意**加上**绝不将通过的入站令牌透传到上游 API（MCP 服务器 `MUST` 获取自己的独立上游令牌）。

### 混合攻击（服务器无法提供的客户端侧防御）

客户端在其生命周期内会与许多授权服务器通信。恶意 AS 可以尝试让客户端在攻击者的令牌端点兑换诚实 AS 的授权码。受众绑定在此无济于事——攻击发生在任何令牌生成之前。防御措施位于客户端（RFC 9207）：

1. 重定向前，客户端记录从已验证 AS 元数据中获得的预期 `issuer`。
2. 在授权响应中，客户端在将代码发送到任何地方之前，将返回的 `iss` 参数与该记录的发行者进行比较（简单字符串比较，无需规范化）。
3. 不匹配（或当 AS 宣告 `authorization_response_iss_parameter_supported` 时 `iss` 缺失）→ 拒绝，甚至不显示 `error` 字段。

仅靠 PKCE 无法阻止混合攻击，因为客户端会将其 `code_verifier` 交给它被引导到的任何令牌端点。这就是为什么规范要求在每次请求时将发行者与 PKCE 验证器和 `state` 一起记录。

### 故障模式

- **JWKS 过期。** 验证器在 AS 轮换密钥后拒绝有效令牌。修复方法是上述的 cron 刷新 + 缓存未命中重新拉取模式。切勿在没有刷新任务的情况下缓存 JWKS。
- **以轮换作为回退。** 将缓存未命中路径连接到轮换并生成新密钥而不是重新拉取是一个真实 Bug：它永远不会产生缺失的 `kid`，并且会将攻击者控制的 `kid` 值转化为密钥创建 DoS。回退必须是幂等的 `refresh-jwks`。
- **缺失 `aud` 声明。** 某些 IdP 默认省略 `aud`，除非令牌请求中包含 `resource`。验证器必须拒绝缺失 `aud` 的令牌，不能将缺失视为通配符。
- **通过缺失 `iss` 检查导致的混合攻击。** 未验证 RFC 9207 `iss` 授权响应参数是否与重定向前记录的发行者匹配的客户端，可能会被引导至在攻击者的令牌端点兑换诚实 AS 的代码。这是客户端侧故障；资源服务器无法弥补。
- **作用域升级竞态。** 同一用户的两个并发逐步升级流程可能都成功，并生成两个具有不同作用域的访问令牌。验证器必须使用请求中呈现的令牌，而不是查找“用户当前的作用域”——这会创造 TOCTOU 窗口。
- **注册令牌泄露。** 泄露的 `registration_access_token` 允许攻击者重写重定向 URI。在静止状态下对这些令牌进行哈希处理；要求客户端在每次更新时出示明文；怀疑泄露时轮换。
- **未固定 `iss`。** 接受任何 `iss` 的验证器允许攻击者建立自己的授权服务器，为目标受众注册客户端并发行令牌。受保护资源元数据的 `authorization_servers` 列表就是白名单；必须强制执行。

## 使用方式

`code/main.py` 使用 stdlib Python 和三个角色——`AuthorizationServer`、`ResourceServer` 和 `Client`——演示完整的生产环境流程。流程如下：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其注册选项（`client_id_metadata_document_supported` 用于 CIMD，`registration_endpoint` 用于 DCR）以及 `S256` PKCE 支持情况。
3. 本演练走 DCR 回退路径：客户端向 `/register`（RFC 7591）发送 POST 请求并获得 `client_id`。（CIMD 客户端则会展示其自身的 HTTPS `client_id` URL 并跳过此步骤。）
4. MCP 客户端使用 `resource` 指示符（RFC 8707）运行受 PKCE 保护的授权码流（RFC 7636）。
5. MCP 客户端使用 `Authorization: Bearer ...` 在 MCP 服务器上调用工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存中解析签名密钥。
7. IdP 轮换密钥；定时刷新任务将 JWKS 重新拉取到缓存中。
8. 下一次调用使用刷新后的密钥进行验证而无需重启，且在重叠窗口期内之前的令牌仍然有效。
9. 针对不同 MCP 资源的受众重放尝试将收到 401 响应，附带 `audience mismatch` 和 `resource_metadata` 指针。

此处的 JWT 使用 HS256 和共享密钥（以便课程仅使用标准库运行）。生产环境使用 RS256 或 EdDSA 配合上述 JWKS 模式；验证逻辑其余部分完全相同。由于 IdP 和资源服务器位于同一进程中，`refresh_jwks` 直接读取授权服务器的密钥列表；通过网络传输时则是向 `jwks_uri` 发起 HTTP `GET` 请求。

## 交付物

本课产出 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和 IdP 能力集，该技能会输出待搭建的认证交互面——受保护资源元数据、要使用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、作用域映射，以及在 IdP 不支持完整 RFC 配置文件时应应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪流程。注意 IdP 如何在第 6 步轮换密钥，定时 `refresh_jwks` 如何重新拉取已发布的集合，以及旧令牌（重叠窗口）和新令牌如何在无需重启的情况下均通过验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加一个新 IdP。签发由新 IdP 签名的令牌并确认验证器接受它。签发由未列出的 IdP 签名的令牌并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝它。

3. 向 `register_client` 添加一个在注册中心接受请求前运行的速率限制检查。使用按源 IP 划分的令牌桶，存储在以 IP 为键的小型字典中。

4. 阅读 RFC 7591 并识别课程 `/register` 处理器未验证的两个字段。添加验证逻辑。（提示：`software_statement` 和 `redirect_uris` URI 方案。）

5. 添加客户端 ID 元数据文档路径。提供一个 `client.json`，使其 `client_id` 等于其自身 URL，并让授权服务器拉取并验证它（若 `client_id` ≠ URL 则拒绝）。确认 CIMD 客户端可在无 `register_client` 调用的情况下完成注册。

6. 证明 DoS 修复有效。向验证器发送带有随机 `kid` 的令牌，确认 `refresh_jwks` 最多运行一次且授权服务器的密钥数量不增长。然后故意将回退重新连接至轮换并生成新密钥，观察每个错误令牌密钥数量攀升——随后恢复为重新拉取。

7. 实现客户端侧 RFC 9207 `iss` 检查（来自混合攻击部分）：在授权请求前记录预期发行者，然后拒绝 `iss` 不匹配的授权响应。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| ASM | “OAuth 元数据文档” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | “客户端元数据 URL” | 客户端 ID 元数据文档 —— 用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。自 2025-11-25 起推荐默认 |
| DCR | “自助客户端注册” | RFC 7591 `POST /register` 流程；在 2025-11-25 中被降级为 `MAY` 回退 |
| JWKS | “JWT 验证的公钥” | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| Rotate vs refresh | “更新密钥” | *Rotate* = AS 生成/退役签名密钥；*refresh* = 资源服务器重新拉取已发布的集合。资源服务器永远只执行 refresh |
| Resource indicator | “受众参数” | RFC 8707 `resource` 参数，将令牌固定到单个服务器 |
| `aud` claim | “受众” | 验证器与规范资源 URL 进行比较的 JWT 声明 |
| Audience replay | “令牌重放” | 为服务器 A 颁发的令牌提交给服务器 B；通过受众验证防御（规范术语：访问令牌权限限制） |
| Confused deputy | “代理令牌滥用” | 具有静态客户端 ID 的 MCP 代理在未获每客户端同意的情况下转发令牌；与受众重放不同 |
| Mix-up attack | “错误的令牌端点” | 客户端被引导至在攻击者端点兑换诚实 AS 的代码；通过客户端侧 RFC 9207 `iss` 防御 |
| `iss` allow-list | “受信任的授权服务器” | 受保护资源元数据中 `authorization_servers` 命名的集合 |
| `resource_metadata` | “在哪里找到 PRM 文档” | 在 401/403 响应中命名 RFC 9728 元数据 URL 的 `WWW-Authenticate` 参数 |
| Public client | “原生或浏览器客户端” | 无 `client_secret` 的 OAuth 客户端；PKCE 予以补偿 |
| `WWW-Authenticate` | “401/403 响应头” | 携带驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP 认证配置文件
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 版本的变更内容（CIMD、XAA、DCR 降级）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD 优于 DCR 的理由
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（回退路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防御混合攻击的 `iss` 参数
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 整合后的 OAuth 底层基础
