# 综合项目 13 —— 带有注册表与治理的 MCP 服务器

> 模型上下文协议（MCP）在 2026 年不再只是“未来趋势”，而是成为了默认的工具调用规范。Anthropic、OpenAI、Google 以及所有主流 IDE 均内置了 MCP 客户端。Pinterest 发布了其内部的 MCP 服务器生态体系。AAIF 注册表在 `.well-known` 处规范化了能力元数据。AWS ECS 发布了无状态部署参考方案。Block 的 goose-agent 将同一协议集成到了托管式助手内部。2026 年的生产环境形态为：StreamableHTTP 传输层、OAuth 2.1 作用域、OPA 策略门控，以及一个让平台团队能够发现、验证并启用服务器的注册表。请从头到尾构建该系统。

**类型：** 综合项目
**语言：** Python（服务器端，通过 FastMCP）或 TypeScript（@modelcontextprotocol/sdk）、Go（注册表服务）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具与 MCP）、阶段 14（智能体）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**预计耗时：** 25 小时

## 问题描述

MCP 已成为工具调用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及所有托管式智能体现在都消费 MCP 服务器。生产环境中的挑战不在于编写服务器（FastMCP 使这变得很简单），而在于以企业级要求大规模部署它们：按租户划分的 OAuth 作用域、针对破坏性工具的 OPA 策略、支持水平扩展的 StreamableHTTP 无状态架构、用于发现的注册表，以及每次工具调用的审计日志。Pinterest 的内部 MCP 生态体系和 AAIF 注册表规范确立了 2026 年的行业标准。

你将构建一个暴露 10 个内部工具的 MCP 服务器（只读 Postgres 查询、S3 列表、Jira、Linear、Datadog 等）、一个供平台使用的注册表 UI，以及针对破坏性工具的人工审批网关。负载测试将演示 StreamableHTTP 的水平扩展能力。审计追踪将满足企业安全审查要求。

## 概念

MCP 2026 修订版规定 StreamableHTTP 为默认传输层。与早期的 stdio 和 SSE 形态不同，StreamableHTTP 默认是无状态的：单个 HTTP 端点接受 JSON-RPC 请求、流式响应，并支持用于通知的长连接。无状态意味着在负载均衡器后方可以水平扩展。

授权采用 OAuth 2.1 及按工具划分的作用域。令牌携带如 `jira:read`、`s3:list`、`postgres:query:readonly` 等作用域。MCP 服务器在工具调用时检查作用域，而不仅仅是在会话开始时。对于高风险工具，如果最近 N 分钟内未提升至 `approved:by:human`，服务器将拒绝任何调用——该提升需通过 Slack 审批卡片完成。

注册表是一个独立的服务。每个 MCP 服务器都会暴露一个包含其工具清单、传输 URL 和认证要求的 `.well-known/mcp-capabilities` 文档。注册表进行轮询、验证和索引。平台团队使用注册表 UI 查看可用工具、所需作用域及其所属团队。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输层：基于 HTTPS 的 StreamableHTTP（无状态）
- 认证：通过 SPIFFE / SPIRE 实现工作负载身份的 OAuth 2.1
- 策略：每个工具的 OPA / Rego 规则；每次请求的策略决策服务
- 注册表：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工审批：针对破坏性工具的 Slack 交互式消息
- 部署：AWS ECS Fargate 或 Fly.io，每租户一个服务器或共享但按租户隔离
- 审计：按租户分桶的结构化 JSONL，包含每次调用的血缘信息

## 构建步骤

1. **工具接口。** 暴露 10 个内部工具：Postgres 只读查询、S3 列出对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查找、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 只读。每个工具都有类型化的 schema 和作用域标签。

2. **FastMCP 服务器。** 挂载这些工具。配置 StreamableHTTP 传输层。添加用于 OAuth 令牌内省和作用域强制执行的中间件。

3. **OPA 策略。** 每个工具的 Rego 策略：允许调用的作用域、适用的 PII 脱敏规则、有效的载荷大小限制。每次工具调用时调用决策服务。

4. **注册表服务。** 独立的 Go 或 TS 服务，从已注册的服务器轮询 `.well-known/mcp-capabilities`，使用 JSON Schema 进行验证，并提供列表/搜索/验证/启用-禁用 UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输 URL、负责人团队、SLO。

6. **破坏性工具隔离。** 修改状态的工具（Jira 创建、Linear 创建、Postgres 写入）位于第二个 MCP 服务器上，采用更严格的认证流程：令牌必须具有 `approved:by:human` 作用域，且需在 15 分钟内通过 Slack 卡片提升。

7. **审计日志。** 按租户的追加型 JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 脱敏。

8. **负载测试。** 对 StreamableHTTP 进行 100 个并发客户端测试。通过添加第二个副本演示水平扩展；展示负载均衡器在不依赖会话粘性的情况下重新分配流量。

9. **合规性测试。** 针对两个服务器运行官方 MCP 合规性套件。通过所有必选部分。

## 使用方式

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付说明

`outputs/skill-mcp-server.md` 描述了交付物。一个面向内部工具的生产级 MCP 服务器 + 注册表 + 审计层，具备 OAuth 2.1 作用域和 OPA 门控。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 规范合规性 | StreamableHTTP + 能力清单通过 MCP 合规性测试 |
| 20 | 安全性 | 作用域强制执行、每个工具的 OPA 覆盖范围、密钥管理规范 |
| 20 | 可观测性 | 每次工具调用的审计日志及 PII 脱敏 |
| 20 | 扩展性 | 100 客户端负载测试的水平扩展演示 |
| 15 | 注册表用户体验 | 发现 / 验证 / 启用-禁用工作流 |
| **100** | | |

## 练习

1. 添加新工具（Confluence 搜索）。在不触碰核心服务器的情况下，通过注册表验证流程发布它。

2. 编写一条 OPA 策略，对包含名为 `email`、`ssn` 或 `phone` 列的 Postgres 查询结果进行脱敏。使用探测查询进行练习。

3. 在本地延迟方面对比基准测试 StreamableHTTP 与 stdio。报告每次调用的 p50/p95 延迟。

4. 实施按租户配额限制：每个租户每个工具每分钟最多 N 次调用。通过第二条 OPA 规则强制执行。

5. 运行来自 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 的 MCP 合规性套件，并修复所有失败项。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | “2026 MCP 传输层” | 无状态 HTTP + 流式传输；取代网络服务器的 SSE + stdio |
| Capability manifest | “Well-known 文档” | `.well-known/mcp-capabilities`，包含工具列表、认证要求和传输 URL |
| OPA / Rego | “策略引擎” | 用于根据外部规则授权工具调用的 Open Policy Agent |
| Scope elevation | “人工批准” | 通过 Slack 审批授予的短期作用域，破坏性工具必需 |
| Registry | “工具发现” | 从其能力清单中索引 MCP 服务器的服务 |
| Workload identity | “SPIFFE / SPIRE” | 用于 OAuth 令牌签发的加密服务身份 |
| Conformance suite | “规范测试” | 针对 StreamableHTTP + 工具清单正确性的官方 MCP 测试套件 |

## 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— StreamableHTTP、能力元数据、注册表
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) —— 2026 年注册表规范
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) —— 参考生产部署方案
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) —— 参考内部部署案例
- [Block `goose` MCP usage](https://block.github.io/goose/) —— 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) —— Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) —— 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) —— 工作负载身份参考
