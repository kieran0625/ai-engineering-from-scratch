# Capstone — 构建完整的工具生态系统

> 第 13 阶段讲解了每一个独立模块。本综合项目（Capstone）将它们整合为一个面向生产环境的系统：包含工具、资源、提示词、任务和 UI 的 MCP 服务器，边缘层的 OAuth 2.1 认证，RBAC 网关，多服务器客户端，A2A 子智能体调用，接入收集器的 OTel 追踪，CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 打包套件。完成后，你将能够为每一项架构决策提供充分依据。

**类型：** 构建
**语言：** Python（标准库，端到端生态测试框架）
**前置条件：** 第 13 阶段 · 第 01 至 21 课
**预计时间：** 约 120 分钟

## 学习目标

- 使用 `ui://` 应用构建一个暴露工具、资源、提示词和任务的 MCP 服务器。
- 在服务器前端部署 OAuth 2.1 网关，强制执行 RBAC 权限控制和固定哈希校验。
- 编写支持端到端 OTel GenAI 属性追踪的多服务器客户端。
- 将部分工作负载委托给 A2A 子智能体；验证内部状态的不可见性（Opacity）得以保持。
- 使用 AGENTS.md + SKILL.md 对整个技术栈进行打包，以便其他智能体能够驱动它。

## 问题背景

交付“研究与报告”系统：

- 用户提问：“总结三篇关于智能体协议的高引用率 2026 年 arXiv 论文。”
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要任务委托给专门的写作智能体；聚合结果；将交互式报告渲染为 MCP Apps `ui://` 资源；将每一步操作记录到 OTel。

第 13 阶段的所有基础原语都将在此体现。这并非玩具项目——Anthropic（Claude Research 产品）、OpenAI（配备 Apps SDK 的 GPTs）以及第三方在 2026 年交付的生产级研究辅助系统均采用完全相同的架构形态。

## 概念设计

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 追踪层级结构

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

单一追踪 ID。每个 Span 均携带正确的 `gen_ai.*` 属性。

### 安全架构

- OAuth 2.1 + PKCE，并通过 Resource Indicator 将受众锁定至网关。
- 网关持有上游凭证；用户永远无法直接查看。
- RBAC：`alice` 拥有 `research:read` 和 `research:write` 权限，可调用所有工具。`bob` 仅拥有 `research:read` 权限，无法调用 `generate_report`。
- 固定描述清单：自动剔除任何工具哈希发生变化的服务器。
- “双重规则”审计：没有任何工具会同时结合未受信输入、敏感数据和具有重大影响的执行动作。

### 渲染机制

最终的 `generate_report` 任务返回内容块及一个 `ui://report/current` 资源。客户端宿主环境（如 Claude Desktop 等）会在沙箱 iframe 中渲染交互式仪表盘。该仪表盘包含排序后的论文列表、引用次数统计，以及一个按钮：当用户点击任意论文时，该按钮将调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包与分发

整个系统以以下形式交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户使用 `docker compose up` 进行部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动该系统。

### 第 13 阶段各课的贡献

| 课程 | 综合项目所用内容 |
|--------|------------------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、Schema 定义、代码检查 |
| 06-10 | MCP 基础原语、服务器、客户端、传输层、资源与提示词 |
| 11-14 | 采样策略、Roots 与交互引导、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒防护、OAuth 2.1、网关与注册中心 |
| 18 | A2A 子智能体委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包规范 |

## 运行与使用

`code/main.py` 将之前课程的模式整合为一个可运行的演示程序。全部基于标准库实现，且均为进程内运行，便于你从头到尾阅读源码。它完整跑通了“研究与报告”场景的流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、将 generate_report 作为任务执行、向 writer 发起 A2A 调用、返回 ui:// 资源、输出 OTel Span。

重点关注以下内容：

- 跨所有节点的单一追踪 ID。
- 网关策略阻止第二位用户写入数据。
- 任务生命周期从 working 流转至 completed，并返回文本与 ui:// 内容。
- A2A 调用的内部状态对编排器保持不可见。
- AGENTS.md 和 SKILL.md 是其他智能体重现该工作流所需的唯一文件。

## 交付与应用

本课将产出 `outputs/skill-ecosystem-blueprint.md`。面对具体的产品需求（研究、摘要、自动化），该技能将生成完整的架构方案：明确使用哪些 MCP 原语、哪些网关控制策略、哪些 A2A 调用、哪些遥测指标以及何种打包方式。

## 练习

1. 运行 `code/main.py`。观察单一的追踪 ID 以及 Span 的嵌套关系。统计该演示程序涉及了多少个第 13 阶段的基础原语。

2. 扩展演示程序：添加第二个后端 MCP 服务器（例如 `bibliography`），并确认网关将其工具合并到了同一命名空间中。

3. 用运行在子进程中的真实 A2A 写作智能体替换掉模拟版本。使用第 19 课的测试框架。

4. 在编排器与 LLM 之间的路由网关中添加 PII（个人身份信息）脱敏步骤。确认用户查询中的电子邮件地址已被成功清除。

5. 为负责维护此系统的团队成员编写一份 AGENTS.md。阅读时间应控制在五分钟以内，并提供他们在 Cursor 或 Codex 中驱动该综合项目所需的一切信息。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Capstone | “第 13 阶段集成演示” | 使用所有基础原语的端到端系统 |
| Research and report | “该场景” | 搜索、摘要、渲染模式 |
| Ecosystem | “所有组件集合” | 服务器 + 客户端 + 网关 + 子智能体 + 遥测 + 包 |
| Trace hierarchy | “单一追踪 ID” | 每个节点的 Span 共享同一 Trace；通过 Span ID 建立父子关系 |
| Gateway-issued token | “传递式认证” | 客户端仅能看到网关令牌；网关持有上游凭证 |
| Merged namespace | “所有工具平铺在一个列表中” | 网关处进行多服务器合并，冲突时添加前缀 |
| Opacity boundary | “A2A 调用隐藏内部细节” | 编排器无法窥探子智能体的推理过程 |
| Three-layer stack | “AGENTS.md + SKILL.md + MCP” | 项目上下文 + 工作流 + 工具 |
| Defense-in-depth | “多层安全防护” | 固定哈希、OAuth、RBAC、双重规则审计、审计日志 |
| Spec compliance matrix | “我们交付了规范要求的内容” | 将交付物映射至 2025-11-25 版规范的检查清单 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 整合参考文档
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议演进方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考指南
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 标准化追踪约定
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级智能体运行时模式
