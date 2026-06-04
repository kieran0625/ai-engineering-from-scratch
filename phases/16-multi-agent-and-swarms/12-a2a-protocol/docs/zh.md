# A2A —— 智能体对智能体协议

> Google 于 2025 年 4 月发布了 A2A；截至 2026 年 4 月，该规范已更新至 https://a2a-protocol.org/latest/specification/，并获得 150 多家组织的支持。A2A 是 MCP（第 13 课）的横向补充：MCP 是垂直的（智能体 ↔ 工具），而 A2A 是对等的（智能体 ↔ 智能体）。它定义了智能体卡片（发现机制）、带有工件的任务（文本、结构化数据、视频）、不透明的任务生命周期以及身份验证。生产系统越来越倾向于将 MCP 与 A2A 结合使用。Google Cloud 在 2025-2026 年间将 A2A 支持集成到了 Vertex AI Agent Builder 中。

**类型：** 学习 + 实践
**语言：** Python（标准库、`http.server`、`json`）
**前置要求：** 阶段 16 · 04（基础模型）
**耗时：** 约 75 分钟

## 问题

你的智能体需要调用另一个系统上的智能体。该如何实现？你可以暴露一个 HTTP 端点，定义一套自定义的 JSON 架构，并祈祷对方能识别它。每一对智能体的交互都会变成一次定制化的集成开发。

A2A 正是为此类调用设计的通用网络协议。它提供标准的发现机制、标准的任务模型、标准的传输层和标准的工件格式。如同 HTTP+REST 之于 Web，但将智能体视为一等公民。

## 概念

### 四大核心要素

**智能体卡片（Agent Card）**。位于 `/.well-known/agent.json` 的 JSON 文档，用于描述智能体：名称、技能、端点、支持的模态、身份验证要求。通过读取该卡片即可完成发现。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务（Task）**。工作的基本单元。一个具有生命周期的异步有状态对象：`submitted → working → completed / failed / canceled`。客户端发送任务后，轮询或订阅更新。

**工件（Artifact）**。任务产生的结果类型。包括文本、结构化 JSON、图像、视频、音频。工件采用类型化设计，使不同模态的数据都能作为一等公民处理。

**不透明生命周期**。A2A 并不规定远程智能体*如何*解决任务。客户端仅能看到状态转换和工件；具体实现可自由选用任何框架。

### MCP 与 A2A 的职责划分

- **MCP**（第 13 课）：智能体 ↔ 工具。智能体通过 JSON-RPC 向工具服务器读写数据。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；双方均为具备独立推理能力的智能体。

生产环境中的多智能体系统通常会同时使用两者。一个 A2A 对等节点会在其本地调用 MCP 工具。这种划分使得两类关注点保持清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或使用流式传输：订阅 `/tasks/{id}/events` 的 SSE（Server-Sent Events）以接收推送更新。

### 身份验证

A2A 支持三种常见模式：

- **Bearer Token**——OAuth2 或无符号令牌。
- **mTLS**——双向 TLS；组织间相互证明身份。
- **签名请求**——对载荷进行 HMAC 签名。

身份验证方式在智能体卡片中声明；客户端据此发现并遵守相应规则。

### 截至 2026 年 4 月，已有 150 多家组织加入

企业级采用推动了 A2A 的规模化。核心亮点在于：A2A 已成为企业智能体系统跨越信任边界的通行方案。Google Cloud 已在 Vertex AI Agent Builder 中发布 A2A 支持；Microsoft Agent Framework 也提供了支持；大多数主流框架（LangGraph、CrewAI、AutoGen）均内置了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用。** A 公司的智能体调用 B 公司的智能体。若无 A2A，每对智能体交互都需定制契约。
- **异构框架互通。** LangGraph 智能体调用 CrewAI 智能体，再调用自定义 Python 智能体。A2A 实现了标准化。
- **类型化工件。** 视频结果、结构化 JSON、音频——全部作为一等公民支持。
- **长时任务。** 不透明生命周期配合轮询机制，使长达数小时的任务变得简单直接。

### A2A 的局限场景

- **低延迟微调用。** A2A 的生命周期是异步的。亚毫秒级的智能体间通信不适用；应直接使用 RPC。
- **进程内紧耦合智能体。** 若两个智能体运行在同一 Python 进程中，A2A 的 HTTP 往返开销过大。
- **小型团队。** 规范带来的开销是实实在在的；纯内部使用的智能体可能无需如此正式的规范。

### A2A 与 ACP、ANP、NLIP 对比

2024-2026 年间涌现出多个相关规范：

- **ACP**（IBM/Linux 基金会）—— A2A 的前身，范围较窄。
- **ANP**（智能体网络协议）—— 侧重对等发现，以去中心化为先。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）—— 专注于自然语言内容类型。

截至 2026 年 4 月，A2A 是应用最广泛的对等协议。详细对比请参阅 arXiv:2505.02279（Liu 等人，《智能体互操作协议综述》）。

## 动手实践

`code/main.py` 使用 `http.server` 和 JSON 实现了一个极简的 A2A 服务端与客户端。服务端：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 返回工件。

客户端：

- 获取智能体卡片，
- 提交任务，
- 轮询直至完成，
- 读取工件。

运行：

```
python3 code/main.py
```

该脚本在后台线程中启动服务端，随后在其上运行客户端。你将看到完整的流程：发现、提交、轮询、获取工件。

## 实际应用

`outputs/skill-a2a-integrator.md` 设计了一项 A2A 集成方案：涵盖智能体卡片内容、任务架构、身份验证选择、流式传输与轮询的权衡。

## 交付上线

检查清单：

- **锁定规范版本。** A2A 仍在演进中；智能体卡片应声明协议版本。
- **幂等任务创建。** 重复提交（如网络重试）应仅生成一个任务。
- **工件架构。** 明确声明智能体返回的数据结构；消费者需进行校验。
- **速率限制与身份验证。** A2A 面向公网；需实施标准的 Web 安全策略。
- **失败任务死信队列。** 随时间推移分析模式，排查 recurring 失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端能够发现服务端并接收到正确的工件。
2. 为服务端添加第二个技能（例如“摘要生成”）。更新智能体卡片。编写一个根据任务类型自动选择技能的客户端。
3. 实现一个 SSE 流式端点：`/tasks/{id}/events`，用于发射状态变更事件。客户端需要做哪些不同的处理？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出规范中强制要求但本演示未实现的三项内容。
5. 对比 A2A（基于智能体卡片的发现机制）与 MCP（通过 `listTools` 进行服务端能力列表查询）。自描述型智能体与能力探测之间有何权衡？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| A2A | “智能体对智能体” | 智能体跨系统调用的对等协议。Google 于 2025 年推出。 |
| Agent Card | “智能体的名片” | 位于 `/.well-known/agent.json` 的 JSON 文件，描述技能、端点和身份验证。 |
| Task | “工作单元” | 具有生命周期的异步有状态对象；完成后产出工件。 |
| Artifact | “结果” | 类型化输出：文本、结构化 JSON、图像、视频、音频。作为一等媒体支持。 |
| Opaque lifecycle | “求解过程是智能体的私事” | 客户端仅见状态转换；服务端可自由选择框架或工具。 |
| Discovery | “寻找智能体” | 访问 `GET /.well-known/agent.json` 即可获取卡片。 |
| MCP vs A2A | “工具 vs 对等节点” | MCP：垂直的智能体 ↔ 工具。A2A：水平的智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | “兄弟协议” | 相邻规范；A2A 是 2026 年采用率最高的。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) —— 官方权威规范
- [Google Developers 博客 —— A2A 发布公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) —— 2025 年 4 月上线推文
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) —— 参考实现与 SDK
- [Liu 等人 —— 智能体互操作协议综述](https://arxiv.org/html/2505.02279v1) —— MCP、ACP、A2A、ANP 对比分析
