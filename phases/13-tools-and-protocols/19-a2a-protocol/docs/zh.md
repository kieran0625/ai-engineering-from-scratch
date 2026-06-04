# A2A —— 智能体对智能体协议

> MCP 是智能体对工具（agent-to-tool）。A2A（Agent2Agent）是智能体对智能体（agent-to-agent）—— 一项开放协议，旨在让基于不同框架构建的不透明智能体进行协作。该协议由 Google 于 2025 年 4 月发布，同年 6 月捐赠给 Linux 基金会，并于 2026 年 4 月达到 v1.0 版本，获得包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow 在内的 150 多家机构支持。它吸收了 IBM 的 ACP 协议，并增加了 AP2 支付扩展。本课将带你了解 Agent Card、Task 生命周期以及两种传输绑定方式。

**类型：** 构建
**语言：** Python（标准库，Agent Card + Task 运行环境）
**前置知识：** 第 13 阶段 · 06（MCP 基础），第 13 阶段 · 08（MCP 客户端）
**预计时间：** 约 75 分钟

## 学习目标

- 区分智能体对工具（MCP）与智能体对智能体（A2A）的使用场景。
- 在 `/.well-known/agent.json` 发布包含技能与端点元数据的 Agent Card。
- 梳理 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用包含 Parts（文本、文件、数据）的 Messages 以及作为输出的 Artifacts。

## 问题背景

一个客服智能体需要将报告撰写任务委派给一个专门的写作智能体。在 A2A 出现之前的选项有：

- 自定义 REST API。可行，但每次配对都是定制开发。
- 共享代码库。要求两个智能体运行相同的框架。
- MCP。不适用：MCP 用于调用工具，而非两个智能体在保留各自内部推理过程不透明的前提下进行协作。

A2A 填补了这一空白。它将交互建模为一个智能体向另一个智能体发送 Task，并包含生命周期、消息和产物。被调用智能体的内部状态保持不透明——调用方仅能看到任务状态转换和最终输出。

A2A 是“让跨框架的智能体相互通信”的协议。它不会取代 MCP；两者是互补关系。

## 核心概念

### Agent Card

每个符合 A2A 标准的智能体都会在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现机制基于 URL：获取卡片，获知 A2A 端点的 URL，并枚举技能。

### 签名版 Agent Cards (AP2)

AP2 扩展（2025 年 9 月）为 Agent Cards 添加了加密签名。发布者使用 JWT 对其卡片进行签名；消费者负责验证。可防止身份冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起请求。被调用智能体会经历状态流转；客户端可通过 SSE 订阅或轮询来接收状态更新。

### Messages 与 Parts

一条 Message 携带一个或多个 Parts：

- `text` —— 纯文本内容。
- `file` —— 带有 mimeType 的 base64 二进制块。
- `data` —— 类型化的 JSON 负载（供被调用智能体使用的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而非原始字符串。Artifact 是一个命名且类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以分块流式传输。调用方负责累积拼接。

### 两种传输绑定方式

1. **HTTP 上的 JSON-RPC。** `/a2a` 端点，POST 用于请求，可选 SSE 用于流式传输。默认绑定方式。
2. **gRPC。** 适用于原生支持 gRPC 的企业环境。

两种绑定方式承载相同的逻辑消息结构。

### 保持不透明性

核心设计原则之一：被调用智能体的内部状态是不透明的。调用方只能看到任务状态和产物。被调用智能体的思维链（chain-of-thought）、工具调用、子智能体委派——全部不可见。这与 MCP 不同，MCP 中的工具调用是透明的。

设计理由：A2A 使竞争对手能够在不暴露内部细节的前提下进行协作。A2A 可以实现“调用这个客服智能体”，而无需调用方知晓该智能体具体如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux 基金会。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）正式发布。
- **2026-04。** v1.0 版本发布，拥有 150 多家支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|------|-----|-----|
| 使用场景 | 智能体对工具 | 智能体对智能体 |
| 不透明性 | 工具调用透明 | 内部推理不透明 |
| 典型调用方 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 带生命周期的 Task |
| 授权机制 | OAuth 2.1（第 13 阶段 · 16） | JWT 签名的 Agent Cards（AP2） |
| 传输层 | Stdio / Streamable HTTP | HTTP 上的 JSON-RPC / gRPC |

当你需要调用特定工具时使用 MCP。当你需要将整个任务委派给另一个智能体时使用 A2A。许多生产系统会同时使用两者：智能体使用 MCP 作为其工具层，使用 A2A 作为其协作层。

## 实践应用

`code/main.py` 实现了一个极简的 A2A 运行环境：一个研究智能体发布其卡片，一个写作智能体接收包含 PDF 和文本指令的 `tasks/send`，经历 working → input_required → working → completed 的状态流转，并返回一个文本 artifact。全部基于标准库实现；使用内存传输层以聚焦于消息结构。

重点关注：

- Agent Card 的 JSON 结构。
- Task ID 分配与状态流转。
- 包含混合类型 Parts 的 Messages。
- 任务中途的 input_required 分支。
- 完成时的 Artifact 返回。

## 交付成果

本课将生成 `outputs/skill-a2a-agent-spec.md`。给定一个应能被其他智能体调用的新智能体，该技能将生成 Agent Card JSON、技能 Schema 以及端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用智能体请求澄清时的 input_required 暂停阶段。

2. 添加签名版 Agent Card。使用 HMAC 对卡片的规范 JSON 进行签名。编写验证器，并确认其在卡片被篡改时会失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个增量 artifact 块，调用方负责累积它们。

4. 设计一个封装 MCP 服务器的 A2A 智能体。将每个 MCP 工具映射到 A2A 技能。注意权衡取舍——失去了哪些不透明性？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月尚未被任何框架实现的一项功能。（提示：它与多跳任务委派有关。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| A2A | “智能体对智能体协议” | 面向不透明智能体协作的开放协议 |
| Agent Card | “`.well-known/agent.json`” | 描述智能体技能与端点的已发布元数据 |
| Skill | “可调用单元” | 智能体支持的命名操作（类比 MCP 工具） |
| Task | “委派单元” | 具有生命周期和最终产物的工作项 |
| Message | “任务输入” | 携带 Parts（文本、文件、数据） |
| Part | “类型化块” | Message 的 `text` / `file` / `data` 元素 |
| Artifact | “任务输出” | 完成时返回的命名、类型化输出 |
| AP2 | “智能体支付协议” | 用于信任与支付的签名 Agent Cards 扩展 |
| Opacity | “黑盒协作” | 被调用智能体的内部细节对调用方隐藏 |
| Input-required | “任务暂停” | 智能体需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) —— 官方 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) —— 参考实现与 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) —— 2025 年 6 月治理权移交
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) —— 路线图与合作伙伴动态
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) —— v1.0 发行说明与向后兼容指南
