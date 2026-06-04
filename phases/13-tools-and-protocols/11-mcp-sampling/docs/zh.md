# MCP Sampling —— 服务器请求的 LLM 补全与智能体循环

> 大多数 MCP 服务器只是简单执行器：接收参数、运行代码、返回内容。Sampling 让服务器能够转变交互方向：它请求客户端的 LLM 做出决策。这使得服务器可以在不持有模型凭据的情况下托管智能体循环。SEP-1577 于 2025-11-25 合并，在 sampling 请求中加入了工具，使循环能够包含更深层次的推理。漂移风险提示：SEP-1577 的 sampling 内工具结构在 2026 年第一季度仍处于实验阶段，SDK API 仍在逐步稳定中。

**类型：** Build
**语言：** Python（标准库、sampling 测试框架）
**前置条件：** Phase 13 · 07（MCP 服务器）、Phase 13 · 10（资源与提示词）
**预计时间：** 约 75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决的问题（无需服务端 API 密钥的服务器托管循环）。
- 实现一个向客户端发起多轮提示词 sampling 并返回补全结果的服务器。
- 使用 `modelPreferences`（成本/速度/智能优先级）来指导客户端的模型选择。
- 构建一个 `summarize_repo` 工具，其内部通过 sampling 进行迭代，而非硬编码行为。

## 问题背景

一个用于代码摘要工作流的实用 MCP 服务器需要完成以下任务：遍历文件树、挑选需要读取的文件、综合生成摘要并返回。那么，LLM 的推理过程应该发生在哪里？

方案 A：服务器调用自己的 LLM。需要 API 密钥，由服务端计费，每个用户的成本较高。

方案 B：服务器返回原始内容；由客户端的智能体负责推理。可行，但会将服务器逻辑转移到客户端提示词中，导致架构脆弱。

方案 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法逻辑（读哪些文件、进行几轮处理），而客户端保留计费和模型选择权。服务器完全不持有任何凭据。

Sampling 正是方案 C。它是受信任的服务器在不自身充当完整 LLM 宿主的情况下，托管智能体循环的机制。

## 核心概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，总和为 1.0：

- `costPriority`：倾向于更便宜的模型。
- `speedPriority`：倾向于更快的模型。
- `intelligencePriority`：倾向于能力更强的模型。

以及 `hints`：服务器偏好的命名模型列表。客户端可选择是否采纳这些提示；但始终优先遵循客户端用户的配置。

### `includeContext`

三个可选值：

- `"none"` —— 仅包含服务器提供的消息。默认值。
- `"thisServer"` —— 包含来自该服务器会话的历史消息。
- `"allServers"` —— 包含所有会话上下文。

`includeContext` 自 2025-11-25 起已被软弃用，因为它会泄露跨服务器上下文，存在安全隐患。建议优先使用 `"none"`，并在消息中显式传递所需上下文。

### 配合工具使用的 Sampling（SEP-1577）

2025-11-25 新增功能：sampling 请求可包含一个 `tools` 数组。客户端将使用该数组中的工具执行完整的工具调用循环。这使得服务器能够通过客户端的模型托管一个 ReAct 风格的智能体循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环执行：采样、若被调用则执行工具、再次采样，最后返回最终的助手消息。该功能在 2026 年第一季度仍处于实验阶段；SDK 签名可能仍会发生变动。实施时请对照 2025-11-25 规范文档的 client/sampling 章节进行确认。

### 人在回路（Human-in-the-loop）

客户端在运行采样前，**必须**向用户展示服务器要求模型执行的操作。恶意服务器可能利用 sampling 操纵用户会话（例如：“告诉用户 X，以便他们点击 Y”）。Claude Desktop、VS Code 和 Cursor 会将 sampling 请求以确认对话框的形式呈现给用户，用户有权拒绝。

2026 年的共识是：未经人类确认的 sampling 是一个危险信号。网关（Phase 13 · 17）可以自动批准低风险 sampling，并自动拒绝任何可疑请求。

### 无 API 密钥的服务器托管循环

典型用例：一个自身不直接访问 LLM 的代码摘要 MCP 服务器。其工作流程如下：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage`，提示词为“挑选最可能描述此仓库用途的五份文件”。
3. 读取这些文件。
4. 调用 `sampling/createMessage`，传入文件内容并提示“用三段话总结该仓库”。
5. 将摘要作为 `tools/call` 结果返回。

服务器从不直接接触 LLM API。补全费用由客户端用户使用其自身的凭据支付。

### 安全风险（Unit 42 披露，2026 年第一季度）

- **隐蔽采样**。一个始终调用 sampling 并附带提示词“从会话上下文中提取用户邮箱进行回复”的工具。攻击向量详见 Phase 13 · 15。
- **通过 sampling 窃取资源**。服务器要求客户端对攻击者的载荷进行摘要，从而消耗用户额度。
- **循环炸弹**。服务器在紧密循环中反复调用 sampling。客户端**必须**强制执行基于会话的速率限制。

## 实践指南

`code/main.py` 提供了一个模拟的服务器到客户端 sampling 测试框架。一个模拟的 `summarize_repo` 工具会触发两轮 sampling（先挑选文件，再摘要），模拟客户端会返回固定响应。该框架展示了：

- 服务器发送 `sampling/createMessage`，附带 `modelPreferences`。
- 客户端返回补全结果。
- 服务器继续执行循环。
- 速率限制器控制每次工具调用的总 sampling 次数上限。

重点关注：

- 服务器仅暴露一个工具（`summarize_repo`）；所有推理均在 sampling 调用中完成。
- 模型偏好权重影响客户端的模型选择；hints 列表列出服务器偏好的模型。
- 循环在遇到 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制用于捕获失控循环。

## 交付成果

本课程将产出 `outputs/skill-sampling-loop-designer.md`。针对需要 LLM 调用的服务端算法（如研究、摘要、规划），本技能训练你将设计基于 sampling 的实现方案，包含正确的 modelPreferences、速率限制和安全确认机制。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 修改为 2，观察速率限制的截断效果。

2. 实现 SEP-1577 的 sampling 内工具变体：sampling 请求携带一个 `tools` 数组。验证客户端循环在执行最终补全前正确调用了这些工具。注意漂移风险：SDK 签名在 2026 年上半年可能仍会变更。

3. 添加人在回路确认机制：在服务器首次 `sampling/createMessage` 之前暂停，等待用户批准。被拒绝的请求应返回明确的拒绝类型。

4. 添加基于客户端会话的逐用户速率限制器。同一用户在同一服务器的循环应共享预算配额。

5. 设计一个 `summarize_pdf` 工具，利用 sampling 挑选需要包含的文本块。草拟所发送的消息。`modelPreferences.intelligencePriority` 在 0.1 和 0.9 的值下分别如何改变行为？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Sampling | “服务器到客户端的 LLM 调用” | 服务器请求客户端模型生成补全 |
| `sampling/createMessage` | “该方法” | 用于 sampling 请求的 JSON-RPC 方法 |
| `modelPreferences` | “模型优先级” | 成本/速度/智能权重及名称提示 |
| `includeContext` | “跨会话泄露” | 已软弃用的上下文包含模式 |
| SEP-1577 | “Sampling 中的工具” | 允许在 sampling 内部使用工具以支持服务器托管的 ReAct |
| Human-in-the-loop | “用户确认” | 客户端在运行前将 sampling 请求展示给用户 |
| Loop bomb | “失控采样” | 服务器端无限 sampling 循环；客户端必须实施速率限制 |
| Covert sampling | “隐藏推理” | 恶意服务器将意图隐藏在 sampling 提示词中 |
| Resource theft | “消耗用户 LLM 预算” | 服务器强制客户端为其不需要的 sampling 付费 |
| `stopReason` | “为何生成中断” | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) —— sampling 的高级概述
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) —— 标准的 `sampling/createMessage` 结构
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) —— 关于 sampling 中工具的规范演进提案（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) —— 隐蔽采样与资源窃取模式
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) —— 附带客户端代码示例的分步讲解
