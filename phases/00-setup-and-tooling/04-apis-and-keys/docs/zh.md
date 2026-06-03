# API 与密钥

> 每个 AI API 的工作方式都一样：发送请求，接收响应。细节会变，模式不变。

**类型：** Build
**语言：** Python、TypeScript
**前置要求：** Phase 0, Lesson 01
**时间：** ~30 分钟

## 学习目标

- 使用环境变量和 `.env` 文件安全存储 API 密钥
- 使用 Anthropic Python SDK 和原始 HTTP 两种方式调用 LLM API
- 对比基于 SDK 和原始 HTTP 的请求/响应格式，用于调试
- 识别并处理常见 API 错误，包括认证错误和速率限制

## 问题

从 Phase 11 开始，你将调用 LLM API（Anthropic、OpenAI、Google）。在 Phase 13-16 中，你将构建在循环中使用这些 API 的 agent。你需要了解 API 密钥的工作原理、如何安全存储它们，以及如何完成你的第一次 API 调用。

## 概念

```mermaid
sequenceDiagram
    participant C as Your Code
    participant S as API Server
    C->>S: HTTP Request (with API key)
    S->>C: HTTP Response (JSON)
```

每个 API 调用都包含：
1. 一个 endpoint（URL）
2. 一个 API key（认证）
3. 一个 request body（你想要什么）
4. 一个 response body（你得到什么）

## 动手实现

### 步骤 1：安全存储 API 密钥

切勿将 API 密钥硬编码在代码中。使用环境变量。

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

或者使用 `.env` 文件（将其添加到 `.gitignore`）：

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### 步骤 2：第一次 API 调用（Python）

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
)

print(response.content[0].text)
```

### 步骤 3：第一次 API 调用（TypeScript）

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: "What is a neural network in one sentence?" }],
});

console.log(response.content[0].text);
```

### 步骤 4：原始 HTTP（不使用 SDK）

```python
import os
import urllib.request
import json

url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
}
body = json.dumps({
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method="POST")
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["content"][0]["text"])
```

这就是 SDK 在底层实际做的事情。理解原始 HTTP 调用有助于调试。

## 使用场景

对于本课程：

| API | 何时需要 | 免费额度 |
|-----|---------|---------|
| Anthropic (Claude) | Phases 11-16（agent、工具） | 注册时赠送 $5 额度 |
| OpenAI | Phase 11（对比） | 注册时赠送 $5 额度 |
| Hugging Face | Phases 4-10（模型、数据集） | 免费 |

你现在不需要全部设置。等课程需要时再设置。

## 交付成果

本节课产出：
- `outputs/prompt-api-troubleshooter.md` - 诊断常见 API 错误

## 练习

1. 获取 Anthropic API 密钥并完成你的第一次 API 调用
2. 尝试原始 HTTP 版本，并对比其响应格式与 SDK 版本的区别
3. 故意使用错误的 API 密钥，阅读错误信息

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------|---------|
| API key | "API 的密码" | 唯一字符串，用于标识你的账户并授权请求 |
| Rate limit | "他们在限我流" | 每分钟/每小时的最高请求数，防止滥用并确保公平使用 |
| Token | "一个词"（在 API 语境中） | 计费单位：输入 token 和输出 token 分别计数和计费 |
| Streaming | "实时响应" | 逐字获取响应，而非等待完整响应返回 |
