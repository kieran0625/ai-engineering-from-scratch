# 聊天机器人 —— 从基于规则到神经网络再到 LLM 智能体

> ELIZA 通过模式匹配回复。DialogFlow 映射意图。GPT 从权重中生成答案。Claude 运行工具并验证结果。每个时代都解决了前一个时代最糟糕的失败。

**类型：** 学习
**语言：** Python
**前置条件：** Phase 5 · 13（问答系统），Phase 5 · 14（信息检索）
**时间：** ~75 分钟

## 问题所在

用户说"我想改签航班"。系统必须弄清楚用户想要什么、缺少什么信息、如何获取这些信息，以及如何完成操作。然后用户说"等等，如果我取消呢？"，系统必须记住上下文、切换任务并保持状态。

对话对机器学习系统来说很困难。输入是开放式的。输出必须在多轮对话中保持连贯。系统可能需要对世界采取行动（改签航班、刷卡扣费）。每一步错误都会暴露给用户。

聊天机器人架构经历了四种范式，每一种都是因为前一种失败得太明显而引入的。本课按顺序介绍它们。2026 年的生产环境是最后两种的混合体。

## 核心概念

![聊天机器人演进：基于规则 → 检索 → 神经网络 → 智能体](../assets/chatbot.svg)

**基于规则（ELIZA、AIML、DialogFlow）。** 手工编写的模式匹配用户输入并生成回复。意图分类器将请求路由到预定义的流程。槽位填充状态机收集所需信息。在设计范围内的狭窄场景中表现卓越。一旦超出范围立即失败。仍然部署在安全关键领域（银行认证、航空预订），因为这些场景无法容忍幻觉。

**基于检索。** FAQ 风格的系统。编码每一对（话语，回复）。运行时，编码用户的消息并检索最接近的存储回复。想想 Zendesk 经典的"相似文章"功能。比规则更好地处理同义改写。没有生成，因此没有幻觉。

**神经网络（seq2seq）。** 在对话日志上训练的编码器-解码器模型。从零生成回复。流畅但容易产生通用输出（"我不知道"）和事实漂移。永远无法可靠地保持主题。这就是 Google、Facebook 和 Microsoft 在 2016-2019 年聊天机器人表现令人失望的原因。

**LLM 智能体。** 一个被包裹在循环中的语言模型，能够规划、调用工具和验证结果。不是带有长提示词的聊天机器人。而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的 grounding（RAG）防止幻觉。工具调用让它真正执行操作。这就是 2026 年的架构。

这四种范式不是顺序替代关系。2026 年的生产聊天机器人会路由通过所有四种：基于规则用于认证和破坏性操作，检索用于 FAQ，神经生成用于自然措辞，LLM 智能体用于模糊的开放式查询。

## 动手实现

### 步骤 1：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行代码实现 ELIZA。反射技巧（"我感到难过" → "你为什么感到难过"）是 Weizenbaum 1966 年经典心理治疗师演示的核心。至今仍有启发意义。

### 步骤 2：基于检索（FAQ）

以下示意代码片段需要 `pip install sentence-transformers`（引入了 torch）。本课可运行的 `code/main.py` 使用标准库中的 Jaccard 相似度替代，因此本课无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级处理。

### 步骤 3：神经生成（基线）

使用小型指令调优的编码器-解码器模型（FLAN-T5）或微调后的对话模型。单独在 2026 年生产环境中不可使用（矛盾、离题漂移、事实 nonsense），但作为混合系统的一部分用于自然措辞。DialoGPT 风格的仅解码器模型需要显式的轮次分隔符和 EOS 处理才能生成连贯回复；FLAN-T5 的 text2text 管道在教学示例中可以开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤 4：LLM 智能体循环

2026 年生产环境的形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需要说明三点。工具是 LLM 可以调用的可调用函数。当 LLM 返回最终答案而非工具调用时，循环终止。步骤预算防止模糊任务上的无限循环。

真实生产环境还会添加：检索优先的 grounding（每次 LLM 调用前注入相关文档）、护栏（未经确认拒绝破坏性操作）、可观测性（记录每一步）和评估（自动检查智能体行为是否符合规范）。

### 步骤 5：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：确定性规则用于任何破坏性操作，检索用于预置 FAQ，LLM 智能体用于其他一切。这就是 2026 年客户支持系统的部署方式。

## 实际应用

2026 年技术栈：

| 使用场景 | 架构 |
|---------|------|
| 预订、支付、认证 | 基于规则的状态机 + 槽位填充 |
| 客户支持 FAQ | 基于精选答案的检索 |
| 开放式帮助聊天 | 带 RAG + 工具调用的 LLM 智能体 |
| 内部工具 / IDE 助手 | 带工具调用的 LLM 智能体（搜索、读取、写入） |
| 陪伴 / 角色聊天机器人 | 带角色设定系统提示词的微调 LLM，基于知识检索 |

生产环境始终使用混合路由。没有单一架构能很好地处理每个请求。路由层本身通常是一个小型意图分类器。

## 仍然存在的失败模式

- **自信的虚构。** LLM 智能体声称完成了并未执行的操作。缓解措施：验证结果、记录工具调用、绝不让 LLM 在没有成功工具返回的情况下声称已完成某事。
- **提示注入。** 用户插入覆盖系统提示词的文本。在 OWASP LLM 应用 Top 10 2025 中排名 LLM01。两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编码基准测试中，前沿模型的实测成功率约为 0.5-8.5%。特定高风险设置（针对 AI 编码智能体的自适应攻击、脆弱的编排系统）已达到约 84%。生产环境 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）—— 由攻击者控制的邮件触发的 Microsoft 365 Copilot 零点击数据泄露漏洞。

  缓解措施：在整个循环中将用户输入视为不可信；工具调用前进行清理；将工具输出与主提示词隔离；使用 Plan-Verify-Execute（PVE）模式，即智能体先规划，然后针对该规划验证每个动作再执行（这阻止工具结果注入新的未计划动作）；破坏性操作需要用户确认；对工具范围应用最小权限原则。

  再多的提示工程也无法完全消除这种风险。需要外部运行时防御层（LLM Guard、允许列表验证、语义异常检测）。
- **范围蔓延。** 智能体因工具调用返回了间接相关的信息而偏离任务。缓解措施：缩小工具契约；保持系统提示词聚焦；添加离题率评估。
- **无限循环。** 智能体反复调用同一工具。缓解措施：步骤预算、工具调用去重、LLM 评判"我们是否正在取得进展"。
- **上下文窗口耗尽。** 长对话将早期轮次推出上下文。缓解措施：总结较早轮次、通过相似度检索相关历史轮次，或使用长上下文模型。

## 部署上线

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习题

1. **简单。** 用 10 个模式实现上述基于规则的回复，用于咖啡店点餐机器人。测试边界情况：双份订单、修改、取消、意图不明确。
2. **中等。** 构建混合 FAQ + LLM 降级系统。为 SaaS 产品准备 50 条预置 FAQ 条目，LLM 降级基于文档站点检索。在 100 个真实支持问题上测量拒绝率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送邮件）实现上述智能体循环。运行 50 个测试场景的评估，包括提示注入尝试。报告离题率、任务失败率和任何注入成功情况。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Intent（意图） | 用户想要什么 | 分类标签（book_flight、reset_password）。路由到处理程序。 |
| Slot（槽位） | 一条信息 | 机器人需要的参数（日期、目的地）。槽位填充是一系列询问。 |
| RAG | 检索加生成 | 检索相关文档，然后为 LLM 的回复提供 grounding。 |
| Tool call（工具调用） | 函数调用 | LLM 发出包含名称 + 参数的结构化调用。运行时执行并返回结果。 |
| Agent loop（智能体循环） | 规划、行动、验证 | 控制器，运行 LLM 调用并与工具调用交错，直到任务完成。 |
| Prompt injection（提示注入） | 用户攻击提示词 | 试图覆盖系统提示词的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) —— 原始基于规则的聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) —— Google 晚期神经聊天机器人论文，就在 LLM 智能体接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) —— 命名智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) —— 2024 年生产指导，在 2026 年仍然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) —— 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) —— 使提示注入成为首要安全关切的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) —— 实用的编排层防御，包括 Plan-Verify-Execute 和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) —— 间接提示注入导致的典型零点击数据泄露 CVE。说明为什么写入权限的智能体需要运行时防御的参考案例。
