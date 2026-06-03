# 结构化输出与约束解码

> 让 LLM 返回 JSON。大多数时候能得到 JSON。但在生产环境中，"大多数时候"就是问题所在。约束解码通过在采样前修改 logits，把"大多数时候"变成"总是"。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 17（聊天机器人），Phase 5 · 19（子词分词）
**时间：** ~60 分钟

## 问题所在

一个分类器向 LLM 发起提示："返回 {positive, negative, neutral} 中的一个。" 模型返回："The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器崩溃。你的分类器 F1 为 0.0。

自由形式生成不是契约。它只是建议。生产系统需要契约。

2026 年存在三个层次。

1. **提示工程。** 礼貌地请求。"仅返回 JSON 对象。" 在前沿模型上约 80% 有效，小模型上更低。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。在支持的 schema 上可靠。厂商锁定。
3. **约束解码。** 在每一步生成时修改 logits，使模型*无法*输出无效 token。按构造保证 100% 有效。适用于任何本地模型。

本课建立对这三者的直觉，并指明何时该用哪个。

## 核心概念

![约束解码在每一步屏蔽无效 token](../assets/constrained-decoding.svg)

**约束解码的工作原理。** 在每一步生成时，LLM 输出覆盖整个词表（~100k token）的 logit 向量。*logit 处理器* 位于模型和采样器之间。它根据目标语法（JSON Schema、正则表达式、上下文无关语法）中的当前位置，计算哪些 token 是有效的，并将所有无效 token 的 logits 设为负无穷。对剩余 logits 的 softmax 只把概率质量放在有效的后续 token 上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机。每个 token 都有 O(1) 的有效下一 token 查找。基于 FSM，因此递归 schema 需要扁平化。
- **XGrammar / llguidance。** 上下文无关语法引擎。处理递归 JSON Schema。解码开销接近零。OpenAI 在 2025 年的结构化输出实现中致谢了 llguidance。
- **vLLM guided decoding。** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端实现。
- **Instructor。** 基于 Pydantic 的 LLM 包装器。验证失败时重试。跨厂商，但不修改 logits——依赖重试 + 结构化输出感知提示。

### 反直觉的结果

约束解码通常*比*无约束生成*更快*。两个原因。第一，它缩小了下一 token 的搜索空间。第二，聪明的实现会跳过强制 token（如 `{"name": "` 等脚手架——每个字节都是确定的）的生成。

### 让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 之前，模型会在思考前就给出答案。JSON 是有效的。答案是错的。没有任何验证能捕获。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

## 动手构建

### 步骤 1：从零实现正则约束生成

参见 `code/main.py` 了解独立的 FSM 实现。核心思想用 30 行代码表达：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 追踪我们已满足语法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词表 token 可以在不离开接受路径的前提下推进 FSM。

### 步骤 2：用 Outlines 实现 JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远。FSM 使无效输出不可达。

### 步骤 3：用 Instructor 实现跨厂商 Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同机制。Instructor 不触碰 logits。它将 schema 格式化到 prompt 中，解析输出，验证失败时重试（默认 3 次）。适用于任何厂商。重试增加延迟和成本。跨厂商可移植性是卖点。

### 步骤 4：原生厂商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。对支持的 schema 与 Outlines 可靠性持平。无需本地模型管理。锁定到特定厂商。

## 常见陷阱

- **递归 schema。** Outlines 将递归扁平化为固定深度。树状结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **巨大枚举。** 10,000 个选项的枚举编译缓慢或超时。改用检索器：先预测 top-k 候选，再约束到这些。
- **语法过于严格。** 强制 `date: "YYYY-MM-DD"` 正则后，模型无法为缺失日期输出 `"unknown"`。模型会通过编造日期来补偿。允许 `null` 或哨兵值。
- **过早承诺。** 见上文字段顺序陷阱。始终先放推理字段。
- **无 schema 的厂商 JSON mode。** 纯 JSON mode 只保证有效 JSON，不保证符合你的用例。始终提供完整 schema。

## 如何选择

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生厂商结构化输出 |
| 任意厂商，Pydantic 工作流，可接受重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines (FSM) |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 批处理，可接受重试 | Instructor + 最便宜模型 |

## 交付物

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习

1. **简单。** 对小型开源权重模型（如 Llama-3.2-3B）发起 `Review(sentiment, confidence, evidence_span)` 提示，不使用约束解码。在 100 条评论上测量解析为有效 JSON 的比例。
2. **中等。** 同一语料库使用 Outlines JSON mode。比较合规率、延迟和语义准确性。
3. **困难。** 从零实现电话号码（`\d{3}-\d{3}-\d{4}`）的正则约束解码器。在 1000 个样本上验证 0 无效输出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Constrained decoding | 强制有效输出 | 在每一步生成时屏蔽无效 token 的 logits。 |
| Logit processor | 那个做约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；O(1) 有效下一 token 查找。 |
| CFG | 上下文无关语法 | 处理递归的语法；比 FSM 慢但更具表达力。 |
| Schema field order | 这重要吗？ | 重要——第一个字段会承诺；始终先放推理再放答案。 |
| Guided decoding | vLLM 的叫法 | 同一概念，集成到推理服务器中。 |
| JSON mode | OpenAI 早期版本 | 保证 JSON 语法；不保证符合 schema。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) —— Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) —— 基于快速 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) —— 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) —— API 参考 + 注意事项。
- [Instructor library](https://python.useinstructor.com/) —— 跨厂商的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) —— 6 个约束解码框架的基准测试。
