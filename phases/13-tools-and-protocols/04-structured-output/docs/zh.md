# 结构化输出 —— JSON Schema、Pydantic、Zod、约束解码

> “礼貌地请求模型返回 JSON”即使在使用前沿模型时也有 5% 到 15% 的概率失败。结构化输出通过约束解码（constrained decoding）弥补了这一差距：系统会直接阻止模型生成违反模式的 token。OpenAI 的严格模式、Anthropic 的模式类型化工具调用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 以及 Zod 的 `.parse` 都是同一理念的五种表面形式。本课程将构建模式验证器，并定义严格模式契约，供学员在生产级提取流水线中统一使用。

**类型：** 构建
**语言：** Python（标准库、JSON Schema 2020-12 子集）
**前置知识：** 第 13 阶段 · 02（函数调用深度解析）
**耗时：** 约 75 分钟

## 学习目标

- 为提取目标编写符合要求的 JSON Schema 2020-12（正确使用 enum、min/max、required、pattern 等约束）。
- 解释为何严格模式和约束解码提供的保证不同于“生成后验证”。
- 区分三种失败模式：解析错误、模式违规、模型拒绝。
- 交付具备类型化修复与类型化拒绝处理能力的提取流水线。

## 问题所在

一个读取采购订单邮件的智能体需要将自由文本转换为 `{customer, line_items, total_usd}`。这里有三种方法。

**方法一：提示词要求 JSON。** “以 JSON 格式回复，包含字段 customer、line_items、total_usd。”在前沿模型上有效率为 85% 到 95%。但会在六种情况下失败：缺少大括号、尾部逗号、类型错误、幻觉字段、在 token 限制处截断、泄露非 JSON 文本（如“这是您的 JSON：”）。

**方法二：生成后验证。** 自由生成、解析、对照模式验证、失败则重试。可靠但昂贵——每次重试都要付费，且截断 bug 会导致每次发生都额外消耗一轮对话。

**方法三：约束解码。** 提供商在解码时强制执行模式。无效 token 会被从采样分布中屏蔽。输出保证可解析且保证符合模式。失败情况简化为一种：拒绝（模型判定输入不符合模式）。

所有 2026 年的前沿提供商都提供了某种形式的第三种方法。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，若模型拒绝则在响应中包含 `refusal`。
- **Anthropic。** 对 `tool_use` 输入进行模式强制；`stop_reason: "refusal"` 并不存在，但 `end_turn` 且无工具调用即为信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定类型提供 token 级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发射结构化的 `RunResult`，类型为 `InvoiceModel`。
- **Zod (TypeScript)。** 运行时解析器，用于将提供商输出与 Zod 模式进行校验；可与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同主线：声明一次模式，端到端强制执行。

## 核心概念

### JSON Schema 2020-12 —— 通用标准

所有提供商均支持 JSON Schema 2020-12。你最常用的构造如下：

- `type`：取值之一为 `object`、`array`、`string`、`number`、`integer`、`boolean`、`null`。
- `properties`：字段名到子模式的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字）、`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子模式。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI 严格模式增加了三项要求：每个属性都必须列在 `required` 中，处处使用 `additionalProperties: false`，且不得存在未解析的 `$ref`。若违反这些规则，API 将在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从类 dataclass 的模型生成 JSON Schema。Pydantic AI 对此进行了封装，让你只需编写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架会在边缘层将模式转换为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出将以类型化的 `Invoice` 实例形式返回。验证错误会抛出带有类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 语言的等价实现。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，它会转换为 API 的 JSON Schema 负载。

### 拒绝处理

严格模式无法强迫模型回答。如果输入无法匹配模式（例如“这封邮件是一首诗，而不是发票”），模型会输出一个包含原因的 `refusal` 字段。你的代码必须将其作为首要处理结果，而非视为失败。拒绝机制也可用作安全信号：当被要求从受保护内容的邮件中提取信用卡号时，模型会返回附带安全理由的拒绝响应。

### 开源环境下的约束解码

开放权重实现主要采用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：根据模式构建确定性有限自动机（DFA）；在每一步中，屏蔽会违反该自动机的 token 的 logits。
2. **结合 JSON 解析器的 Logit 屏蔽**：与模型同步运行流式 JSON 解析器；在每一步计算合法的下一个 token 集合。
3. **带验证器的推测解码**：廉价的草稿模型提议 token，验证器强制执行模式。

商业提供商在幕后选择其中一种技术。2026 年的最先进技术对于短的结构化输出比纯生成更快，对于长输出则速度大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效的 JSON。在严格模式下不会发生。在非严格提供商上仍可能发生。
2. **模式违规。** 输出可解析但不符合模式。在严格模式下不会发生。在非严格模式下很常见。
3. **拒绝。** 模型拒绝回答。必须作为类型化结果进行处理。

### 重试策略

当你处于非严格模式时（Anthropic 工具调用、非严格 OpenAI、旧版 Gemini），恢复模式如下：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常一次重试就足够了。三次重试可以捕获弱模型的随机波动。超过三次则是模式设计不良的信号：模型在某些输入下无法满足该模式，需要修复提示词或模式。

### 小模型支持

约束解码在小模型上同样有效。在结构化任务中，带有语法强制的 30 亿参数开源模型的表现优于仅靠原始提示词的 700 亿参数模型。这正是结构化输出在生产环境中至关重要的主要原因：它将可靠性与模型规模解耦。

## 实践应用

`code/main.py` 在标准库中提供了一个最小化的 JSON Schema 2020-12 验证器（支持 types、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` 模式，并将伪造的 LLM 输出传入验证器，演示解析错误、模式违规和拒绝路径。在生产环境中，可将伪造输出替换为任何提供商的真实响应。

重点关注：

- 验证器返回带有 path 和 message 的类型化 `[ValidationError]` 列表。这就是你需要暴露给重试提示词的数据结构。
- 拒绝分支**不**进行重试。它会记录日志并返回类型化的拒绝结果。第 14 阶段 · 09 会将拒绝用作安全信号。
- `additionalProperties: false` 检查会在对抗性测试输入上触发，展示了严格模式如何杜绝幻觉字段。

## 交付成果

本课程将产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、工单、简历等），该技能将生成一个兼容严格模式的 JSON Schema 2020-12 和一个与之对应的 Pydantic 模型，并预留类型化的拒绝与重试处理桩代码。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器通过 `minimum` 约束路径拒绝该输入。

2. 扩展验证器以支持带判别器的 `oneOf`。常见场景：`line_item` 要么是产品要么是服务，由 `kind` 标记。此处严格模式有细微规则；请查阅 OpenAI 的结构化输出指南。

3. 将相同的 Invoice 模式写为 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手动编写的模式进行比较。找出 Pydantic 默认设置而手动版本遗漏的那一个字段。

4. 测量拒绝率。构造十个不应被提取的输入（歌词、数学证明、空白邮件），并通过启用严格模式的真实提供商运行它们。统计拒绝次数与幻觉输出次数。这将为你构建感知拒绝的重试策略提供基准事实。

5. 通读 OpenAI 的结构化输出指南。找出它在严格模式中明确禁止、而普通 JSON Schema 允许的单一构造。然后设计一个非必需使用该构造的模式，并将其重构为兼容严格模式的形式。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| JSON Schema 2020-12 | “模式规范” | 现代提供商均支持的 IETF 草案模式方言 |
| Strict mode | “保证符合模式” | OpenAI 的标志位，通过约束解码强制执行模式 |
| Constrained decoding | “Logit 屏蔽” | 解码时强制执行，屏蔽无效的下一个 token |
| Refusal | “模型拒绝” | 输入无法匹配模式时的类型化结果 |
| Parse error | “无效 JSON” | 输出未能解析为 JSON；在严格模式下不可能发生 |
| Schema violation | “形状错误” | 已解析但违反了类型/必填/枚举/范围约束 |
| `additionalProperties: false` | “不允许额外字段” | 禁止未知字段；OpenAI 严格模式所必需 |
| Pydantic BaseModel | “类型化输出” | 生成并验证 JSON Schema 的 Python 类 |
| Zod schema | “TypeScript 输出类型” | 用于提供商输出校验的 TS 运行时模式 |
| Grammar enforcement | “开源约束解码” | 基于有限状态机的 logit 屏蔽，如 outlines / guidance 所示 |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) —— 严格模式、拒绝处理及模式要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) —— 2024 年 8 月发布博文，解释解码保证机制
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) —— 类型化的 output_type 绑定，可序列化至各提供商
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) —— 权威规范说明
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) —— 企业部署说明及严格模式注意事项
