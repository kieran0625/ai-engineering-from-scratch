# Self-Refine 与 CRITIC：迭代式输出优化

> Self-Refine（Madaan 等，2023）在一个循环中使用同一个 LLM 扮演三个角色——生成、反馈、优化。在 7 个任务上的平均绝对提升为 +20。CRITIC（Gou 等，2023）通过将验证路由到外部工具来强化反馈步骤。到了 2026 年，该模式已作为“评估器-优化器”（Anthropic）或护栏循环（OpenAI Agents SDK）内置于所有主流框架中。

**类型：** 构建
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 03（反思机制）
**耗时：** 约 60 分钟

## 学习目标

- 阐述 Self-Refine 的三个提示词（generate、feedback、refine），并解释为何历史记录对 refine 提示词至关重要。
- 解释 CRITIC 的核心洞察：在没有外部依据的情况下，LLM 进行自我验证是不可靠的。
- 使用标准库实现带有历史记录和可选外部验证器的 Self-Refine 循环。
- 将该模式映射到 Anthropic 的“评估器-优化器”工作流以及 OpenAI Agents SDK 的输出护栏。

## 问题背景

智能体生成了一个几乎正确的答案。可能是一行代码存在语法错误，可能是一个摘要过长，也可能是一个计划遗漏了边界情况。你希望的是：智能体能够批判性地审视自己的输出，然后进行修正。

Self-Refine 证明了仅凭单一模型即可实现这一目标，无需训练数据，也无需强化学习。但有一个前提：LLM 在处理复杂事实时很不擅长自我验证。CRITIC 指出了改进方案——将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年的迭代优化默认范式：生成、验证（尽可能借助外部工具）、优化，直到验证器通过为止。

## 核心概念

### Self-Refine（Madaan 等，NeurIPS 2023）

一个 LLM，三种角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 会看到完整的历史记录——包括所有先前的输出和批评意见——因此它不会重蹈覆辙。论文对此进行了消融实验：若丢弃历史记录，质量会急剧下降。

核心结论：在包含 GPT-4 在内的 7 个任务（数学、代码、首字母缩写、对话）上，平均绝对提升达 +20。无需训练，无需外部工具，仅使用单一模型。

### CRITIC（Gou 等，arXiv:2305.11738，v4 版 2024年2月）

Self-Refine 的弱点在于：反馈步骤是 LLM 给自己打分。对于事实性声明，这并不可靠（模型产生的幻觉往往在它自己看来很有说服力）。CRITIC 用 `feedback(task, output)` 替换了 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术计算的计算器。
- 领域特定验证器（单元测试、类型检查器、代码风格检查器）。

验证器会基于工具结果生成结构化的批评意见。随后，优化器会基于这些批评意见进行调整。

核心结论：由于批评意见具有外部依据，CRITIC 在事实性任务上优于 Self-Refine。在没有外部验证器的任务（创意写作、格式排版）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形态：

1. **验证器通过。** 外部测试返回成功。在有可用条件时优先采用（如单元测试、类型检查器、护栏断言）。
2. **未发出反馈。** 模型表示“输出没问题”。成本更低但可靠性差；需配合最大迭代次数限制使用。

2026 年默认做法：结合两者。“如果验证器通过，或者模型表示没问题且迭代次数 >= 2，或者迭代次数 >= 最大迭代次数，则停止。”

### 评估器-优化器（Anthropic，2024）

Anthropic 在 2024 年 12 月的文章中将其列为五种工作流模式之一。包含两个角色：

- 评估器：对输出进行评分并生成批评意见。
- 优化器：根据批评意见修改输出。

循环直至评估器通过。这在 Anthropic 的语境下即为 Self-Refine/CRITIC。Anthropic 补充的关键工程细节是：评估器和优化器的提示词应有显著差异，以防止模型只是机械地盖章认可。

### OpenAI Agents SDK 输出护栏

OpenAI Agents SDK 将此模式实现为“输出护栏”。护栏是一种在智能体最终输出上运行的验证器。如果护栏触发异常（抛出 `OutputGuardrailTripwireTriggered`），输出将被拒绝，智能体可重试。护栏可以调用外部工具（CRITIC 风格）或作为纯函数运行（Self-Refine 风格）。

### 2026 年常见陷阱

- **机械盖章循环。** 同一模型使用相同的提示词风格同时负责生成和批评，容易收敛于“看起来不错”。应使用结构不同的提示词，或使用更小、更便宜的模型专门负责批评。
- **过度优化。** 每次优化都会增加延迟和 Token 消耗。建议预算控制在 1-3 次；超过后应升级至人工审核。
- **在简单任务上使用 CRITIC。** 如果没有外部验证器，CRITIC 会退化为 Self-Refine；不要为了一个占位验证器而支付额外的延迟成本。

## 动手实践

`code/main.py` 在一个玩具任务上实现了 Self-Refine 和 CRITIC：根据给定主题生成简短的项目符号列表。验证器检查格式（3 个项目符号，每项不超过 60 个字符）。CRITIC 增加了一个外部“事实验证器”，会对已知的事实幻觉进行惩罚。

组件：

- `generate` —— 脚本化生成器。
- `feedback` —— LLM 风格的自我批评。
- `verify_external` —— CRITIC 风格的外部依据验证器。
- `refine` —— 基于历史记录重写输出。
- 停止条件 —— 验证器通过或最多 4 次迭代。

运行方式：

```
python3 code/main.py
```

对比 Self-Refine 与 CRITIC 的运行结果。CRITIC 捕捉到了 Self-Refine 遗漏的事实性错误，因为外部验证器拥有自我批评所不具备的外部依据。

## 实际应用

Anthropic 的“评估器-优化器”是该模式在 Claude 友好语境下的表述。OpenAI Agents SDK 的输出护栏呈现 CRITIC 形态（护栏可调用工具）。LangGraph 提供了类似 Self-Refine 的反思节点。Google 的 Gemini 2.5 Computer Use 增加了每步安全评估器，这是一种 CRITIC 变体：每个操作在执行前都会经过验证。

## 部署上线

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算配置评估器-优化器循环。它会生成生成器、评估器/验证器和优化器的提示词，以及一套停止策略。

## 练习

1. 将玩具任务的 max_iterations 设为 1 运行。CRITIC 是否仍然有效？
2. 用一个有噪声的验证器替换外部验证器（随机 30% 误报率）。循环会如何表现？这正是 2026 年大多数护栏栈的现实情况。
3. 实现“不同模型分工（生成器与批评者）”的变体：大模型负责生成，小模型负责批评。效果是否优于同模型方案？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。列出三种验证工具类别，并为每种提供一个示例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。该 SDK 做错了什么，又做对了什么？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Self-Refine | “能自我修复的 LLM” | 单模型内的生成 -> 反馈 -> 优化循环，并携带历史记录 |
| CRITIC | “基于工具的验证” | 用外部验证器（搜索、代码、计算、测试）替代反馈步骤 |
| Evaluator-Optimizer | “Anthropic 工作流模式” | 双角色设计——评估器打分，优化器修订——循环直至收敛 |
| Output guardrail | “事后检查” | OpenAI Agents SDK 中的验证器，在智能体生成输出后运行 |
| Verify step | “批评阶段” | 关键决策点：依赖外部依据还是自我评分 |
| Refine history | “模型已尝试过的内容” | 将历史输出与批评意见前置追加至优化提示词；若丢弃则质量骤降 |
| Rubber-stamp loop | “自我认同失效” | 相同提示词的批评只会返回“看起来不错”；需用结构不同的提示词修复 |
| Stop condition | “收敛测试” | 验证器通过 或 无反馈且达到迭代上限；绝不应是单一条件 |

## 延伸阅读

- [Madaan 等，Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— 奠基性论文
- [Gou 等，CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) —— 基于工具的验证
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 评估器-优化器工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/) —— 将输出护栏作为 CRITIC 形态的验证器
