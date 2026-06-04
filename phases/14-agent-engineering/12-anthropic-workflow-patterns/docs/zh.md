# Anthropic 的工作流模式：简单优于复杂

> Schluntz 和 Zhang（Anthropic，2024年12月）区分了工作流（预定义路径）与智能体（动态工具使用）。五种工作流模式涵盖了大多数场景。从直接调用 API 开始。仅在步骤无法预测时才添加智能体。

**类型：** 学习 + 构建
**语言：** Python (stdlib)
**前置知识：** 第 14 阶段 · 01（智能体循环）
**耗时：** 约 60 分钟

## 学习目标

- 说出 Anthropic 的五大工作流模式：提示词链式调用（prompt chaining）、路由（routing）、并行化（parallelization）、编排器-工作者（orchestrator-workers）、评估器-优化器（evaluator-optimizer）。
- 解释智能体与工作流的区别，以及各自的工程成本。
- 识别何时选择工作流而非智能体（反之亦然）。
- 在标准库中针对脚本化 LLM 实现所有五种模式。

## 问题所在

团队常常为只需单次函数调用的问题去套用多智能体框架。这种成本是实实在在的：框架增加了抽象层，使提示词变得晦涩、隐藏了控制流，并引入了过早的复杂性。Schluntz 和 Zhang 在 2024 年 12 月的文章是业界引用最多的反拨观点：从简单开始，仅在复杂性证明其值得时才引入它。

## 核心概念

### 工作流与智能体

- **工作流（Workflow）**。通过预定义的代码路径编排 LLM 和工具。工程师掌握图结构。
- **智能体（Agent）**。LLM 动态指挥自己的工具并自主执行步骤。模型掌握图结构。

两者各有适用场景。工作流成本更低、速度更快且更易于调试。智能体能够解决开放式问题，但会使故障模式的推理变得更加困难。

### 增强型 LLM

所有五种模式的基础：一个集成了三项能力的 LLM —— 搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用均可利用这些能力。

### 五大模式

1. **提示词链式调用（Prompt chaining）**。第 1 次调用的输出作为第 2 次调用的输入。当任务具有清晰的线性分解时使用。步骤之间可选编程逻辑门控。
2. **路由（Routing）**。分类器 LLM 决定调用哪个下游 LLM 或工具。当类别截然不同的输入需要不同处理方式时使用（例如一级支持 vs 退款 vs 缺陷 vs 销售）。
3. **并行化（Parallelization）**。并发运行 N 次 LLM 调用，聚合结果。两种形态：分块处理（不同数据块）和投票/综合（相同提示词，N 次运行，取多数或综合）。
4. **编排器-工作者（Orchestrator-workers）**。编排器 LLM 动态决定运行哪些工作者（也是 LLM），并综合它们的输出。类似于智能体循环，但编排器不会无限循环。
5. **评估器-优化器（Evaluator-optimizer）**。一个 LLM 提出答案，另一个 LLM 对其进行评估。迭代直到评估器通过。这是 Self-Refine（第 05 课）的泛化版本。

### 工作流优于智能体的场景

- **可预测的任务**。如果你能枚举出步骤，就应该这么做。
- **成本受限的任务**。工作流的步骤数量有界；智能体可能会陷入无限循环。
- **合规性要求的任务**。审计人员希望直接阅读图结构，而不是从轨迹中推断它。

### 智能体优于工作流的场景

- **开放式研究**。当下一步取决于上一步的返回结果时。
- **变长任务**。耗时从几分钟到几小时不等，且步骤数未知的任务。
- **新领域**。当你尚未确定正确的工作流时——先探索，后固化。

### 上下文工程的配套内容

《AI 智能体的有效上下文工程》（Anthropic，2025）形式化了相邻学科：200k 上下文窗口是一个预算，而非容器。该包含什么、何时压缩、何时让上下文增长。在第 14 阶段的上下文压缩课程中有详细介绍（本课程重新编号前的第 14 阶段早期第 06 课）。

## 动手构建

`code/main.py` 针对 `ScriptedLLM` 实现了所有五种工作流模式：

- `prompt_chain(input, steps)` —— 顺序执行。
- `route(input, classifier, handlers)` —— 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` —— N 次运行，聚合结果。
- `orchestrator_workers(task, workers)` —— 编排器选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` —— 循环直至通过。

运行方式：

```
python3 code/main.py
```

每个模式都会打印其执行轨迹。每种模式的代码总行数约为 10-15 行；而框架的成本通常以千行计。

## 实际应用

- 大多数任务直接使用 API 调用。
- 仅当模式真正需要持久状态（LangGraph）、Actor 模型并发（AutoGen v0.4）或角色模板（CrewAI）时才使用框架。
- 当你想要 Claude Code 的底层架构但不想重复造轮子时，可以使用 Claude Agent SDK。

## 交付部署

`outputs/skill-workflow-picker.md` 会根据给定的任务描述选择正确的模式，包括决策依据，以及如果工作流无法满足需求时的重构为智能体的路径。

## 练习

1. 实现带有置信度阈值的路由。低于阈值 -> 升级给人工客服。对于一级支持用例，阈值应设在哪里？
2. 为 `parallel_vote` 添加超时机制。当一个调用挂起时会发生什么？如何在缺少投票的情况下进行聚合？
3. 将 `evaluator_optimizer` 改造为多臂老虎机（bandit）算法：跨迭代保留前 2 个输出，以免后期出现的好结果被后期的坏结果覆盖。
4. 结合提示词链式调用与路由：路由器从三条链中选择一条。测量其与单一大型提示词方案的 Token 成本差异。
5. 挑选你生产环境中的一个功能。绘制工作流图。统计步骤数。在这里使用智能体会真的更好吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Workflow（工作流） | “预定义流程” | 由工程师掌控的 LLM 与工具调用图 |
| Agent（智能体） | “自主 AI” | 由模型掌控的图；动态工具调度 |
| Augmented LLM（增强型 LLM） | “带工具的 LLM” | LLM + 搜索 + 工具 + 记忆；基础原子单元 |
| Prompt chaining（提示词链式调用） | “顺序调用” | 第 N 次调用的输出作为第 N+1 次调用的输入 |
| Routing（路由） | “分类器分发” | 选择由哪条链/模型处理输入 |
| Parallelization（并行化） | “扇出” | N 次并发调用；按分块或投票聚合 |
| Orchestrator-workers（编排器-工作者） | “分发器智能体” | 编排器 LLM 动态选择专家 LLM |
| Evaluator-optimizer（评估器-优化器） | “提议者 + 裁判” | 迭代至评估器通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 五大工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 配套学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 何时状态图能证明其成本合理
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 编排器-工作者模式的商业化产品
