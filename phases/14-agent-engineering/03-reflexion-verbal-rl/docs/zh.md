# Reflexion：语言强化学习

> 基于梯度的强化学习需要数千次试验和 GPU 集群才能修复一种失败模式。Reflexion（Shinn 等人，NeurIPS 2023）通过自然语言实现这一目标：每次试验失败后，智能体会撰写反思，将其存储在情景记忆中，并以此作为下一次试验的条件。这也是 Letta 的休眠期计算、Claude Code 的 CLAUDE.md 学习记录以及 pro-workflow 的 learn-rule 背后的模式。

**类型：** 构建
**语言：** Python (标准库)
**前置知识：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 02（ReWOO）
**耗时：** 约 60 分钟

## 学习目标

- 说出 Reflexion 的三个组成部分（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 使用标准库实现带有二元评估器、反思缓冲区和新重试机制的 Reflexion 循环。
- 为给定任务在标量、启发式和自评估反馈源之间做出选择。
- 解释为什么语言强化能够捕捉到基于梯度的强化学习需要数千次试验才能修复的错误。

## 问题背景

智能体未能完成任务。在标准强化学习中，你需要运行数千次额外的试验，计算梯度，更新权重。这成本高昂且缓慢，而大多数生产环境中的智能体并没有为每次失败分配训练预算。

Reflexion（Shinn 等人，arXiv:2303.11366）提出了一个不同的问题：如果智能体只是思考自己为何失败，并在下次尝试时将这一思考放入提示词中呢？无需更新权重，无需计算梯度。只需在试验之间存储自然语言。

结果如下：在 ALFWorld 上，它击败了 ReAct 和其他未微调的基线模型；在 HotpotQA 上，它优于 ReAct；在代码生成（HumanEval/MBPP）上，它在当时达到了最先进水平。所有这些都不需要任何梯度步骤。

## 核心概念

### 三个组成部分

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

再加一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验运行 Actor。Evaluator 对其进行评分。如果分数较低，Self-Reflector 会生成一条反思（例如：“我选错了工具，因为我把原本问 Y 的问题误读成了问 X”）。该反思会被存入情景记忆。下一次试验从头开始，但能看到这条反思。

### 三种评估器类型

1. **标量（Scalar）**——外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或不通过。最简单，信号最强。
2. **启发式（Heuristic）**——预定义的失败特征。“如果智能体连续两次执行了相同的操作，标记为卡住。”“如果轨迹超过 50 步，标记为低效。”
3. **自评估（Self-evaluated）**——LLM 对自己的轨迹进行评分。在没有真实标签（ground truth）时需要使用。信号较弱；与基于工具的验证配合良好（课程 05 — CRITIC）。

2026 年的默认配置是混合模式：有可用信号时使用标量评估，没有时使用自评估，并将启发式规则作为安全护栏。

### 为何具有通用性

Reflexion 与其说是一种新算法，不如说是一个命名模式。几乎所有生产环境中的“自愈型”智能体都在运行某种变体：

- Letta 的休眠期计算（课程 08）：一个独立的智能体对过往对话进行反思并写入内存块。
- Claude Code 的 `CLAUDE.md` / “保存记忆”模式：将反思捕获为学习记录，并追加到未来的会话中。
- pro-workflow 的 `/learn-rule` 命令：将修正捕获为显式规则。
- LangGraph 的反思节点：一个对输出进行评分并在需要时路由至优化流程的节点。

它们都源于同一个洞察：自然语言是一种足够丰富的媒介，可以在多次运行之间传递“我从失败中学到了什么”。

### 适用场景与不适用场景

在以下情况下 Reflexion 有效：

- 存在明确的失败信号（测试失败、工具报错、答案错误）。
- 任务类别可复现（同一类问题可以再次提问）。
- 反思有空间改进轨迹（有足够的动作预算）。

在以下情况下 Reflexion 无济于事：

- 智能体第一次尝试就成功了。
- 失败是外部原因导致的（网络中断、工具损坏）——反思“网络中断了”对后续运行没有帮助。
- 反思变成了迷信——存储关于某次偶然波动的叙述。

2026 年常见陷阱：记忆腐烂（memory rot）。反思不断累积；其中一些已过时或错误；随着情景缓冲区的增长，重新运行的速度会变慢。缓解措施：定期压缩（课程 06）、设置反思的 TTL，或使用独立的休眠期清理智能体（Letta）。

## 动手构建

`code/main.py` 在一个玩具谜题上实现了 Reflexion：生成一个总和等于目标值的 3 元素列表。Actor 生成候选列表；Evaluator 检查总和；Self-Reflector 写下一行关于出错原因的说明。该反思会被存入情景记忆，供下一次试验使用。

组件：

- `Actor` —— 一种脚本化策略，在看到反思时会进行改进。
- `Evaluator.binary()` —— 针对目标总和的通过/失败判定。
- `SelfReflector` —— 生成一行失败的诊断说明。
- `EpisodicMemory` —— 带有 TTL 语义的有界列表。

运行方式：

```
python3 code/main.py
```

追踪日志显示进行了三次试验。试验 1 失败，存储了一条反思；试验 2 看到反思后有所改进但仍失败；试验 3 成功。与基线运行（无反思）对比——基线会一直卡在试验 1 的答案上。

## 实际应用

LangGraph 将反思作为节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情景缓冲区外化为 Markdown 文件。Letta 的休眠期计算会在空闲时段运行 Self-Reflector，从而使主智能体保持低延迟。OpenAI Agents SDK 并未直接内置 Reflexion；你需要通过自定义 Guardrail（根据分数拒绝轨迹）和跨运行持久化的 memory `Session` 来构建它。

## 部署上线

`outputs/skill-reflexion-buffer.md` 创建并维护一个情景缓冲区，具备反思捕获、TTL 和去重功能。给定任务类别和失败情况，它会生成一条真正有助于下一次试验的反思（而不是泛泛的“请更仔细一点”）。

## 练习

1. 将二元评估器切换为返回距离度量的标量评估器（距离目标值有多远）。收敛速度是否更快？
2. 为反思添加 10 次试验的 TTL。过了这个时间点，旧反思是有害还是有益？
3. 实现启发式评估器：如果重复执行相同操作，则将试验标记为卡住。这与 Self-Reflector 如何交互？
4. 使用无视反思的对抗性 Actor 运行 Reflexion。需要什么样的最小化反思提示词工程才能迫使 Actor 注意到它们？
5. 阅读 Reflexion 论文中关于 AlfWorld 的第 4 节。从概念上复现 130% 的成功率提升：与原始 ReAct 相比，关键差异是什么？

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | “自我修正” | Shinn 等人 2023 —— Actor、Evaluator、Self-Reflector 加上情景记忆 |
| Verbal reinforcement | “无梯度学习” | 将自然语言反思追加到下一次试验的提示词中 |
| Episodic memory | “单次任务反思” | 针对单一任务类别的先前反思有界缓冲区 |
| Scalar evaluator | “二元成功信号” | 来自真实标签的通过/失败或数值分数 |
| Heuristic evaluator | “基于模式的检测器” | 预定义的失败特征（例如：死循环、步骤过多） |
| Self-evaluator | “LLM 自评自身轨迹” | 无真实标签时的降级备选方案——需与基于工具的验证配合使用 |
| Memory rot | “过期反思” | 情景缓冲区被过时条目填满；可通过压缩或 TTL 修复 |
| Sleep-time reflection | “异步自我反思” | 在非主路径上运行 Self-Reflector，以保持主智能体的高速响应 |

## 延伸阅读

- [Shinn 等人, Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) —— 经典论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) —— 生产环境中的异步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 将情景缓冲区作为上下文的一部分进行管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 反思节点模式
