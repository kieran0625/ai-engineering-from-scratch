# 自主编程智能体全景图（2026）

> 在不到三年的时间里，SWE-bench Verified 的得分从 4% 跃升至 80.9%。同一款 Claude Sonnet 4.5 模型在 SWE-agent v1 上得分为 43.2%，在 Cline 自主模式下得分为 59.8%——围绕模型的脚手架现在与模型本身同等重要。OpenHands（前身为 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 操作，而非通过 JSON 工具调用。这些亮眼数字掩盖了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个仅需 1–2 行的修改，而针对相同前沿模型，SWE-bench Pro（10+ 行修改任务）的得分仅在 23–59% 之间。

**类型：** 学习
**语言：** Python（标准库，CodeAct 与 JSON 工具调用对比）
**前置条件：** 第 14 阶段 · 07（工具使用），第 15 阶段 · 01（长周期智能体）
**耗时：** 约 45 分钟

## 核心问题

“哪个编程智能体最好”是个错误的问题。正确的问题是：在与我的工作相匹配的任务分布下，使用我将在生产环境中运行的脚手架，我能获得怎样的端到端可靠性？

2022 年至 2026 年间，该领域认识到：脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承重结构。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分为 43.2%；同一模型在 Cline 自主脚手架中的得分为 59.8%。绝对分值相差 16.6 点，权重完全相同。基础模型只是一个组件；循环机制才是产品。

伴随的问题是基准测试饱和会掩盖性能倒退。SWE-bench Verified 已接近饱和，且简单任务的长尾部分（500 个任务中有 161 个要求修改 ≤2 行）拉高了头部分数。真实世界的质量更适合用 SWE-bench Pro（10+ 行修改）等分布来衡量，在此之上，同样的领先模型得分仍停留在 23–59%。

## 核心概念

### SWE-bench 简介

SWE-bench（Jimenez 等人）选取带有真实补丁和测试套件的 GitHub 实际 Issue，要求智能体生成能使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是一个由人工筛选的 500 个任务子集，已剔除模糊和有缺陷的任务。SWE-bench Pro 是更难的后续版本——要求修改 10 行以上的任务，当前前沿智能体在该数据集上的得分仅为 23–59%。

### 2022 → 2026 趋势曲线揭示了什么

- **2022**：研究模型在原始 SWE-bench 上得分约 4%。
- **2024**：GPT-4 + Devin 风格脚手架得分约 14%；SWE-agent 得分约 12%。
- **2025**：集成在 Aider 和 SWE-agent 中的 Claude 3.5/3.7 Sonnet 将得分推至 40–55% 区间。
- **2026**：Claude Sonnet 4.5 及前沿竞品在 SWE-bench Verified 上达到 70–80%+。Epoch AI 排行榜实时追踪此数据。

这一增长斜率来自三个叠加因素：更好的基础模型、更优的脚手架（CodeAct、反思机制、验证器循环）以及更完善的基准测试（Verified 剔除了噪声）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身为 OpenDevin）采取了一种特定的架构策略：模型不输出由宿主解码执行的 JSON 工具调用，而是直接输出 Python 代码，由类 Jupyter 的内核在沙箱中运行。智能体可以在单次操作中遍历文件、串联工具并捕获自身异常。

权衡取舍：

- **JSON 工具调用**：每次操作仅占一个回合；易于审计；组合能力有限；默认安全，因为每次调用都经过显式验证器。
- **CodeAct**：单次操作可包含整个程序；具备组合性；需要加固的沙箱（OpenHands 使用 Docker 隔离）；故障模式涵盖沙箱运行时允许的任何行为。

两种架构均已投入生产环境。CodeAct 在开源平台（OpenHands、smolagents）中占据主导。JSON 工具调用则在托管服务（Anthropic Managed Agents、OpenAI Assistants）中保持主流，因为这些服务由提供商控制执行器。

### 2026 年生态中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开源平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口 (ACI) | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库 diff 编辑 | 极简脚手架，回归稳定性强 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上得分最高的开源脚手架 |
| Devin (Cognition) | 专有 | 托管虚拟机 + 规划器 | 首个“AI 软件工程师”产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细讲解智能体循环 |

### 为何脚手架占据主导地位

一次编程运行是一条长周期轨迹（第 1 课）。可靠性在步骤间累积。脚手架在以下三个方面带来优势：

1. **检索**：找到正确的待读文件是隐形的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库映射均致力于解决此问题。
2. **验证器循环**：运行测试、阅读堆栈跟踪并重试，能在 SWE-bench 上带来 10+ 分的提升。
3. **故障隔离**：出错时自动回滚的沙箱可防止损害累积。有无验证器循环的同一模型，表现宛如两款不同的产品。

### 基准测试饱和与真实分布

OpenHands 作者与 Epoch AI 均指出，SWE-bench Verified 存在简单任务长尾：500 个任务中有 161 个仅需 1–2 行修改。高分部分由此拉动。SWE-bench Pro 限定为 10+ 行修改，即使对于前沿系统，得分也仅在 23–59% 区间。你的生产环境任务分布几乎肯定更接近 Pro 而非 Verified。

对选择智能体的启示：运行你自身 Bug 积压列表中类似 Pro 的子集。真正有意义的分数，是代表你实际交付内容的任务得分。

## 实践应用

`code/main.py` 在固定的微型任务分布上对比了两种玩具级智能体脚手架：

1. 采用 **JSON 工具调用** 的脚手架，每回合执行一次操作。
2. 采用 **CodeAct** 的脚手架，每次操作可输出小型 Python 代码片段。

两者均使用存根“模型”（确定性规则），以便将比较结果与模型质量解耦。输出结果表明，CodeAct 脚手架能以更少的回合解决更多任务，代价是单次操作的爆炸半径更大。

## 部署实施

`outputs/skill-scaffold-audit.md` 帮助你在采纳前审计拟采用的编程智能体脚手架：包括检索质量、验证器配置、沙箱隔离程度，以及基准测试与实际分布的匹配度。

## 练习

1. 运行 `code/main.py`。每种脚手架在同一任务集上各需多少回合？各自的单次操作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上优于 JSON 工具调用。找出论文承认的一种故障模式，并用一句话说明在生产环境中该模式何时会占主导。

3. 从你的 Bug 积压列表中挑选一项需要跨两个文件修改 10+ 行的任务。估算前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。解释两者差距的原因。

4. SWE-bench Verified 包含 161 个单文件、1–2 行修改的任务。构建一个排除它们的评分指标。排行榜会发生怎样的洗牌？

5. 阅读《Introducing SWE-bench Verified》（OpenAI）。解释用于剔除模糊任务的具体方法学，并指出人工筛选可能遗漏的一类任务。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| SWE-bench | “编程基准测试” | 带有真实补丁和测试套件的 GitHub 实际 Issue |
| SWE-bench Verified | “清洗后的子集” | 500 个人工筛选任务，存在简单任务长尾 |
| SWE-bench Pro | “更难子集” | 10+ 行修改；前沿模型得分 23–59% |
| CodeAct | “代码即操作” | 智能体输出 Python；类 Jupyter 内核在沙箱中执行 |
| JSON tool call | “函数调用” | 每次操作为结构化 JSON 载荷，执行前经验证 |
| Scaffold | “智能体框架” | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI (Agent-Computer Interface) | “SWE-agent 的格式” | 专为 LLM 交互体验设计的命令集，而非人类 Shell |
| Verifier loop | “测试并重试” | 运行测试、读取输出、修订补丁；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez 等人 — SWE-bench](https://www.swebench.com/) — 原始基准测试与方法论。
- [OpenAI — 介绍 SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 人工筛选子集的构建过程。
- [Wang 等人 — OpenHands：面向 AI 软件开发者的开放平台](https://arxiv.org/abs/2407.16741) — CodeAct 架构与事件流设计。
- [Epoch AI — SWE-bench 排行榜](https://epoch.ai/benchmarks) — 实时追踪的得分数据。
- [Anthropic — 测量智能体自主性](https://www.anthropic.com/research/measuring-agent-autonomy) — 长周期编程智能体可靠性的理论框架。
