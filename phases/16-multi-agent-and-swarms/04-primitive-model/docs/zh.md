# 多智能体基本原语模型

> 2026 年发布的每一个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。只有四个基本原语，别无其他：智能体（agent）、交接（handoff）、共享状态（shared state）、编排器（orchestrator）。本课程从零构建它们，在一个玩具系统上运行这四个原语，然后将每个主流框架映射到相同的坐标轴上，让你只需一段话就能读懂任何新发布版本。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** 第 14 阶段（智能体工程），第 16 阶段 · 01（为什么需要多智能体）
**耗时：** 约 60 分钟

## 问题

每六个月就有一个新的多智能体框架发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每次新闻稿都宣称自己是“正确的抽象”。

如果你试图逐个学习它们，你会迅速耗尽精力。API 看起来各不相同。文档对什么是“智能体”的说法互相矛盾。一个框架把共享内存称为“黑板”，另一个叫它“消息池”，第三个叫它“StateGraph”。你开始怀疑这个领域只是在原地打转。

事实并非如此。抛开营销包装，这四个基本原语是稳定的。学一次，用一段话就能读懂每个新框架。

## 概念

### 四大基本原语

1. **智能体（Agent）** —— 系统提示词加上一组工具列表。无状态；每次运行都从其系统提示词和当前消息历史开始。
2. **交接（Handoff）** —— 控制权从一个智能体到另一个智能体的结构化转移。机械实现上，是一个返回新智能体的工具调用，或是一个跟随条件的图边。
3. **共享状态（Shared state）** —— 多个智能体均可读取（有时写入）的任何数据结构。如消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）** —— 决定下一个谁发言的角色。选项包括：显式图（确定性）、LLM 发言选择器（软性）、上一个发言者的交接调用（OpenAI Swarm），或基于队列的调度器（群集架构）。

这就是整个设计空间。每个框架为每个轴选择不同的默认值；其余的都是表层语法。

### 2026 年各框架的映射方式

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回智能体 | 调用方的问题 | LLM 的下一次交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言选择器 | 消息池 | 选择函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 链式传递的任务输出 | 管理器 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 归约器 | 图本身，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 特定于模式的 | 线程 / 上下文 | 特定于模式的 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 产物 | 主机决定 |

表面差异看似巨大。底层：同样的四个旋钮。

### 为什么这很重要

一旦看清这些原语，框架比较就变成了一个简单的检查清单：

- 编排器是信任 LLM 进行路由（Swarm），还是将路由逻辑写死在代码中（LangGraph）？
- 共享状态是全历史记录（GroupChat）还是投影视图（StateGraph 归约器）？
- 智能体能否修改彼此的提示词（CrewAI 管理器），还是只能交接（Swarm）？

这三个问题能解决 80% 的框架选型问题。你将不再盲目寻找“最好的多智能体框架”，而是开始为你真正关心的轴进行设计。

### 无状态洞察

除了共享状态外，所有基本原语都是无状态的。智能体是（提示词，工具）的函数。交接是函数调用。编排器是调度器。**系统中唯一有状态的部分就是共享状态。** 所有有趣的 bug 都藏在这里：记忆污染（第 15 课）、消息排序、版本控制、写入竞争。

隐藏共享状态的框架（Swarm）将问题推给了调用方。集中管理共享状态的框架（LangGraph checkpoint、AutoGen pool）使其可检查，但将协调成本转移到了共享状态实现上。

### 单个原语的解剖结构

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。拥有相同系统提示词和工具的两个智能体是可互换的。所有看起来像智能体专属状态的东西，实际上都存在于共享状态或交接协议中。

#### 交接

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占据主导：

- **函数返回** —— 工具返回下一个智能体。这是 OpenAI Swarm 的模式。智能体将其路由逻辑携带在工具模式中。
- **图边** —— LangGraph。图边是声明式的。LLM 生成一个值；条件选择下一个节点。
- **发言选择** —— AutoGen GroupChat。一个选择函数（有时本身就是 LLM 调用）读取池并决定下一个谁发言。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常更多：结构化产物（CrewAI 任务输出）、类型化上下文（LangGraph 归约器）、外部记忆（MCP、向量数据库）。

两种拓扑结构：**全量池**（每个智能体都能看到所有消息）和**投影视图**（智能体看到角色限定的视图）。全量池简单但扩展性差。投影视图可扩展，但需要预先进行模式设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种变体：

- **静态** —— 图在构建时固定（LangGraph 确定性、CrewAI Sequential）。
- **LLM 选择** —— LLM 读取池并选择下一个发言者（AutoGen、CrewAI Hierarchical）。
- **交接驱动** —— 当前智能体通过调用交接工具来决定（Swarm）。
- **队列驱动** —— 工作节点从共享队列拉取任务；没有显式的下一个发言者（群集架构、Matrix）。

### 框架之间的差异

一旦基本原语确定，剩余的设计决策包括：

- **记忆策略** —— 临时与持久化检查点（LangGraph checkpointer）。
- **安全边界** —— 谁能批准交接（人在回路）。
- **成本核算** —— 按智能体的令牌预算。
- **可观测性** —— 追踪交接、持久化状态以供回放。

所有这些都可以基于基本原语实现。它们都不是新的基本原语。

## 动手构建

`code/main.py` 使用约 150 行 Python 标准库代码实现了这四个基本原语。不依赖真实的 LLM——每个智能体都是一个脚本化策略，以便将重点保持在协调结构上。

该文件导出了：

- `Agent` —— 包含名称、系统提示词、工具、策略函数的数据类。
- `Handoff` —— 返回新智能体的函数。
- `SharedState` —— 线程安全的消息池。
- `Orchestrator` —— 三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示程序将相同的三智能体流水线（研究 → 撰写 → 审查）通过所有三种编排器类型运行一次，并在最后打印消息池。你可以看到输出仅在*谁来选择下一个*上有所不同；智能体和共享状态在多次运行中完全一致。

运行它：

```
python3 code/main.py
```

预期输出：三次编排器运行，每种模式一次。每次都会打印最终的消息池。如果研究员提前决定完成，交接驱动的运行模式会涉及更少的智能体——这正是 LLM 路由权衡的微缩体现。

## 实际应用

`outputs/skill-primitive-mapper.md` 是一项技能，它可以读取任意多智能体代码库或框架文档，并返回四原语映射。在新框架发布时运行它，可以在深入阅读文档之前获得一段话级别的理解。

## 交付使用

在采用新框架之前，为其编写原语映射。如果你做不到，说明文档不完整，或者该框架正在发明第五个基本原语（罕见——检查一下是否有你没见过的共享状态变体）。

将映射关系固化在你的架构文档中。当新团队成员加入时，先发送映射关系，再发送 API 文档。当框架版本更新时，对比映射关系，而不是更新日志。

## 练习

1. 使用不同的智能体策略运行 `code/main.py` 三次。观察编排器的选择如何改变实际运行的智能体。
2. 实现第四种编排器类型：一种队列驱动的编排器，其中智能体轮询共享状态以获取任务。可能发生什么死锁，你如何检测它？
3. 使用 LangGraph 快速入门指南（https://docs.langchain.com/oss/python/langgraph/workflows-agents），并将其重写为四个基本原语。LangGraph 的哪些抽象是 1:1 映射，哪些只是便利封装？
4. 阅读 OpenAI Swarm 示例手册（https://developers.openai.com/cookbook/examples/orchestrating_agents）。指出 Swarm 让哪个基本原语最易用，以及它将哪个原语推给了调用方。
5. 在本表中找到一个完全隐藏共享状态的框架。解释当智能体需要在交接过程中协调而不重新读取历史记录时，会发生什么故障。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Agent（智能体） | “带工具的 LLM” | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff（交接） | “控制权转移” | 命名下一个智能体及可选负载的结构化调用。三种实现：函数返回、图边、发言选择。 |
| Shared state（共享状态） | “记忆” / “上下文” | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| Orchestrator（编排器） | “协调器” | 决定下一个谁运行的角色。静态图、LLM 选择器、交接驱动或队列驱动。 |
| Primitive（基本原语） | “抽象” | 每个框架参数化的四个轴之一。不是框架功能。 |
| Message pool（消息池） | “共享聊天记录” | 全历史记录共享状态。易于推理，但扩展性差。 |
| Projected state（投影状态） | “限定视图” | 针对角色的共享状态视图。可扩展，但需要模式设计。 |
| Speaker selection（发言选择） | “下一个谁说话” | 编排器模式，其中函数（通常是 LLM）从一组中选择下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 交接驱动编排的最清晰阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) —— GroupChat + 发言选择是 LLM 选择编排的参考实现
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— 图边编排与基于归约器的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) —— 角色-目标-背景智能体，Sequential / Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) —— 微软将 v0.4 转入维护后，仍在活跃的 AutoGen v0.2 分支
