# Agent 框架权衡：LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都在推销同样的演示（研究代理生成报告），却掩盖了同样的缺陷（状态模式与编排层冲突）。选择抽象概念与你问题形态相匹配的框架；其余部分都是你需要重复编写的胶水代码。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 11 · 09（Function Calling）、Phase 11 · 16（LangGraph）
**预计时间：** 约 45 分钟

## 问题所在

你有一个需要多次调用 LLM 的任务。可能是一个研究工作流（规划、搜索、总结、引用）。可能是一个代码审查流水线（解析 diff、提出批评、打补丁、验证）。也可能是一个多轮对话助手，负责预订机票、撰写邮件和提交报销单。于是你选了一个框架。

三天后，你发现该框架的抽象概念开始“泄漏”。CrewAI 给了你角色定义，但当“研究员”需要将结构化计划交给“写手”时，它就开始跟你作对。AutoGen 提供了代理间的聊天功能，但没有一等公民级别的状态管理，导致你的检查点只是一个对话日志的 pickle 序列化文件。LangGraph 给了你一个状态图，但在你还没搞清楚代理会做什么之前，就强迫你命名每一个转换节点。Agno 提供了一个单代理抽象，当你试图将其扩展为三个并发工作者时，它会直接报错抗议。

解决方案不是“挑选最好的框架”，而是让框架的核心抽象与你的问题形态相匹配。本课将为你绘制这张映射图。

## 核心概念

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

在 2026 年的生态格局中，四大框架占据主导地位。它们的核心抽象并不相同。

| Framework | Core abstraction | Best fit | Worst fit |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` —— 强类型状态、节点、条件边、检查点持久化。 | 具有显式状态和人工介入中断的工作流；需要时间旅行调试的生产级代理。 | 拓扑结构未知的松散型、角色驱动的头脑风暴。 |
| **CrewAI** | `Crew` —— 角色（目标、背景故事）、任务、流程（顺序或分层）。 | 角色扮演或人格驱动的工作流，且具备简短的线性/分层计划。 | 超出 Crew 回合历史的任何有状态逻辑；复杂的分支结构。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或多个代理轮流发言，直到满足退出条件。 | 多代理*对话*（师生、提议者-批评者、执行者-审核者），思维从聊天中自然涌现。 | 已知 DAG 的确定性工作流；需要在重启间保持持久状态的场景。 |
| **Agno** | `Agent` —— 单个 LLM + 工具 + 记忆，可组合成团队。 | 快速构建的单代理和轻量级团队；强大的多模态支持和内置存储驱动。 | 具有自定义归约器的深层、显式分支图。 |

### “抽象”究竟指什么

框架的核心抽象就是你在白板前向他人推介架构时会画出的东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转换，每个位置的状态对象都有明确类型。心智模型是状态机。
- **CrewAI** → 你画一张组织架构图。每个角色都有职位描述，经理负责分配任务。心智模型是一组小型专家团队。
- **AutoGen** → 你画一个 Slack 私聊窗口。两个代理互相发消息；如果需要主持人，第三个会加入。心智模型是聊天。
- **Agno** → 你画一个单独的方框，旁边挂着工具。把方框并排放在一起就是一个团队。心智模型是“开箱即用的代理”。

### 状态管理问题

在生产环境中，大多数框架选型失败都源于状态管理。

- **LangGraph。** 强类型状态（`TypedDict` 或 Pydantic 模型）、逐字段归约器、一等公民级别的检查点持久化（SQLite/Postgres/Redis）。断点续传、中断控制和时间旅行调试均免费内置。*（参见 Phase 11 · 16。）*
- **CrewAI。** 状态通过 `context` 字段以字符串形式在任务间流转，或通过 `output_pydantic` 进行结构化。默认不提供持久的单 Crew 存储；如果 Crew 必须能在重启后恢复，你需要自行外挂存储方案。
- **AutoGen。** 状态即聊天记录以及任何用户定义的 `context`。对话记录会持久化；除非编写适配器，否则任意工作流状态不会自动保存。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 绑定到 `Agent` —— 对话会话和用户记忆会自动持久化。它不是完整的图检查点持久化工具，而是一个会话存储。

### 分支决策问题

所有非平凡代理都会涉及分支。谁来决定分支至关重要。

- **LangGraph** —— 由开发者决定，通过条件边实现。路由是一个带有命名分支的 Python 函数。分支在编译后的图中是一等公民；检查点会记录实际走的是哪条分支。
- **CrewAI** —— 在分层模式下由经理决定；在顺序模式下于构建时由开发者决定。路由隐含在任务列表中；除了经理的提示词外，没有第一优先级的“if”逻辑。
- **AutoGen** —— 由代理通过聊天决定。分支是从“下一个谁发言”中自然涌现的。`GroupChatManager` 用于选择下一位发言人；你可以手写一个 `speaker_selection_method`，但默认由 LLM 驱动。
- **Agno** —— 由代理决定下一步调用哪个工具。团队提供协调器/路由器/协作者模式；超出此范围的分支逻辑需由开发者负责。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任何 OTel 导出器支持 OpenTelemetry。每次节点转换都是一个追踪跨度（trace span）；检查点同时可作为可重放的追踪记录。LangSmith 是官方首选；Langfuse/Phoenix 也提供适配器。
- **CrewAI** —— 自 2025 年底起原生支持 OpenTelemetry；支持与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** —— 通过 `autogen-core` 集成 OpenTelemetry；AgentOps 和 Opik 提供连接器。追踪粒度为每条代理消息，而非每个节点。
- **Agno** —— 内置 `monitoring=True` 标志位及 OpenTelemetry 导出器；与 Langfuse 深度集成以支持会话追踪。

### 成本与延迟

这四个框架都会增加单次调用的开销（框架逻辑、校验、序列化）。开销大致递增顺序为：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架额外执行的 LLM 路由次数。CrewAI 的分层经理会消耗 token 来决定下一步由谁执行；AutoGen 的 `GroupChatManager` 同理。LangGraph 仅在你编写 `llm.invoke` 的地方消耗 token。Agno 的单代理路径开销最轻。

当单次运行成本至关重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`），而非 LLM 自动选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 工具、检索器、LLM。提供一等公民级别的 MCP 适配器（工具作为 MCP 服务器导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具和 MCP 工具均可适配接入。通过 `allow_delegation=True` 实现 Crew 间的委托。
- **AutoGen** → `FunctionTool` 可包装任意 Python 可调用对象；提供 MCP 适配器。与 AG2 生态系统紧密耦合，专攻代理间交互模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；提供 MCP 适配器；工具可在代理和团队间共享。

## 核心技能

> 你能用一句话解释，为什么某个特定框架适合解决某个特定的代理问题。

构建前检查清单：

1. **画出形态。** 这是一张图（强类型状态、命名转换）？一场角色扮演（专家交接工作）？一次聊天（代理交谈至完成）？还是一个带工具的单一代理？
2. **决定谁控制分支。** 开发者决定分支 → LangGraph。经理代理决定 → CrewAI 分层模式。聊天自然涌现 → AutoGen。工具调用决定 → Agno。
3. **检查状态预算。** 是否需要断点续传？时间旅行？运行中途的人工中断？如果是，LangGraph 是默认首选；Agno 的会话机制可覆盖对话范围内的状态。
4. **检查成本预算。** LLM 自动选择的路由会在每轮消耗额外 token。如果代理每天运行数千次，请优先选择显式路由。
5. **评估框架开销。** 每个框架都是另一个依赖项。如果任务仅需两次 LLM 调用和一个工具，直接写 30 行纯 Python 即可；没有任何框架比不用框架更便宜。

在你无法画出对应的图、组织架构图、聊天窗口或代理方框之前，拒绝盲目引入框架。拒绝选择那种迫使你为了实际需求而与其状态模型搏斗的方案。

## 决策矩阵

| Problem shape | Preferred framework | Why |
|---------------|---------------------|-----|
| 具有强类型状态、人工审批、长运行的工作流 DAG | LangGraph | 一等公民级别的状态管理、检查点持久化、中断控制、时间旅行调试。 |
| 具有明确分工的研究/写作流水线 | CrewAI（顺序模式）或 LangGraph 子图 | 在 CrewAI 中按角色分配任务成本极低；当分支变复杂时，使用 LangGraph 进行扩展。 |
| 提议者-批评者或师生对话 | AutoGen | 双代理聊天是其原生形态。 |
| 带工具、会话、记忆的单一代理 | Agno | 配置最轻量，内置存储和记忆功能。 |
| 带有归约器的数千次并行扇出 | LangGraph + `Send` | 唯一提供一等公民级别并行分发 API 的框架。 |
| 快速原型验证，无框架绑定需求 | 纯 Python + 提供商 SDK | 没有任何框架比不用框架更快。 |

## 练习

1. **简单。** 针对同一任务——“调研 Anthropic 总部，撰写 200 字简报，引用来源”——分别在 LangGraph（四个节点：规划、搜索、撰写、引用）和 CrewAI（三个角色：研究员、写手、编辑）中实现。汇报单次运行的 token 消耗和代码行数。
2. **中等。** 在 AutoGen（研究员 ↔ 写手聊天，编辑通过 `GroupChat` 加入）和 Agno（单个代理配合 `search_tools` 和 `write_tools`，外加会话存储）中实现同一任务。对四种实现在以下方面进行排名：(a) 单次运行成本，(b) 崩溃后恢复能力，(c) 在撰写步骤前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接收简短的问题描述（JSON 格式：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），返回一条推荐建议及一句话理由。使用你自己设计的六个案例对其进行验证。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 编排（Orchestration） | “代理如何协调” | 决定下一个运行哪个节点/角色/代理的层级。 |
| 持久状态（Durable state） | “重启后恢复” | 进程终止后仍能保留的状态，通常绑定到检查点或会话存储。 |
| LLM 自动路由（LLM-selected routing） | “让模型决定” | 规划型 LLM 每轮选择下一步；灵活但每次决策都会消耗 token。 |
| 显式路由（Explicit routing） | “开发者决定” | 由 Python 函数或静态边选择下一步；成本低且可审计。 |
| Crew | “一个 CrewAI 团队” | 将角色 + 任务 + 流程（顺序或分层）绑定为单个可运行单元。 |
| GroupChat | “AutoGen 的多代理聊天” | 带有发言人选择器的 N 个代理之间的受管对话。 |
| Team (Agno) | “多代理 Agno” | 在一组代理上运行的路由/协调/协作模式。 |
| StateGraph | “LangGraph 的图” | 强类型状态、节点、条件边、检查点持久化的抽象。 |

## 延伸阅读

- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— StateGraph、检查点持久化、中断控制、时间旅行调试。
- [CrewAI 文档](https://docs.crewai.com/) —— Crews、Flows、Agents、Tasks、Processes。
- [AutoGen 文档](https://microsoft.github.io/autogen/) —— ConversableAgent、GroupChat、teams、tools。
- [Agno 文档](https://docs.agno.com/) —— Agent、Team、Workflow、storage、memory。
- [Anthropic —— 构建高效代理（2024年12月）](https://www.anthropic.com/research/building-effective-agents) —— 模式库（提示词链、路由、并行化、编排者-工作者、评估者-优化者），与框架无关。
- [Yao 等人，《ReAct：推理与行动的协同》（ICLR 2023）](https://arxiv.org/abs/2210.03629) —— 每个框架都在包装的核心循环。
- [Wu 等人，《AutoGen：通过多代理对话赋能下一代 LLM 应用》（2023）](https://arxiv.org/abs/2308.08155) —— AutoGen 的设计论文。
- [Park 等人，《生成式代理：人类行为的交互式模拟》（UIST 2023）](https://arxiv.org/abs/2304.03442) —— CrewAI 风格的人格堆栈所基于的角色扮演理论基础。
- Phase 11 · 16（LangGraph）—— 本课进行基准对比的框架。
- Phase 11 · 19（Reflexion）—— 一种能清晰映射到 LangGraph，但在 CrewAI 中映射起来较为别扭的模式。
- Phase 11 · 22（生产环境可观测性）—— 如何对你选择的任何框架进行插桩监控。
