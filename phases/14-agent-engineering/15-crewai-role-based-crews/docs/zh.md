# CrewAI：基于角色的团队与工作流

> CrewAI 是 2026 年推出的基于角色的多智能体框架。四大核心原语：智能体（Agent）、任务（Task）、团队（Crew）、流程（Process）。两种顶层形态：团队（Crew，自主的基于角色协作）与工作流（Flow，事件驱动、确定性执行）。文档直言不讳：“对于任何生产就绪的应用，请从工作流开始。”

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** 第 14 阶段 · 12（工作流模式），第 14 阶段 · 14（Actor 模型）
**耗时：** 约 75 分钟

## 学习目标

- 说出 CrewAI 的四大原语（Agent、Task、Crew、Process）及其各自职责。
- 区分 Sequential（顺序）、Hierarchical（分层）和计划中的 Consensus（共识）流程；根据负载类型选择其一。
- 区分团队（Crew，自主基于角色）与工作流（Flow，事件驱动确定性），并解释文档的生产环境建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类接入工具；辨析结构化输出与自由文本的差异。
- 列举 CrewAI 的四种记忆类型及其适用场景。
- 使用标准库实现一个包含三个智能体（研究员、撰稿人、编辑）的团队，用于生成简报。
- 识别 CrewAI 的三种失败模式：提示词膨胀、Manager LLM 额外开销、脆弱的交接机制。

## 问题所在

采用多智能体框架的团队总会撞上同一堵墙。“自主协作”在演示中听起来很棒。但当客户提交 Bug 需要确定性重放时，或者财务询问每次运行的 LLM 路由成本时，又或者值班人员需要知道凌晨 3 点是哪个智能体卡住时，这些就全都不管用了。

自由形式的 LLM 路由团队无法清晰回答上述任何问题。纯 DAG（有向无环图）能全部解决，但会失去头脑风暴类智能体所需的探索性结构。

CrewAI 的划分诚实地反映了这种权衡。团队（Crew）用于协作式、基于角色、探索性的工作。工作流（Flow）用于事件驱动、代码可控、可审计的生产环境。同一个框架，两种形态，按场景选用。

## 核心概念

### 四大原语

CrewAI 的接口很精简。记住这些，剩下的就是配置。

- **智能体（Agent）**。`role + goal + backstory + tools + (optional) llm`。背景故事（backstory）至关重要。它塑造语气、判断力以及智能体的停止时机。工具是智能体可调用的函数（详见下文）。
- **任务（Task）**。`description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出将作为输入传递。`output_pydantic` 强制结构化输出格式。
- **团队（Crew）**。容器。管理 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **流程（Process）**。执行策略。包括 Sequential（顺序）、Hierarchical（分层）、Consensus（计划中）。决定运行的结构形态。

智能体之间不直接通信。任务引用智能体。团队负责编排任务顺序。流程决定由谁来选择下一个任务。这就是整个心智模型。

> **已验证版本** CrewAI 0.86（2026-05）。新版本可能会重命名或合并流程类型；在具体依赖某种形态前，请查阅 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential（顺序）**。任务按声明顺序执行。任务 N 的输出可作为 `context` 传递给任务 N+1。成本最低。最可预测。适用于顺序固定的场景。
- **Hierarchical（分层）**。一个 Manager 智能体（独立的 LLM 调用）在专家智能体之间进行路由。CrewAI 会根据你的 `manager_llm` 配置或默认值创建该管理器。管理器每轮选择下一个任务，并可拒绝或重新路由。适用于拥有四个或以上专家智能体，且顺序真正依赖于先前输出的场景。
- **Consensus（共识）**。计划中功能，当前公共 API 尚未实现。文档保留此名称供未来基于投票的流程使用。目前请勿依赖它。

Hierarchical 会在每个专家调用之上增加一轮 LLM 调用（即管理器）。在五步运行中，Token 成本可能翻三倍。仅在确实需要路由时才为此付费。

### Crews vs Flows

这是 2026 年文档强调的核心框架。

- **Crew（团队）**。由 LLM 驱动的自主性。框架在运行时决定结构。适用于：研究、头脑风暴、初稿撰写，任何路径本身属于答案一部分的场景。难以重放。难以测试。原型开发成本低。
- **Flow（工作流）**。你完全掌控的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当其他步骤发出该主题时触发的步骤。每个步骤都是纯 Python 代码（内部可调用 Crew）。适用于：生产环境。可观测。可测试。确定性执行。

文档 2026 年的生产建议：从 Flow 开始。当自主性证明其价值时，将 Crew 作为 `Crew.kickoff()` 调用嵌入到 Flow 步骤中。Flow 提供审计轨迹，Crew 提供探索能力。组合使用，而非二选一。

### 工具集成

赋予智能体工具的三种方式。选择最简单且匹配的一种。

1. **`@tool` 装饰器**。纯函数变为工具。函数签名即 Schema；文档字符串即 LLM 可见的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类**。基于类的工具，具有显式的参数 Schema、异步支持和重试机制。适用于工具带有状态（如客户端、缓存）或需要结构化参数的场景。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包**。CrewAI 自带官方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。通过一次导入即可接入。

结构化输出使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会根据模型校验 LLM 响应，并进行强制转换或重试。配合紧凑的 `expected_output` 字符串使用效果最佳。自由文本输出适合初稿；结构化输出才是下游 Flow 能够消费的数据。

### 记忆钩子

CrewAI 开箱即支持四种记忆类型。它们可组合使用：一个 Crew 可同时启用全部四种。

> **已验证版本** CrewAI 0.86（2026-05）。近期版本将所有内容路由至统一的 `Memory` 系统，该系统封装了这四种存储。下方的概念模型依然成立，但新版中公共类接口可能收敛为单个 `Memory` 入口点；请查阅 [CrewAI memory 文档](https://docs.crewai.com/concepts/memory) 获取当前 API。

- **短期记忆（Short-term）**。单次运行内的对话缓冲区。运行结束时清除。
- **长期记忆（Long-term）**。跨运行持久化。存储在向量数据库（默认 Chroma，可替换）。通过与当前任务的相似度进行检索。
- **实体记忆（Entity）**。针对特定实体的事实记录。例如“客户 X 使用的是企业版套餐”。按实体键索引，而非按相似度。跨运行存活。
- **上下文记忆（Contextual）**。组装时检索。在智能体需要时动态拉取相关记忆，而非预加载。

在 Crew 上通过 `memory=True` 或按类型配置来启用。底层由你配置的 Embeddings 提供商支撑（默认 OpenAI，可替换为本地模型）。记忆功能是 CrewAI 相比轻量级框架的优势之一；纯 LangGraph 需要你自行接入所有这些组件。

### 何时适合使用 CrewAI

- 三到六个具有明确角色和协作工作流的智能体。适用于起草、审阅、规划、头脑风暴。
- 路由场景，其中 LLM 对下一步的判断本身就是核心价值（Hierarchical）。
- 任何团队更愿意阅读 `role + goal + backstory` 而不是图定义的场景。

### 何时不适合使用 CrewAI

- 具有严格顺序的确定性 DAG。请使用 LangGraph（课程 13）。图结构是正确的抽象；CrewAI 的角色框架在此会产生摩擦。
- 亚秒级延迟预算。Hierarchical 会增加往返调用。即使是 Sequential，也会序列化包含背景故事和先前输出的提示词。
- 单智能体循环。跳过框架；直接使用智能体循环（课程 1）加上工具注册表更简洁。

课程 17（智能体框架权衡）用矩阵详细说明了这一点。简而言之：CrewAI 位于“协作式基于角色”的象限。

### 依赖形态

独立于 LangChain。支持 Python 3.10 至 3.13。使用 `uv`。Star 数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026-05 快照）。已记录 AWS Bedrock 集成；厂商基准测试报告在 QA 负载下相比 LangGraph 有显著加速，但由于方法论（数据集、硬件、评估指标）未公开，因此请将框架对比数据仅视为方向性参考。

### 该模式的常见陷阱

- **背景故事导致的提示词膨胀**。每个智能体 2000 字的背景故事加上五个智能体的团队，会在第一次工具调用前就耗尽上下文预算。保持背景故事在 200 字以内。跨智能体重用短语；不要重复五次相同的风格描述。
- **Manager LLM 的 Token 开销**。Hierarchical 流程在每个专家调用前都会增加一次管理器 LLM 调用。对于五任务团队，这意味着六次 LLM 调用而非五次，且管理器调用携带完整任务列表及先前输出。除非路由依赖输出，否则请切换至 Sequential。
- **脆弱的交接机制**。任务 N 的 `expected_output` 是“一份大纲”。任务 N+1 将其读作 `context` 并尝试解析三个部分。但 LLM 生成了四个。下游智能体会即兴发挥。通过在任务 N 上使用 `output_pydantic` 修复，使任务 N+1 读取类型化对象而非自由文本。
- **将 Crew 直接用于生产**。未加 Flow 包装的自由形式 Crew 被部署到生产环境。输出变异性高；无法重放；值班人员无法将失败运行与成功运行进行对比。请用 Flow 进行包装。

## 动手构建

`code/main.py` 实现了两种形态的标准库版本，外加一个三智能体团队。

结构：

- `Agent`、`Task` 数据类，匹配 CrewAI 的接口表面。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行任务，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 增加一个 Manager 智能体，每轮选择下一个专家，遇到“done”时停止。
- `Flow` 配合 `@start` 和 `@listen(topic)` 装饰器、微型事件循环及追踪日志。
- `tool(name)` 装饰器，镜像 CrewAI 的 `@tool` 结构。
- `Memory` 包含 `short_term`、`long_term`、`entity` 存储；模拟相似度计算使用 numpy。

模拟的 LLM 响应是基于角色加输入前缀键控的硬编码字符串。无需网络。确定性执行。

具体演示：研究员、撰稿人、编辑团队生成关于“2026 智能体工程”的简报。研究员拉取（模拟）资料。撰稿人起草。编辑精简。同一团队通过 Flow 运行以展示确定性结构。

运行方式：

```bash
python3 code/main.py
```

追踪覆盖：顺序团队通过 `context` 传递输出；分层团队由管理器选择（研究员、撰稿人、编辑，随后“done”）；工作流以显式主题（`researched`、`drafted`、`edited`）运行相同三步；工具调用通过 `@tool` 路由；长期记忆在两次启动间存活。

Crew 的追踪是动态的；理论上管理器可以重新排序。Flow 的追踪是固定的。这一差异即是本课要点。

## 使用指南

- **CrewAI Flow** 用于生产环境。即使 Flow 仅包含一步调用 `Crew.kickoff()`。Flow 提供审计边界。
- **CrewAI Crew (Sequential)** 用于顺序明确的协作工作，尤其是初稿和审阅循环。
- **CrewAI Crew (Hierarchical)** 当路由依赖输出且拥有四个或以上专家时使用。
- **LangGraph**（课程 13）用于显式状态机、持久化恢复、严格顺序。
- **AutoGen v0.4**（课程 14）用于 Actor 模型并发与故障隔离。
- **OpenAI Agents SDK**（课程 16）用于具备交接机制与安全护栏的 OpenAI 优先产品。
- **Claude Agent SDK**（课程 17）用于具备子智能体与会话存储的 Claude 优先产品。

## 交付实践

`outputs/skill-crew-or-flow.md` 为任务选择 Crew 或 Flow，并生成最小实现脚手架。对无背景故事的 Crew、无显式主题的 Flow、少于三个专家的分层架构实施硬性拒绝。

## 避坑指南

- **背景故事并非点缀**。它会直接影响输出。为每个智能体测试三个变体；差异是真实存在的。选定一个后锁定它。
- **跳过 `expected_output`**。若无每个任务的契约，下游任务将接收 LLM 产生的任意内容。Crew 能运行，但审计会失败。
- **记忆始终开启**。每次运行都写入长期记忆。向量数据库不断膨胀。检索结果变得嘈杂。将写入范围限定在事实具有持久性的任务中。
- **管理器提示词漂移**。Hierarchical 的管理器提示词是隐式的。如果路由行为异常，请在详细模式下导出并阅读。
- **Crew 中的工具副作用**。Crew 可能会比预期更频繁地调用工具。POST、DELETE、支付操作应放在 Flow 步骤中，绝不应作为 Crew 工具。

## 练习

1. 将 Sequential 团队转换为 Flow。统计变异性下降的接触点数量。记录可读性下降的位置。
2. 为团队添加实体记忆：关于客户的事实跨启动周期持久化。验证检索是否拉取了正确的实体。
3. 实现一个 Hierarchical 流程，其中管理器在撰稿人输出至少包含三段之前，拒绝路由给编辑。追踪重试过程。
4. 接入 `BaseTool` 子类用于（模拟）网页搜索。对比追踪结构与 `@tool` 装饰器版本的差异。
5. 为编辑任务添加 `output_pydantic=Brief`，其中 `Brief` 包含 `title`、`summary`、`sections`。让撰稿人任务故意输出一次格式错误的 JSON；在追踪中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 文档简介。将玩具示例移植到真实的 `crewai` API。标准库版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（课程 24）接入实际运行。标准库版本遗漏了哪些追踪信息？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | “人格设定” | 角色 + 目标 + 背景故事 + 工具 |
| Task | “工作单元” | 描述 + 预期输出 + 负责人 + 可选的结构化输出 |
| Crew | “智能体团队” | Agent + Task + Process 的容器 |
| Process | “执行策略” | Sequential / Hierarchical / Consensus（计划中） |
| Flow | “确定性工作流” | 事件驱动、代码可控、可测试 |
| Backstory | “人格提示词” | 塑造 Agent 语气与判断力的文本 |
| `@tool` | “函数工具” | 将函数转换为 Agent 可调用的工具的装饰器 |
| `BaseTool` | “类工具” | 基于类的工具，含参数 Schema、重试、异步支持 |
| Entity memory | “按实体分类的事实” | 限定于客户/账户/问题的记忆范围 |
| Long-term memory | “跨运行记忆” | 基于向量数据库的记忆，在启动间隔间存活 |
| Contextual memory | “即时检索” | 在 Agent 需要时动态拉取的记忆 |
| Manager LLM | “路由智能体” | Hierarchical 流程中用于选择下一个任务的额外 LLM |
| `expected_output` | “任务契约” | 告知 Agent（及审计系统）返回格式的字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：核心概念与推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动结构、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文记忆
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有效，何时无效
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
