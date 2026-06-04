# LangGraph —— Agent 的状态机

> 手写实现的 ReAct 循环是一个 `while True`。而在 LangGraph 中编写的 ReAct 循环是一个你可以进行状态检查点保存、中断、分支和时光回溯的图。Agent 本身没有变，包裹它的外围框架变了。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 11 · 09（函数调用）, Phase 11 · 14（模型上下文协议）
**耗时：** 约 75 分钟

## 问题所在

你交付了一个支持函数调用的 Agent。它运行了三个回合后出了问题：模型尝试调用一个返回 500 错误的工具，用户在任务中途改变了主意，或者 Agent 未经人工审批就决定退款。这个 `while True:` 循环没有任何钩子（hooks）。你无法暂停它，无法回退它，也无法分支出“如果模型当时选了另一个工具会怎样”的假设。一旦你把它从演示阶段推向生产，这个 Agent 就变成了一个要么成功要么失败的黑色盒子。

看清之后，下一步显而易见。Agent 本质上已经是一个状态机——系统提示词加上消息历史、待处理的工具调用以及下一步动作。将状态机显式化：为“模型思考”、“工具执行”、“人工审批”设置节点，并在它们之间通过边实现条件转换。一旦图结构变得显式，外围框架就能免费获得四项能力：检查点保存（步骤间保存状态）、中断（暂停等待人工）、流式传输（流式输出 token 和中间事件）以及时光回溯（回退到之前的状态并尝试不同的分支）。

LangGraph 正是提供这种抽象的库。它不是 LangChain 意义上的 Agent 框架（“这是 AgentExecutor，祝你好运”）。它是一个具备一等公民状态、一等公民持久化和一等公民中断能力的图运行时。Agent 循环是你画出来的，而不是手写的。

## 核心概念

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 包含三个要素。

1. **状态（State）**。一种贯穿图的类型化字典（TypedDict 或 Pydantic 模型）。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的 *reducer（归约器）* 进行合并 —— 对于需要累积的列表使用 `operator.add`，默认行为是覆盖。
2. **节点（Nodes）**。Python 函数 `state -> partial_state`。每个节点都是一个离散的步骤：“调用模型”、“运行工具”、“总结”。
3. **边（Edges）**。节点之间的转换。静态边指向固定位置。条件边接受一个路由函数 `state -> next_node_name`，以便图能根据模型输出进行分支。

你编译（compile）该图。编译操作绑定拓扑结构，附加检查点保存器（可选，但对生产环境至关重要），并返回一个可运行对象。你使用初始状态和一个 `thread_id` 来调用它。执行的每一步都会基于 `(thread_id, checkpoint_id)` 作为键持久化一个检查点。

### 四大超能力

**检查点保存（Checkpointing）**。每次节点转换都会将新状态写入存储（测试用内存存储，生产用 Postgres/Redis/SQLite）。通过使用相同的 `thread_id` 再次调用图来恢复。图会从暂停的地方继续执行。

**中断（Interrupts）**。使用 `interrupt_before=["human_review"]` 标记一个节点，执行会在该节点运行前停止。状态会被持久化。你的 API 向用户响应“等待审批”。稍后对同一个 `thread_id` 发送带有 `Command(resume=...)` 的请求即可恢复执行。

**流式传输（Streaming）**。`graph.stream(state, mode="updates")` 实时产出状态增量。`mode="messages"` 在模型节点内流式传输 LLM token。`mode="values"` 产出完整快照。由你决定在 UI 中展示哪些内容。

**时光回溯（Time-travel）**。`graph.get_state_history(thread_id)` 返回完整的检查点日志。将任意先前的 `checkpoint_id` 传入 `graph.invoke`，即可从该点分叉。非常适合调试（“如果模型当时选了工具 B 会怎样？”）以及重放生产轨迹的回归测试。

### 归约器是核心

每个状态字段都有一个归约器。大多数默认值都够用——新值会覆盖旧值。但消息列表需要 `operator.add`，以便新消息追加而不是替换。并行边通过归约器合并它们的更新。如果两个节点都更新了 `messages` 而你忘了配置 `Annotated[list, add_messages]`，第二个节点会静默获胜，导致你丢失一半的回合信息。归约器是库中唯一微妙的地方；弄懂它，其余部分自然能组合运作。

### 四节点 ReAct 图

生产级的 ReAct Agent 由四个节点和两条边组成：

1. `agent` —— 使用当前消息历史调用 LLM。返回助手消息（可能包含 tool_calls）。
2. `tools` —— 执行最后一条助手消息中的任何 tool_calls，并将工具结果作为 tool 消息追加。
3. 一条从 `agent` 出发的条件边：如果最后一条消息包含 tool_calls，则路由到 `tools`，否则路由到 `END`。
4. 一条从 `tools` 回到 `agent` 的静态边。

就是这样。你仅用大约 40 行代码就获得了完整的 ReAct 循环（思考 → 行动 → 观察 → 思考 → …），并附带检查点保存、中断和流式传输功能。

### StateGraph 与 Send（扇出）

`Send(node_name, state)` 允许节点分发并行子图。例如：Agent 决定同时查询三个检索器。每个 `Send` 都会为目标节点生成并行执行；它们的输出通过状态归约器合并。这就是 LangGraph 在不使用线程原语的情况下表达编排者-工作者模式的方式。

### 子图（Subgraphs）

编译后的图可以作为另一个图中的节点。外层图将其视为单个节点；内层图拥有自己的状态和独立的检查点。团队借此构建主管-工作者 Agent：主管图将用户意图路由到各个领域的专用工作者子图。

## 动手构建

### 步骤 1：定义状态与节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是使消息列表能够累积而非覆盖的归约器。忘记配置它是 LangGraph 最常见的 Bug。

### 步骤 2：使用线程运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每次更新都是一个字典 `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，让用户看到“Agent 正在思考… 调用 search_web… 获取结果… 正在回答。”

### 步骤 3：添加人机协同中断

标记一个节点，使其在执行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、检查点和线程都会在跨中断时持久化。除了执行期间，内存中不会保留任何数据。

### 步骤 4：利用时光回溯进行调试

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

将 `None` 作为输入传入会从给定检查点开始重放；传入一个值会将其作为对该检查点状态的更新追加，然后再恢复。这样你就可以在不重新运行整个对话的情况下复现 Agent 的失败运行。

### 步骤 5：切换至生产级检查点保存器

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

内置了 SQLite、Redis 和 Postgres。`MemorySaver` 用于测试。任何需要跨重启持久化的场景都需要真实的存储后端。

## 核心技能

> 你要把 Agent 构建为图，而不是 `while True` 循环。

在使用 LangGraph 之前，花 60 秒进行设计：

1. **命名节点**。每个离散的决策或产生副作用的操作都是一个节点。“Agent 思考”、“工具运行”、“审核员批准”、“响应流式输出”。如果你列不出它们，说明该任务还构不成 Agent 形态。
2. **声明状态**。使用最小的 TypedDict，并为每个列表字段配置归约器。不要把所有东西都塞进 `messages`；将任务特定的字段（工作 `plan`、`budget` 计数器、`retrieved_docs` 列表）提升到顶层。
3. **绘制边**。除非下一步依赖模型输出，否则使用静态边。每条条件边都需要一个带有命名分支的路由函数。
4. **提前选择检查点保存器**。测试用 `MemorySaver`，其他情况用 Postgres/Redis/SQLite。不要在没有它的情况下发布——没有检查点保存器就意味着无法恢复、无法中断、无法时光回溯。
5. **在工具运行前决定中断，而不是运行后**。审批应放在进入产生副作用节点的边上，以便在造成损害前取消；验证应放在模型输出的边上，以便低成本拒绝错误调用。
6. **默认启用流式传输**。UI 使用 `mode="updates"`，模型节点内的 token 级流式传输使用 `mode="messages"`，评估期间的完整快照使用 `mode="values"`。

拒绝发布没有检查点保存器的 LangGraph Agent。拒绝发布在副作用发生后才中断的 Agent。拒绝发布没有将 `add_messages` 作为归约器的 `messages` 字段。

## 练习

1. **简单**。使用计算器工具和网页搜索工具实现上述四节点 ReAct 图。验证 `list(app.get_state_history(config))` 在一次两回合的对话中至少返回四个检查点。
2. **中等**。添加一个 `planner` 节点，它在 `agent` 之前运行，并将结构化的 `plan: list[str]` 写入状态。让 `agent` 将计划步骤标记为已完成。如果在检查点恢复过程中丢失了 `plan`（归约器配置错误），则测试失败。
3. **困难**。构建一个主管图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图都有独立的状态和检查点保存器。在外层图上添加一个 `interrupt_before=["writer"]`，以便人工可以批准研究简报。确认从先前检查点进行时光回溯时，仅重放分叉出的分支。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| StateGraph | “LangGraph 的图” | 编译前用于添加节点和边的构建器对象。 |
| Reducer | “字段如何合并” | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认为覆盖，`add_messages` 用于追加。 |
| Thread | “会话 ID” | 限定单次会话所有检查点范围的 `thread_id` 字符串。 |
| Checkpoint | “暂停的状态” | 节点转换后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| Interrupt | “暂停等待人工” | `interrupt_before` / `interrupt_after` 在节点边界停止执行；使用 `Command(resume=...)` 恢复。 |
| Time-travel | “从先前步骤分叉” | `graph.invoke(None, config_with_old_checkpoint_id)` 从该检查点开始向前重放。 |
| Send | “并行子图分发” | 节点可返回的一个构造器，用于生成目标节点的 N 个并行执行实例。 |
| Subgraph | “作为节点的编译图” | 用作另一图中节点的已编译 StateGraph；保留其独立的状态作用域。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— StateGraph、归约器、检查点保存器和中断的权威参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) —— 本课程使用的思维模型，直接源自官方源码。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) —— 关于 Postgres/SQLite/Redis 存储、检查点命名空间和线程 ID 的详细说明。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) —— `interrupt_before`、`interrupt_after`、`Command(resume=...)` 以及编辑状态模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个 LangGraph Agent 都实现的模式；阅读它以理解推理轨迹的原理。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 何时优先选择哪种图结构（链式、路由、编排者-工作者、评估者-优化者）。
- Phase 11 · 09（函数调用）—— 每个 LangGraph Agent 节点复用的工具调用原语。
- Phase 11 · 14（模型上下文协议）—— 外部工具发现机制，通过 MCP 适配器接入 LangGraph `ToolNode`。
- Phase 11 · 17（Agent 框架权衡）—— 何时在 CrewAI、AutoGen 或 Agno 之外选择 LangGraph。
