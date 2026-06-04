# AutoGen v0.4：Actor 模型与 Agent 框架

> AutoGen v0.4（微软研究院，2025年1月）围绕 Actor 模型重新设计了智能体编排。支持异步消息交换、事件驱动型智能体、故障隔离与自然并发。该框架现已进入维护模式，而 Microsoft Agent Framework（2025年10月公开预览版）将成为其继任者。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 · 01（Agent Loop）、Phase 14 · 12（Workflow Patterns）
**耗时：** 约 75 分钟

## 学习目标

- 描述 Actor 模型：将智能体视为 Actor，消息作为唯一的进程间通信（IPC）方式，每个 Actor 具备故障隔离能力。
- 列举 AutoGen v0.4 的三个 API 层级——Core、AgentChat、Extensions，并说明各自的用途。
- 解释为何将消息投递与处理解耦能够实现故障隔离与自然并发。
- 使用 Python 标准库实现一个 Actor 运行时，并将双智能体代码审查流程移植到该运行时上。

## 问题背景

大多数智能体框架是同步的：一个智能体生产，另一个在调用栈中消费。故障会导致整个调用栈崩溃。并发能力是后期附加的。分布式部署则需要重写代码。

AutoGen v0.4 的解决方案：Actor 模型。每个智能体都是一个拥有私有收件箱（inbox）的 Actor。消息是唯一的交互方式。运行时将消息投递与处理解耦。故障被限制在单个 Actor 内。并发是原生支持的。分布式部署只需更换传输层即可。

## 核心概念

### Actor

一个 Actor 包含：

- 私有状态（外部无法直接访问）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中可执行的操作包括“回复”、“向其他 Actor 发送消息”、“派生新 Actor”、“更新状态”、“终止自身”。

两个 Actor 之间不能共享内存。它们只能通过发送消息进行交互。

### AutoGen v0.4 的三大 API 层级

1. **Core。** 底层 Actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。支持异步消息交换与事件驱动。
2. **AgentChat。** 任务驱动的高级 API（用于替代 v0.2 中的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成模块——涵盖 OpenAI、Anthropic、Azure、工具调用及记忆功能。

### 为何解耦至关重要

在 v0.2 模型中，同步调用 `agent_a.chat(agent_b)` 会阻塞 agent_a，直到 agent_b 返回。而在 v0.4 中，`send(agent_b, msg)` 仅将消息放入 agent_b 的收件箱并立即返回。后续投递由运行时负责。这带来三个关键影响：

- **故障隔离。** Agent B 崩溃不会导致 Agent A 崩溃——运行时会捕获 B 处理器中的故障，并决定后续处理方式（记录日志、重试或转入死信队列）。
- **自然并发。** 可同时存在大量处于飞行状态的消息；多个 Actor 可并发处理各自的收件箱。
- **开箱即用的分布式支持。** 无论 Actor 位于同一进程还是不同主机，收件箱加传输层的抽象完全一致。

### 拓扑结构

- **RoundRobinGroupChat。** 智能体按固定顺序轮流发言。
- **SelectorGroupChat。** 选择器智能体根据对话上下文决定下一位发言者。
- **Magentic-One。** 面向网页浏览、代码执行、文件处理的参考级多智能体团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息都会生成一个 Span；工具调用会附带 `gen_ai.*` 属性，遵循 2026 年 OTel GenAI 语义规范（第 23 课）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 已稳定，适用于研究与原型开发。微软已将活跃开发重心转移至 Microsoft Agent Framework（2025年10月1日公开预览版；计划于 2026 年第一季度末发布 1.0 GA 正式版）。AutoGen 的设计模式可平滑迁移——Actor 模型是经得起时间考验的核心思想。

## 动手实现

`code/main.py` 实现了基于标准库的 Actor 运行时：

- `Message` —— 携带 `sender`、`recipient`、`topic`、`body` 的类型化负载。
- `Actor` —— 定义抽象接口与 `receive(message, runtime)`。
- `Runtime` —— 包含共享队列、消息投递与故障隔离的事件循环。
- 双 Actor 演示：`ReviewerAgent` 负责代码审查，`ChecklistAgent` 执行检查清单；双方通过消息交换直至达成共识。

运行方式：

```
python3 code/main.py
```

运行追踪日志展示了消息投递过程、某一 Actor 的模拟故障（未导致另一 Actor 崩溃），以及最终收敛于统一结论的过程。

## 实际应用

- **AutoGen v0.4/v0.7**（维护模式）—— 适用于研究、原型开发及多智能体模式验证。
- **Microsoft Agent Framework**（公开预览版）—— 未来演进方向；沿用相同的 Actor 模型理念，但采用全新 API。
- **LangGraph swarm topology**（第 13 课）—— 通过共享工具交接实现类似模式。
- **自定义 Actor 运行时** —— 当需要特定传输协议时（如 NATS、RabbitMQ、gRPC）。

## 交付部署

`outputs/skill-actor-runtime.md` 可为指定的多智能体任务生成最小化 Actor 运行时，并附带团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列（Dead-Letter Queue）：当处理器抛出异常时，将失败消息暂存以供人工检查。在你的玩具项目中，DLQ 触发的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器 Actor 根据对话状态决定由谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为基于 HTTP 的 JSON 服务器，使 Actor 可在独立进程中运行。
4. 为每条消息接入 OTel Span（或使用空操作占位符）。按照第 23 课的要求输出 `gen_ai.agent.name` 与 `gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具项目移植到真实的 `autogen_core` API。在生产环境中，你跳过了哪些关键细节？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Actor | “Agent（智能体）” | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | “Event（事件）” | 类型化负载；Actor 交互的唯一方式 |
| Inbox | “Mailbox（邮箱）” | 每个 Actor 专属的待处理消息队列 |
| Runtime | “Agent host（智能体宿主）” | 负责路由消息并隔离故障的事件循环 |
| Topic | “Channel（通道）” | Actor 之间命名的发布/订阅路由 |
| Fault isolation | “Let it crash（随它崩溃）” | 单个 Actor 故障不会导致其他 Actor 崩溃 |
| RoundRobinGroupChat | “Fixed-rotation team（固定轮换团队）” | 智能体按顺序轮流处理 |
| SelectorGroupChat | “Context-routed team（上下文路由团队）” | 选择器决定下一位处理者 |
| Magentic-One | “Reference team（参考团队）” | 面向网页、代码与文件的多人智能体小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 架构重设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 基于图结构的替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— AutoGen 默认生成的 Span 规范
