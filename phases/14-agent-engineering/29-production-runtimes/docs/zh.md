# 生产运行时：队列、事件与定时任务

> 生产型智能体运行于六种运行时形态之上：请求-响应、流式传输、持久化执行、基于队列的后台处理、事件驱动和定时调度。在选择框架之前，请先确定运行时形态。可观测性在所有形态中都是核心支撑（承重级）。

**类型：** 学习
**语言：** Python (标准库)
**前置知识：** 第 14 阶段 · 课程 13 (LangGraph)，第 14 阶段 · 课程 22 (语音)
**预计时间：** 约 60 分钟

## 学习目标

- 说出六种生产运行时形态，并将每种形态与对应的框架/产品模式相匹配。
- 解释为什么持久化执行（LangGraph）对长周期任务至关重要。
- 描述事件驱动运行时，并说明何时适合使用 Claude Managed Agents。
- 解释为何对于多步骤智能体而言，可观测性是核心支撑。

## 问题所在

生产环境中的智能体会以 Jupyter Notebook 无法暴露的方式失败：第 37 步网络超时、语音通话中途用户挂断、机器重启时 cron 任务崩溃、后台工作进程内存耗尽。运行时形态决定了哪些故障是可以被容忍或恢复的。

## 核心概念

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30秒）。
- 技术栈：Agno (Python + FastAPI)、Mastra (TypeScript + Express/Hono/Fastify/Koa)。
- 可观测性：标准 HTTP 访问日志 + OTel span。

### 流式传输

- 使用 SSE 或 WebSocket 进行渐进式输出。
- LiveKit 将其扩展至 WebRTC 以支持语音/视频（课程 22）。
- 技术栈：任何支持流式的框架 + 能够处理 SSE/WS 的前端。
- 可观测性：每个 chunk 的耗时、首 token 延迟、尾部延迟。

### 持久化执行

- 每一步后保存状态检查点；失败时自动恢复。
- AutoGen v0.4 的 Actor 模型将故障隔离到单个智能体（课程 14）。
- LangGraph 的核心差异化特性（课程 13）。
- 当步骤数未知且恢复成本高昂时必不可少。

### 基于队列 / 后台处理

- 任务进入队列，由工作进程领取，结果通过 webhook 或 pub/sub 回传。
- 对长周期智能体至关重要（根据 Anthropic 的 computer use 公告，每个任务包含数十到数百个步骤）。
- 技术栈：Celery (Python)、BullMQ (Node)、SQS + Lambda (AWS)、自定义方案。
- 可观测性：队列深度、单任务延迟分布、DLQ 大小。

### 事件驱动

- 智能体订阅触发器：新邮件、PR 创建、cron 触发。
- Claude Managed Agents 开箱即用地支持此模式（课程 17）。
- CrewAI Flows（课程 15）用于构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动的延迟、智能体处理延迟。

### 定时调度

- 类似 cron 的智能体，定期运行。
- 结合持久化执行，使失败的夜间任务能在下一个调度周期自动恢复。
- 技术栈：Kubernetes CronJob + 持久化框架；托管服务（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动的生产环境。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入集成。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音服务（课程 22）。
- **Claude Managed Agents** 用于托管的长周期异步任务。

### 可观测性是核心支撑

如果没有 OpenTelemetry GenAI span（课程 23）以及 Langfuse/Phoenix/Opik 后端（课程 24），你将无法调试在第 40 步失败的多步骤智能体。在生产环境中这不是可选项。它决定了你是“快速调试”还是“带着更多日志从头重放”。

### 生产运行时常见的故障点

- **形态选择错误。** 为 5 分钟的任务选择请求-响应模式。用户会挂断；工作进程堆积；重试次数叠加。
- **缺乏 DLQ。** 没有死信队列的队列工作进程。失败的任务会直接消失。
- **后台工作不透明。** 后台智能体运行但未导出追踪数据。除非用户上报，否则故障不可见。
- **跳过持久化状态。** 任何超过 30 秒且无法承受重启开销的运行都需要持久化执行。

## 动手实现

`code/main.py` 是一个基于标准库的多形态演示项目：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带有 DLQ 的基于队列的工作进程。
- 事件触发器注册表。
- 定时调度器。

运行方式：

```bash
python3 code/main.py
```

输出：五条追踪记录，展示同一任务在不同形态下的行为表现。智能体逻辑相同，外层包装不同。持久化执行（第六种形态）有意安排在课程 13 中结合 LangGraph 检查点进行讲解。

## 应用场景

- **请求-响应** 用于类聊天交互体验。
- **流式传输** 用于渐进式响应。
- **持久化执行** 用于长周期任务。
- **队列处理** 用于批量 / 异步 / 长耗时任务。
- **事件驱动** 用于智能体响应外部事件。
- **定时调度** 用于日常维护（内存整合、评估测试、成本报告）。

## 交付实践

`outputs/skill-runtime-shape.md` 会为特定任务选择一种运行时形态，并配置相应的可观测性要求。

## 练习

1. 将你课程 01 中的 ReAct 循环移植到你的技术栈中的所有六种形态中。哪种形态适合哪种产品交互面？
2. 为基于队列的演示添加 DLQ。模拟 10% 的任务失败率；监控并展示 DLQ 的大小。
3. 编写一个由 cron 触发的评估智能体，每晚针对当天的 Top 20 追踪记录运行评估。
4. 实现带背压（backpressure）的流式传输：如果客户端处理缓慢，则暂停智能体。这与轮次预算（turn budget）如何相互作用？
5. 阅读 Claude Managed Agents 文档。在什么情况下你会将自托管的长周期智能体迁移至托管服务？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| 请求-响应 | “同步” | 用户等待；仅适用于短任务 |
| 流式传输 | “SSE / WS” | 渐进式输出；提升用户体验；可按 chunk 观测延迟 |
| 持久化执行 | “从故障恢复” | 状态已打快照；从最后一步断点续跑 |
| 基于队列 | “后台任务” | 生产者 / 工作进程池 / DLQ |
| 事件驱动 | “基于触发器” | 智能体对外部事件做出响应 |
| DLQ | “死信队列” | 失败任务的暂存区 |
| Claude Managed Agents | “托管框架” | Anthropic 托管的长周期异步服务，具备缓存与上下文压缩能力 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久化执行详情
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长周期异步服务
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — “每个任务数十到数百个步骤”
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor 模型故障隔离
