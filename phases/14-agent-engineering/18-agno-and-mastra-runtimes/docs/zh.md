# Agno 与 Mastra：生产级运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产级运行时组合。Agno 专注于微秒级智能体实例化以及无状态 FastAPI 后端。Mastra 基于 Vercel AI SDK 底层，提供智能体、工具、工作流、统一模型路由和复合存储功能。

**类型：** 学习
**语言：** Python、TypeScript
**前置知识：** Phase 14 · 01（Agent Loop），Phase 14 · 13（LangGraph）
**预计时间：** 约 45 分钟

## 学习目标

- 识别 Agno 的性能目标及其适用场景。
- 说出 Mastra 的三个核心原语——Agents（智能体）、Tools（工具）、Workflows（工作流）——以及支持的服务器适配器。
- 解释为何无状态会话级 FastAPI 后端是推荐的 Agno 生产部署路径。
- 根据技术栈（优先 Python 还是优先 TypeScript）选择 Agno 或 Mastra。

## 问题背景

LangGraph、AutoGen、CrewAI 等框架较为臃肿。那些希望“在我的运行时中快速实现基础智能体循环”的团队会转向 Agno（Python）或 Mastra（TypeScript）。两者都牺牲了部分框架内置的原语，以换取更高的原始速度和与周边技术栈更紧密的契合度。

## 核心概念

### Agno

- Python 运行时，前身为 Phi-data。
- “没有图结构、链式调用或复杂的模式——只有纯粹的 Python。”
- 官方文档中的性能目标：智能体实例化约 2μs，每个智能体内存占用约 3.75 KiB，支持约 23 个模型提供商。
- 生产路径：无状态会话级 FastAPI 后端。每次请求启动一个全新的智能体；会话状态存储在数据库中。
- 原生支持多模态（文本、图像、音频、视频、文件）及智能体驱动的 RAG。

当你每秒需要处理数千个短生命周期智能体时（例如聊天并发接入、评估流水线），这些速度目标至关重要。但如果单个智能体运行长达 10 分钟，则影响较小。

### Mastra

- TypeScript，基于 Vercel AI SDK 构建。
- 三大核心原语：**Agents**（智能体）、**Tools**（工具，使用 Zod 类型定义）、**Workflows**（工作流）。
- 统一模型路由器——覆盖 94 家提供商的 3,300 多个模型（截至 2026 年 3 月）。
- 复合存储：将记忆、工作流、可观测性数据分别持久化到不同的后端；大规模可观测性推荐使用 ClickHouse。
- 采用 Apache 2.0 许可证，但包含 `ee/` 目录，该目录受源代码可用企业许可证约束。
- 提供 Express、Hono、Fastify、Koa 服务器适配器；对 Next.js 和 Astro 提供一等公民级别的支持。
- 内置 Mastra Studio（本地访问 localhost:4111）用于调试。
- 在 1.0 版本发布时（2026 年 1 月），已获得 22k+ GitHub Stars 和每周 300k+ npm 下载量。

### 定位对比

两者均不试图取代 LangGraph。它们的竞争维度在于：

- **语言契合度。** Agno 适合优先使用 Python 的团队；Mastra 适合优先使用 TypeScript 的团队。
- **运行时体验。** Agno = 近乎零开销；Mastra = 深度集成于 Vercel 生态。
- **可观测性。** 两者均可集成 Langfuse/Phoenix/Opik（第 24 课），但 Mastra Studio 是官方原生提供的。

### 选型建议

- **Agno** —— Python 后端，大量短生命周期智能体，对性能要求高，技术栈为 FastAPI。
- **Mastra** —— TypeScript 后端，部署于 Next.js / Vercel，需要统一的多提供商模型路由，使用 Zod 类型定义工具。
- **LangGraph**（第 13 课）—— 当持久化状态和显式的图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** —— 当你希望直接使用厂商产品化形态的智能体 SDK 时（第 16–17 课）。

### 常见误区

- **为性能而性能。** 仅仅因为“2μs”听起来很酷就选择 Agno，但实际上你的负载是每次请求仅调用一次缓慢的智能体。此时系统开销并非瓶颈。
- **生态绑定。** Mastra 带有 Vercel 特色的集成在 Vercel 上是优势，在其他平台上则可能成为劣势。
- **企业许可证混淆。** Mastra 的 `ee/` 目录属于源代码可用协议，而非 Apache 2.0。如果计划 Fork 代码，请务必仔细阅读相关许可证。

## 动手实践

本课主要侧重于对比分析——没有任何单一代码示例能完全公平地展示两个框架的优势。请参阅 `code/main.py` 获取一个并列对照的玩具示例：一个最小化的“运行智能体、流式输出结果、持久化会话”流程被实现了两次（一次采用 Agno 风格，一次采用 Mastra 风格）。

运行方式：

```
python3 code/main.py
```

两种结构不同但功能等效的执行轨迹。

## 应用场景

- **Agno** —— 需要高性能且采用 FastAPI 架构的 Python 后端。
- **Mastra** —— 涉及多家模型提供商并需要工作流原语的 TypeScript 后端。
- 两者均提供官方可观测性钩子，且均可与 Langfuse 集成。

## 部署上线

`outputs/skill-runtime-picker.md` 会根据技术栈、延迟预算和运维形态，指导你选择 Agno、Mastra、LangGraph 或厂商 SDK。

## 练习

1. 阅读 Agno 官方文档。将标准库实现的 ReAct 循环（第 01 课）移植到 Agno。哪些特性消失了？哪些保留了下来？
2. 阅读 Mastra 官方文档。将相同的循环移植到 Mastra。工具的类型定义发生了哪些变化（Zod 与无类型）？
3. 基准测试：在你的技术栈上测量智能体实例化延迟。Agno 宣称的 2μs 对你的实际负载有意义吗？
4. 设计迁移方案：如果你之前在 Python 中使用 CrewAI，迁移到 Agno 会导致哪些功能失效？
5. 阅读 Mastra 的 `ee/` 许可证条款。其中哪些限制会影响开源 Fork 项目？

## 核心术语

| 术语 | 通常的说法 | 实际含义 |
|------|------------|----------|
| Agno | “高速 Python 智能体” | 无状态会话级智能体运行时 |
| Mastra | “基于 Vercel AI SDK 的 TypeScript 智能体” | 智能体 + 工具 + 工作流 + 模型路由器 |
| Unified Model Router | “多提供商接入” | 通过单一客户端访问 94 家提供商的 3,300+ 模型 |
| Composite storage | “多后端支持” | 记忆/工作流/可观测性数据分别存储至不同后端 |
| Mastra Studio | “本地调试器” | 用于探查智能体的 localhost:4111 Web UI |
| Source-available | “非开源” | 允许查看源码，但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework 官方文档](https://www.agno.com/agent-framework) —— 性能目标、FastAPI 集成指南
- [Mastra 官方文档](https://mastra.ai/docs) —— 核心原语、服务器适配器、模型路由器
- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) —— 面向有状态图架构的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) —— Mastra 集成文档中引用的可观测性对比参考
