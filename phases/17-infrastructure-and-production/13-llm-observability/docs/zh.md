# LLM 可观测性栈选型

> 2026 年的可观测性市场主要分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估（evals）、提示词管理、会话回放捆绑在一起。网关/插桩工具（Helicone、SigNoz、OpenLLMetry、Phoenix）则专注于遥测数据。Langfuse 核心采用 MIT 许可证，开源生态平衡良好（云端免费额度为每月 5 万事件）。Phoenix 基于 Elastic License 2.0，原生支持 OpenTelemetry —— 非常适合漂移/RAG 可视化，但并非持久化的生产后端。Arize AX 采用零拷贝 Iceberg/Parquet 集成方案，声称比单体式可观测性方案便宜 100 倍。LangSmith 在 LangChain/LangGraph 领域领先，定价为 $39/用户/月，仅企业版支持自托管。Helicone 基于代理模式，配置仅需 15-30 分钟，免费额度为每月 10 万次请求，但在智能体追踪方面深度较浅。常见的生产模式是：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 进行粘合。

**类型：** 学习
**语言：** Python（标准库、简易追踪采样模拟器）
**前置知识：** 第 17 阶段 · 08（推理指标）、第 14 阶段（智能体工程）
**耗时：** 约 60 分钟

## 学习目标

- 区分开发平台（捆绑功能：评估 + 提示词 + 会话）与网关/遥测工具（仅提供追踪 + 指标）。
- 将六大主流工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可证、定价及最佳适用场景。
- 解释如何通过 OpenTelemetry 粘合模式将网关工具与独立的评估平台结合使用。
- 指出 2026 年的成本差异点（Arize AX 的零拷贝方案 vs 单体式数据摄入），并说明大约 100 倍的倍数关系。

## 问题背景

你上线了一个 LLM 功能。它能正常工作。但你完全无法感知提示词失败、工具循环、延迟回退、成本飙升或提示词缓存命中率等问题。你在 Google 上搜索“LLM 可观测性”，结果出来八个工具都声称能以三种不同的价格解决同一个问题。

它们解决的并不是同一个问题。LangSmith 回答的是“为什么这个 LangGraph 运行会失败？”；Phoenix 回答的是“我的 RAG 管道是否发生了漂移？”；Helicone 回答的是“哪个应用正在疯狂消耗 Token？”；Langfuse 回答的是“我能否将整个系统自托管？”不同的工具，面向不同的受众。

选型涉及四个维度：技术栈（LangChain？原始 SDK？多供应商混合？）、许可证容忍度（仅限 MIT？接受 Elastic？商业许可也可？）、预算（免费额度？$100/月？$1000/月？）以及自托管需求（必须？可有可无？绝不？）。

## 核心概念

### 两大类别

**开发平台** 将可观测性与评估、提示词管理、数据集版本控制、会话回放等功能捆绑。你可以运行实验，查看哪个提示词效果更好，并将新提示词针对旧优胜者进行数据集回归测试。代表工具：LangSmith、Langfuse、Comet Opik。

**网关/遥测工具** 对推理调用进行插桩（instrumentation）——记录提示词、响应、Token 数、延迟、模型、成本等。设计极简。可通过 OpenTelemetry 与独立的评估工具组合使用。

### Langfuse —— 开源与商业的平衡

- 核心采用 Apache/MIT 双许可证；可通过 Docker 自托管。
- 云端免费额度：每月 5 万事件。付费版：团队版 $29/月。
- 提供评估、提示词管理、追踪、数据集功能。合理覆盖了开发平台的四大特性。
- 最佳适用场景：你需要 LangSmith 级别的功能，但必须自托管或坚持使用开源许可证。

### Phoenix (Arize) —— 遥测优先，原生支持 OpenTelemetry

- 采用 Elastic License 2.0；自托管非常简单。
- 在 RAG 和漂移可视化方面表现优异。嵌入空间散点图作为一等公民功能直接内置。
- 并非设计用于持久化生产后端——主要面向开发阶段的观测。
- 最佳适用场景：RAG 管道开发、漂移调试，在生产环境中需搭配独立的网关使用。

### Arize AX —— 面向大规模的方案

- 商业软件。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称在大规模下比单体式可观测性方案（如 Datadog 级别）便宜约 100 倍。原理在于：你将追踪数据以 Parquet 格式存储在自己的 S3 上；Arize 直接读取。
- 最佳适用场景：日追踪量 >1000 万，已有数据湖基础设施，希望获得 LLM 专属仪表盘且不想承担 Datadog 的高昂费用。

### LangSmith —— 优先服务于 LangChain/LangGraph

- 商业软件，$39/用户/月。仅企业版支持自托管。
- 在 LangChain 和 LangGraph 技术栈中处于行业领先地位。如果你未使用这两者，其吸引力会大幅下降。
- 最佳适用场景：团队已深度绑定 LangChain，且愿意为此付费。

### Helicone —— 基于代理的最小可行方案

- 只需将你的 `OPENAI_API_BASE` 替换为 Helicone 代理，即可在 15-30 分钟内完成配置。
- MIT 许可证；免费额度为每月 10 万次请求，付费版 $20/月起。
- 包含故障转移、缓存、速率限制等功能——同时充当网关。
- 在智能体/多步骤追踪方面的深度较浅。
- 最佳适用场景：快速启动、单一技术栈应用，需要网关与可观测性二合一。

### Opik (Comet) —— 开源开发平台

- Apache 2.0 许可证，完全开源。
- 功能集与 Langfuse 类似，带有 Comet 的技术底蕴。
- 最佳适用场景：ML 团队已在使用 Comet，希望在同一界面内实现 LLM 可观测性。

### SigNoz —— 以 OpenTelemetry 为核心的全栈 APM

- Apache 2.0 许可证。通过 OpenTelemetry 处理通用 APM 及 LLM 相关指标。
- 最佳适用场景：需要在服务与 LLM 调用之间实现统一的可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 于 2025 年底发布了 GenAI 语义约定（`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`）。能够消费 OTel 数据的工具可以相互协作。目前涌现的生产模式如下：

1. 从每次 LLM 调用中生成带有 GenAI 约定的 OTel 数据。
2. 路由至网关（Helicone / Portkey）用于日常运维。
3. 双写至评估平台（Phoenix / Langfuse）用于回归测试。
4. 归档至数据湖（Iceberg），以便后续通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误的层级进行插桩

在智能体框架内部进行插桩（例如添加 LangSmith 追踪）会将你与该框架强耦合。而在 HTTP/OpenAI-SDK 层进行插桩（通过 OpenLLMetry 或你的网关）则具备可移植性。

### 采样策略 —— 你无法保留所有数据

当日请求量超过 100 万时，全量保留追踪的成本将高于 LLM 调用本身。应按规则进行采样：100% 错误请求、100% 高成本请求、5% 成功请求。始终保留聚合数据；仅对长尾数据保留原始追踪。

### 需要记住的关键数字

- Langfuse 云端免费版：每月 5 万事件。
- LangSmith：$39/用户/月。
- Helicone 免费版：每月 10 万次请求。
- Arize AX 宣称：在大规模下比单体方案便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 动手实践

`code/main.py` 模拟了在不同保留策略（100% 全量摄入、采样、采样+错误过滤）下的一天 100 万追踪数据。它会报告每种策略下的存储成本以及丢失的数据情况。

## 交付成果

本课程将产出 `outputs/skill-observability-stack.md`。根据技术栈、规模、预算和许可证策略，为你挑选合适的工具。

## 练习

1. 你的团队基于 LangChain，希望使用开源自托管的可观测性方案。请选择 Langfuse 或 Opik 并给出理由。
2. 假设日追踪量为 500 万，Datadog 报价为 $15 万/月，请计算 Arize AX 的盈亏平衡点。
3. 设计一套 OpenTelemetry GenAI 属性集，规定组织指南应在每次 LLM 调用中强制采集。
4. 论证仅靠 Phoenix 是否足以满足生产环境需求。在什么情况下它不够用？
5. Helicone 会产生 20ms 的代理开销。在 P99 TTFT 为 300 ms 的情况下，这是否可以接受？如果 SLA 要求为 100 ms 呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OpenLLMetry | “LLM 版的 OTel” | 面向 LLM 的开源 OpenTelemetry 插桩库 |
| GenAI conventions | “OTel 属性” | LLM 调用的标准 OTel 属性命名规范 |
| LangSmith | “LangChain 可观测性” | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | “开源版 LangSmith” | 采用 MIT 许可证，功能集相似 |
| Phoenix | “Arize 开发工具” | 原生支持 OpenTelemetry 的开发/评估平台 |
| Arize AX | “大规模可观测性” | 商业级零拷贝 Iceberg/Parquet 可观测性方案 |
| Helicone | “代理型可观测性” | 收集 LLM 遥测数据并提供网关功能的 HTTP 代理 |
| Opik | “Comet LLM” | Comet 出品的 Apache 2.0 开源开发平台 |
| Session replay | “追踪重放” | 重播包含工具调用的完整智能体会话 |
| Eval | “离线测试” | 在标注数据集上运行候选模型/提示词进行评估 |

## 延伸阅读

- [SigNoz — 2026 年顶级 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 配置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 官方文档](https://docs.arize.com/phoenix)
- [Helicone 官方文档](https://docs.helicone.ai/)
