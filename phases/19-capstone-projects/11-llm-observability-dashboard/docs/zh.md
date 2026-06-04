# 毕业设计 11 — LLM 可观测性与评估仪表盘

> Langfuse 转为开源核心模式。Arize Phoenix 发布了 2026 年 GenAI 语义约定映射。Helicone 和 Braintrust 均加倍投入于按用户分摊成本的归因分析。Traceloop 的 OpenLLMetry 已成为事实上的 SDK 自动插桩方案。生产环境的标准架构为：使用 ClickHouse 存储追踪数据，Postgres 存储元数据，Next.js 构建前端界面，并运行一批基于采样追踪数据的评估作业（DeepEval、RAGAS、LLM-judge）。请构建一个自托管版本，接入至少四个 SDK 家族的数据，并演示如何在五分钟之内捕获注入的回归缺陷。

**类型：** 毕业设计
**语言：** TypeScript（UI）、Python / TypeScript（数据接入 + 评估）、SQL（ClickHouse）
**前置要求：** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P11 · P13 · P17 · P18
**耗时：** 25 小时

## 问题描述

2026 年所有在生产环境中运行流量的 AI 团队，都会在模型旁维护一套可观测性平面。涵盖成本归因、幻觉检测、漂移监控、越狱信号识别、SLO 仪表盘以及 PII 泄露告警。开源参考项目——Langfuse、Phoenix、OpenLLMetry——均已收敛至 OpenTelemetry GenAI 语义约定作为数据接入规范。你现在可以使用单一 SDK 对 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 进行插桩，并导出兼容的 Span 数据。

你将构建一个自托管仪表盘，接入至少四个 SDK 家族的数据，在采样追踪数据上运行一组评估作业，检测漂移并发出告警。衡量标准：针对故意注入的回归缺陷（例如某个提示词开始产生 PII 数据），仪表盘需在五分钟之内捕获该问题并触发告警。

## 核心概念

数据接入采用 OTLP HTTP 协议。SDK 生成符合 GenAI 语义约定的 Span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 数据落入 ClickHouse 用于列式分析；元数据（用户、会话、应用）落入 Postgres。

评估作业以批处理形式运行在采样追踪数据之上。DeepEval 负责评分忠实度、毒性和答案相关性。当追踪数据携带检索上下文时，RAGAS 负责评分检索指标。自定义 LLM-judge 运行领域特定检查（PII 泄露、偏离策略的响应）。评估结果会写回同一个 ClickHouse，作为与父级追踪关联的评估 Span。

漂移检测通过观察嵌入空间分布随时间的变化（计算提示词嵌入的 PSI 或 KL 散度）以及评估分数趋势来实现。告警数据推送至 Prometheus Alertmanager，随后转发至 Slack / PagerDuty。UI 基于 Next.js 15 搭配 Recharts 构建。

## 架构设计

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 数据接入：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输协议
- 收集器：OpenTelemetry Collector，配备尾采样处理器（用于成本控制）
- 存储：ClickHouse 存储 Span，Postgres 存储元数据，S3 存储原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估包、自定义 LLM-judge
- 漂移检测：每周计算池化提示词嵌入的 PSI / KL 散度（sentence-transformers）
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + Server Actions
- 开箱即用的支持 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建步骤

1. **收集器配置。** 配置 OpenTelemetry Collector，包含 OTLP HTTP 接收器、尾采样器（保留 100% 的错误追踪和 10% 的成功追踪），以及指向 ClickHouse 和 S3 的导出器。

2. **ClickHouse 表结构。** 创建表 `spans`，列名镜像 GenAI 语义约定：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，另加一个用于长负载的 JSON 字段。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖测试。** 编写一个小型客户端应用，分别使用各 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）配合 OpenLLMetry 自动插桩。验证每个 SDK 是否都能生成规范的 GenAI Span 并成功落入 ClickHouse。

4. **评估作业。** 创建一个定时作业，读取最近 15 分钟的采样追踪数据，运行 DeepEval 的忠实度、毒性和答案相关性评估。输出结果为与父级追踪关联的评估 Span。

5. **自定义 LLM-judge。** 实现一个 PII 泄露检测器：接收响应内容后，调用守卫 LLM 评估 PII 泄露的可能性。高分响应将进入人工分拣队列。

6. **漂移检测。** 每周作业计算本周池化提示词嵌入与过去四周基线之间的 PSI 值。若 PSI 超过阈值，则触发告警。

7. **仪表盘开发。** 基于 Next.js 15 构建页面：概览（Span/sec、每用户成本、p95 延迟）、追踪（搜索 + 瀑布流视图）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链路。** Prometheus 导出器读取评估分数聚合值和延迟百分位数；Alertmanager 将警告路由至 Slack，严重违规路由至 PagerDuty。

9. **回归探针。** 注入缺陷：被评估的聊天机器人有 1% 的概率开始泄露伪造的 SSN。测量 MTTR：从缺陷部署到 Slack 告警触发的时间。

## 使用方式

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付验收

`outputs/skill-llm-observability.md` 是最终交付物。给定一个 LLM 应用，仪表盘需接入其追踪数据、运行评估、对漂移发出告警，并在 Next.js 中展示按用户划分的成本明细。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 追踪 Schema 覆盖率 | 生成规范 GenAI Span 的 SDK 家族数量（目标：6+） |
| 20 | 评估准确性 | DeepEval / RAGAS 评分与人工标注集的对比 |
| 20 | 仪表盘 UX | 注入回归缺陷的 MTTR（目标：低于 5 分钟） |
| 20 | 成本 / 规模 | 持续以 1k spans/sec 速率接入且无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端验证 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证规范的 Span 是否落入 ClickHouse，且 `gen_ai.*` 属性是否完整准确。

2. 在同一组追踪数据上将 DeepEval 替换为 Phoenix 评估器。测量两个评估引擎之间的分数漂移情况。

3. 优化漂移检测器：改为按 app-id 计算 PSI 而非全局计算。展示各应用的漂移轨迹。

4. 新增“用户影响”页面：展示每用户成本与每用户失败率，并附带迷你图（sparklines）。

5. 构建一种尾采样策略：保留所有毒性 > 0.5 的追踪，并对其余部分进行 10% 的分层采样。测量由此引入的采样偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | “OTel LLM 属性” | 2025 年 OpenTelemetry 关于 LLM Span 属性的规范（系统、模型、Token 数等） |
| Tail sampling | “尾部采样” | 收集器在追踪完成后决定保留或丢弃该追踪（可预先查看错误信息） |
| PSI | “群体稳定性指数” | 比较两个分布的漂移指标；> 0.2 通常意味着存在显著漂移 |
| LLM-judge | “模型即评估” | 使用 LLM 根据评分标准对另一个 LLM 的输出进行打分（忠实度、毒性、PII） |
| Tail-sampling policy | “保留规则” | 决定哪些追踪需要持久化、哪些需要丢弃的规则；通常结合错误状态与采样率 |
| Eval span | “关联评估追踪” | 携带评估分数的子 Span，链接至原始 LLM 调用 Span |
| Cost per user | “单位经济模型” | 在指定时间窗口内归因到 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考性的开源核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 提供强大漂移支持的替代参考方案
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 家族
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 数据接入规范
- [Helicone](https://www.helicone.ai) — 替代型托管可观测性服务
- [Braintrust](https://www.braintrust.dev) — 替代型以评估为先的平台
- [ClickHouse documentation](https://clickhouse.com/docs) — 列式 Span 存储文档
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估库
