# LLM 的 FinOps —— 单位经济与多租户归因

> 传统的 FinOps 在 LLM 支出管理上会失效。成本基于 token 交易，而非资源运行时长。标签无法直接映射——API 调用是一次交易，而非一项资产。工程决策（prompt 设计、上下文窗口大小、输出长度）本质上是财务决策。2026 年的操作指南包含三个需在第一天就进行数据采集的归因维度：per-user（`user_id`）用于席位定价与增购，per-task（`task_id` + `route`）用于产品功能面成本与优先级排序，per-tenant（`tenant_id`）用于单位经济模型与续约。四个 token 层级——prompt、tool、memory、response——若合并为一个桶会掩盖真实支出。面向多租户产品的执行阶梯策略：按租户设置速率限制（预期峰值的 2-3 倍，返回清晰的 429 状态码及 `retry-after` 头）；每日支出上限（合同上限的 1.5-3 倍；触发时收紧速率限制并发送告警）；当支出 z-score > 4 时启用熔断开关（自动暂停服务并通知值班人员）。归因模式：打标签后聚合、遥测数据关联器（trace-ID → 账单；准确率最高）、采样与外推、基于模型的分配、事件溯源、实时流处理。核心指标：每次成功解决的查询成本、每次生成内容的成本——而非 $/M tokens。事后补打标签总会遗漏；必须在请求创建时就进行标记。

**类型：** Learn
**语言：** Python (stdlib, toy cost-attribution simulator with kill switch)
**前置知识：** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**耗时：** ~60 minutes

## 学习目标

- 解释为何传统 FinOps（标签+分层）在 LLM 支出上会失效，并列出三个新的归因维度。
- 列举四个 token 层级（prompt、tool、memory、response），并说明为何单一计费桶会掩盖成本。
- 为多租户产品设计一套执行阶梯策略（rate → spend cap → kill switch）。
- 选择核心指标（cost per resolved query / artifact），而非 $/M tokens。

## 问题所在

你的账单显示为 40,000 美元。但你并不清楚：
- 是哪个租户产生的费用。
- 是哪个产品功能导致的。
- 是否有个别用户存在滥用行为。
- 罪魁祸首是 prompt 膨胀、tool 调用还是 memory 放大。

在云服务商侧采用“打标签后聚合”的方式对云资源（EC2, S3）有效，因为标签会传递到明细行中。LLM API 调用不会自动打标签——你必须在调用点手动标记 user/task/tenant 并贯穿始终。事后补录归因总会遗漏边缘情况。

## 核心概念

### 三个归因维度

**Per-user** (`user_id`): 谁产生了多少成本。用于驱动席位定价、增购沟通，并识别高价值用户。

**Per-task** (`task_id` + `route`): 哪个产品功能面花费了多少。用于驱动功能优先级排序，以及淘汰高成本功能的决策。

**Per-tenant** (`tenant_id`): 哪个客户是盈利的。用于驱动单位经济模型分析、续约定价和层级阈值设定。

第一天就在调用点对这三个维度进行埋点采集。事后补录的效果永远更差。

### 四个 token 层级

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40% (agent workloads) |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

将这四个层级合并计费会导致优化方向盲目。请在你的归因架构中将它们拆分出来。

### 执行阶梯策略

1. **Rate limit** per tenant. 2-3x expected peak. Return 429 with `Retry-After`. Tenant sees friction; no surprise bill.

2. **Daily spend cap** per tenant. 1.5-3x contracted ceiling. Trigger: tighten rate limit + alert customer-success.

3. **Kill switch** on spend z-score > 4 relative to tenant baseline. Auto-pause tenant; page on-call; escalate to ops + CS.

### 归因模式

- **Tag-and-aggregate**: stamp metadata headers; aggregate later. Simple; rough.
- **Telemetry joiner**: join traces to billing via trace IDs. Highest accuracy. What mature teams do.
- **Sampling + extrapolation**: sample 5-10%, multiply. Cost-effective for rough spend; misses tails.
- **Model-based allocation**: regression to infer cost driver. For legacy data without tags.
- **Event-sourced**: cost as events in a stream (Kafka / Kinesis). Real-time.
- **Real-time streaming**: dashboard updates sub-second.

### “单次 X 成本”才是核心指标

$/M tokens is vendor speak. Product metrics:

- Cost per resolved support ticket.
- Cost per generated article.
- Cost per successful agent task.
- Cost per user-session-minute.

将成本与产品产出挂钩。否则优化将失去锚点。

### 成本归因的 Trace 结构

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次调用时发出。存储至数据湖。按各维度进行聚合。Phase 17 · 13 observability stack is where this lives.

### 复合节省策略栈

Stack: cache + batch + route + gateway. With all four:
- Cache L2 (Phase 17 · 14): ~10x cheaper input.
- Batch (Phase 17 · 15): 50% off.
- Route to cheap model (Phase 17 · 16): 60% cost reduction.
- Gateway efficiency (Phase 17 · 19): redundancy + retries.

Best-case stacked: ~5-10% of naive baseline. Most teams have 2-3 levers engaged; few stack all four.

### 需要记住的关键数字

- Attribution dimensions: per-user, per-task, per-tenant.
- Four token layers: prompt, tool, memory, response.
- Kill switch: spend z-score > 4.
- Unit metric: cost per resolved query, not $/M tokens.
- Stacked optimizations: ~5-10% of baseline possible.

## 动手实践

`code/main.py` simulates a multi-tenant LLM service with the three-tier enforcement ladder. Injects an abusive tenant and demonstrates the kill switch firing.

## 交付成果

This lesson produces `outputs/skill-finops-plan.md`. Given product and scale, designs the attribution schema and enforcement ladder.

## 练习

1. Run `code/main.py`. At what z-score does the kill switch fire? How do you pick the threshold?
2. Design a per-tenant, per-task cost dashboard. What are the 5 views you build first?
3. Your largest tenant is unit-economics-negative. Propose three interventions ordered by customer impact.
4. Compute cost per resolved ticket for a support product: 3M tokens/ticket, ~800 tickets/day, GPT-5 cached rate.
5. Argue whether retroactive tagging can ever work. When is it acceptable?

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Per-user attribution | "user-level cost" | `user_id` stamped on every call |
| Per-task attribution | "feature cost" | `task_id` + `route` identify product surface |
| Per-tenant attribution | "customer cost" | `tenant_id`; drives unit economics |
| Four token layers | "cost layers" | prompt + tool + memory + response |
| Rate limit | "429 guard" | Per-tenant ceiling enforced at gateway |
| Daily spend cap | "daily ceiling" | Tenant-scoped budget with alert |
| Kill switch | "auto-pause" | Spend z-score > 4 triggers auto-suspension |
| Cost per resolved | "product unit metric" | Cost tied to product outcome, not tokens |
| Telemetry joiner | "trace-to-billing" | Highest-accuracy attribution pattern |
| Stacked optimization | "cache+batch+route+gateway" | Compounding savings to ~5-10% baseline |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
