# 面向 LLM 生产的混沌工程

> 2026 年，LLM 的混沌工程已自成体系。在生产环境运行实验前的先决条件：明确定义的 SLI/SLO、基于 trace+metric+log 的可观测性、自动回滚机制、操作手册（runbooks）以及值班制度（on-call）。架构包含四个平面：控制面（实验调度器）、目标面（服务、基础设施、数据存储）、安全面（防护机制 + 中止逻辑 + 流量过滤）、可观测性面（指标 + 追踪 + 日志），以及反馈回路（用于调整 SLO）。必须设置护栏（Guardrails）：若每日错误预算消耗速率超过预期值的 2 倍，燃烧率告警将暂停实验；通过抑制窗口和 Trace-ID 关联来去重告警噪音。节奏安排：每周进行小规模金丝雀实验并审查 SLO；每月开展故障演练（Game Day）与事后复盘；每季度进行跨团队韧性审计与依赖关系映射。LLM 专属实验：内存过载、网络故障、供应商中断、畸形提示词、KV 缓存驱逐风暴。工具链：Harness Chaos Engineering（基于 LLM 的实验推荐、爆炸半径缩减、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**类型：** 学习
**语言：** Python（标准库，简易混沌实验运行器）
**前置知识：** 第 17 阶段 · 23（AI 领域的 SRE），第 17 阶段 · 13（可观测性）
**预计时间：** 约 60 分钟

## 学习目标

- 说出混沌工程的五项先决条件（SLI/SLO、可观测性、回滚、操作手册、值班制度），并解释为何缺少任何一项都会破坏该实践。
- 绘制四个平面（控制面、目标面、安全面、可观测性面）及反馈至 SLO 的闭环图。
- 列举五项 LLM 专属实验（内存过载、网络故障、供应商中断、畸形提示词、KV 驱逐风暴）。
- 根据技术栈选择一款工具（Harness、LitmusChaos 或 Chaos Mesh）。

## 问题背景

传统技术栈中的混沌测试已相当成熟。但 LLM 技术栈引入了新的故障模式。一个包含“毒字符”的 4K token 提示词可能导致分词器停滞长达 12 秒。上游供应商返回 429 状态码；你的网关进行重试；重试放大的并发量导致你的服务因内存溢出（OOM）而崩溃。突发负载下的 KV 缓存驱逐风暴会引发重新预填充（re-prefill）级联反应，从而耗尽计算资源。

这些情况都不会出现在单元测试中。而混沌工程正是让你在用户发现问题之前提前发现它们的方法。

## 核心概念

### 先决条件

切勿在未满足以下条件时在生产环境运行混沌实验：

1. **SLI/SLO** —— 明确定义的服务级指标与目标。
2. **可观测性** —— 追踪（traces）、指标（metrics）、日志（logs），并已接入仪表盘。
3. **自动回滚** —— 参考第 17 阶段 · 20 的策略标志位回滚机制。
4. **操作手册（Runbooks）** —— 结构化文档，参考第 17 阶段 · 23。
5. **值班制度（On-call）** —— 确保有人负责响应。

缺少其中任何一项，混沌实验就会演变成真实的生产事故。

### 四大平面与反馈闭环

**控制面（Control plane）** —— 实验调度器（如 Litmus 工作流、Chaos Mesh 定时任务、Harness UI）。

**目标面（Target plane）** —— 服务、Pod、节点、负载均衡器、数据存储。

**安全面（Safety plane）** —— 紧急停止开关、抑制窗口、爆炸半径限制、错误预算闸门。

**可观测性面（Observability plane）** —— 常规指标 + Trace-ID 关联，用于区分由混沌实验引发的故障与自然故障。

**反馈闭环（Feedback loop）** —— 实验发现反哺至 SLO 调整、操作手册更新与代码修复。

### 必须设置护栏（Guardrails）

- **燃烧率告警（Burn-rate alert）**：若每日错误预算消耗速率超过预期值的 2 倍，则暂停实验。
- **抑制窗口（Suppression windows）**：在实验期间，对爆炸半径内的非实验相关告警进行静默处理。
- **Trace-ID 关联（Trace-ID correlation）**：所有由实验引发的错误均携带特定标签，以便值班人员去重告警。

### 五项 LLM 专属实验

1. **内存过载（Memory overload）** —— 通过高并发发送长上下文请求，强制触发 KV 缓存抢占风暴。观察：服务是优雅降级还是直接崩溃？
2. **网络故障（Network failure）** —— 切断推理网关与供应商之间的连接。观察：容错机制是否在 SLA 规定时间内生效？（参考第 17 阶段 · 19）
3. **供应商中断模拟（Provider outage simulation）** —— 模拟 OpenAI 100% 返回 429 状态码。观察：路由是否自动故障转移至 Anthropic？（参考第 17 阶段 · 16、19）
4. **畸形提示词（Malformed prompt）** —— 注入会导致分词器停滞的载荷（例如深度嵌套的 Unicode 字符、超大 UTF-8 码点）。观察：单个请求是否会锁死工作进程？
5. **KV 驱逐风暴（KV eviction storm）** —— 通过占满 vLLM 块预算强制触发驱逐。观察：LMCache 能否恢复，还是服务性能持续下降？

### 执行节奏

- **每周** —— 在预发环境进行小规模金丝雀实验，生产环境可逐步放量至 5%。
- **每月** —— 针对特定场景安排故障演练（Game Day）；要求跨团队参与；并进行事后复盘。
- **每季度** —— 开展跨团队韧性审计；更新依赖关系图谱。

### 工具链

- **Harness Chaos Engineering** —— 商业软件；提供基于 AI 的实验推荐；支持爆炸半径缩减；集成 MCP 工具。
- **LitmusChaos** —— CNCF 毕业项目；基于 Kubernetes 工作流。
- **Chaos Mesh** —— CNCF 沙箱项目；采用 Kubernetes 原生的 CRD 风格。
- **Gremlin** —— 商业软件；支持范围广泛。
- **AWS FIS** / **Azure Chaos Studio** —— 云厂商托管服务。

### 从小处着手

首次实验：在稳定流量下，随机终止（pod-kill）一个解码副本。观察流量重定向与服务恢复情况。如果运行平稳且安全，再进阶到网络混沌实验。

首个 LLM 专属实验：注入持续 5 分钟的单一供应商 429 故障。观察容错机制。大多数团队会发现他们的容错方案并未经过充分测试。

### 关键数据速记

- 四大平面：控制面、目标面、安全面、可观测性面。
- 燃烧率暂停阈值：每日预算消耗速率达到预期值的 2 倍。
- 执行节奏：每周金丝雀实验，每月故障演练，每季度审计。
- 五项 LLM 实验：内存、网络、供应商、畸形提示词、KV 风暴。

## 动手实践

`code/main.py` 模拟了三项带有安全面闸门的混沌实验，并报告哪些实验会触发燃烧率中止逻辑。

## 交付成果

本课程将产出 `outputs/skill-chaos-plan.md`。结合技术栈与成熟度，规划前三项实验及对应的工具链。

## 练习

1. 运行 `code/main.py`。哪项实验触发了燃烧率闸门？原因是什么？
2. 为基于 vLLM 的 RAG 服务设计前五项混沌实验。需包含成功标准。
3. 你的燃烧率告警暂停了一项实验。如何确定根本原因——是混沌实验导致的还是自然故障？
4. 辩论混沌工程应在生产环境还是仅在预发环境运行。什么情况下在生产环境运行才是正确的选择？
5. 列举三种通用网络混沌无法复现的 LLM 专属故障模式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SLI / SLO | “服务目标” | 指标与目标；必备先决条件 |
| 爆炸半径 (Blast radius) | “影响范围” | 受实验影响的服务/用户集合 |
| 燃烧率告警 (Burn-rate alert) | “预算闸门” | 当错误预算消耗速率 > 预期值 2 倍时触发 |
| 故障演练 (Game day) | “月度演习” | 计划好的跨团队混沌实验 |
| LitmusChaos | “CNCF 工作流” | CNCF 毕业的 Kubernetes 混沌工具 |
| Chaos Mesh | “CNCF CRD” | CNCF 沙箱项目的 Kubernetes 原生混沌工具 |
| Harness CE | “商业 AI 辅助” | 提供 AI 推荐的 Harness 混沌平台 |
| 畸形提示词 (Malformed prompt) | “分词器炸弹” | 导致分词停滞的输入 |
| KV 驱逐风暴 (KV eviction storm) | “抢占级联” | 大规模驱逐触发重新预填充 |

## 延伸阅读

- [DevSecOps School — 2026 混沌工程指南](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — LLM 可观测性（书籍）](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
