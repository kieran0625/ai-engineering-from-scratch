# 综合项目 06 — Kubernetes DevOps 故障排查代理

> AWS 的 DevOps 代理已正式商用（GA），Resolve AI 发布了其 K8s 运维手册，NeuBird 演示了语义监控，Metoro 将 AI SRE 与按服务的 SLO 绑定。生产环境的形态已定型：告警 Webhook 触发后，代理读取遥测数据，遍历 K8s 对象图，对根因假设进行排序，并发布带有审批按钮的 Slack 简报。默认只读。所有修复操作均需人工审批。本综合项目即构建该代理，并在 20 个合成事件上进行评估，同时与 AWS 代理在三个共享案例中进行对比。

**类型：** 综合项目
**语言：** Python（代理）、TypeScript（Slack 集成）
**前置要求：** 第 11 阶段（LLM 工程）、第 13 阶段（工具与 MCP）、第 14 阶段（代理）、第 15 阶段（自主系统）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**耗时：** 30 小时

## 问题背景

2025-2026 年的 SRE 叙事已演变为：“AI 代理负责事件分诊，人类负责审批修复操作。” AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 均已在生产环境中交付此形态。代理读取 Prometheus 指标、Loki 日志、Tempo 链路追踪、kube-state-metrics 以及 K8s 对象的知识图谱。它能在五分钟内生成带遥测引用的排序根因假设。未经 Slack 明确的人工审批，它绝不会执行破坏性命令。

大部分难点在于范围界定与安全控制，而非推理本身。代理需要具备默认只读的 RBAC 接口、加固的 MCP 工具服务器，以及记录每个“考虑过”与“实际执行”命令的审计日志。它需要知道何时超出自身能力范围并及时升级上报。此外，它的运行成本必须足够低，以免 OOM-kill 级联反应导致产生高达 5000 美元的代理账单。

## 核心概念

代理基于知识图谱运行。节点包括 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）以及遥测源（Prometheus 时序数据、Loki 日志流、Tempo 链路）。边编码了所有权关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus 时序数据）。图谱通过 kube-state-metrics 同步保持新鲜，并在每次告警触发时重新采样。

告警触发后，代理从受影响的对象开始进行根因分析。它沿边遍历，拉取相关的遥测切片（最近 15 分钟），并起草假设。假设根据证据进行排序：支持该假设的遥测引用数量、时效性以及具体程度。前三大假设将发送至 Slack，附带图谱路径可视化图表以及修复操作的审批按钮。

修复操作受控于审批门控。默认允许的动作为只读。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子所需的认证令牌由代理永远不持有。审计日志记录代理*考虑过*的每一个命令——而不仅仅是已执行的——以便审查流程能捕获险些发生的错误操作。

## 架构设计

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性数据源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管版）或 kuzu（嵌入式），存储 K8s 对象及遥测边
- 代理：LangGraph，配备每工具白名单，默认只读
- 工具传输：基于 StreamableHTTP 的 FastMCP；为破坏性工具提供独立服务器，置于审批门控之后
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复机制：ArgoCD 回滚 Webhook、PagerDuty 升级通知、Slack 审批卡片
- 审计：仅追加结构化日志（记录考虑、执行、审批、结果）
- 部署：K8s Deployment，配备独立的窄权限 RBAC 角色；使用独立命名空间

## 构建步骤

1. **图谱摄入**。每 30 秒将 kube-state-metrics 同步至 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边：OBSERVED_BY（一个 Pod 被某个 Prometheus 时序数据观测）。

2. **告警接收器**。FastAPI 端点，接受 PagerDuty 或 Alertmanager Webhook。提取受影响对象及 SLO 违规情况。

3. **只读工具接口**。通过 FastMCP 封装 kubectl、Prometheus 查询、Loki LogQL、Tempo TraceQL。每个工具对应狭窄的 RBAC 动词（“get”、“list”、“describe”）。默认服务器不包含“delete”、“exec”、“scale”。

4. **根因代理**。LangGraph 包含三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图谱中的相邻对象，`hypothesize` 起草带遥测引用的排序根因候选项。

5. **证据评分**。每个假设的得分 = 时效性 × 具体性 × 图谱路径长度倒数 × 引用数量。返回前三名。

6. **Slack 简报**。发布包含假设、图谱路径可视化（服务端渲染的子图图片）以及最多一项修复操作审批按钮的附件。

7. **修复门控**。破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，受审批令牌保护。仅在 Slack 卡片经人工批准后，代理方可调用它们。

8. **审计日志**。仅追加 JSONL：针对每个候选命令，记录是否被考虑、是否被执行、谁批准的。每日同步至 S3。

9. **合成事件套件**。构建 20 种场景：OOMKill 级联、DNS 抖动、HPA 震荡、PVC 写满、邻居噪音干扰、有缺陷的 Sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取退避等。在根因准确率与假设生成时间上对代理进行评分。

## 使用方法

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付标准

`outputs/skill-devops-agent.md` 为最终交付物。给定一个 K8s 集群和告警源，代理将生成排序的根因假设以及受 Slack 管控的修复流程。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件下的 RCA 准确率 | 20 个合成事件中根因正确率 ≥80% |
| 20 | 安全性 | 审计日志中，破坏性操作守卫绝不在无 Slack 审批的情况下触发 |
| 20 | 假设生成时间 | 从告警到 Slack 简报的 p50 耗时低于 5 分钟 |
| 20 | 可解释性 | 每个假设均包含图谱路径与遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端可用 |
| **100** | | |

## 练习

1. 在你的代理上运行 AWS DevOps Agent 演示所用的相同三个事件。发布对比结果。报告代理出现分歧的地方。

2. 添加“险些出错”审计功能，标记代理*考虑过*但若无审批就会造成破坏的命令。统计一周内的险些出错率。

3. 将假设模型从 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 RCA 准确率的变化量及单次事件的成本。

4. 构建因果过滤器：区分相关遥测尖峰与真正的根因。在 20 个场景标签上训练一个小型分类器。

5. 添加回滚预演：使用相同的清单对暂存集群执行 ArgoCD 回滚。在点击 Slack 审批按钮前，在实际集群中验证回滚计划。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 | “集群图” | 节点 = K8s 对象 + 遥测时序；边 = 所有权、调度、观测 |
| 默认只读 | “受限 RBAC” | 代理的服务账号仅拥有 get/list/describe 动词；破坏性动词位于独立服务器且受审批控制 |
| 审计日志 | “考虑过 vs 已执行” | 仅追加记录每个候选命令、是否运行、谁批准 |
| 假设排序 | “证据评分” | 时效性 × 具体性 × 图谱路径长度倒数 × 引用数量 |
| Slack 审批卡片 | “人机协同门控 (HITL)” | 带有修复按钮的交互式 Slack 消息；代理需等待人工点击后才能继续 |
| 遥测引用 | “证据指针” | 支持某项声明的 Prometheus 查询、Loki 选择器或 Tempo 链路 URL |
| MTTR | “解决时间” | 从告警触发到 SLO 恢复的实际耗时 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) —— 2026 年权威参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) —— 竞品参考
- [NeuBird 语义监控](https://www.neubird.ai) —— 语义图谱方案
- [Metoro AI SRE](https://metoro.io) —— 以 SLO 为首的生产环境框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) —— 集群状态数据源
- [LangGraph](https://langchain-ai.github.io/langgraph/) —— 参考代理编排器
- [FastMCP](https://github.com/jlowin/fastmcp) —— Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) —— 受控修复目标
