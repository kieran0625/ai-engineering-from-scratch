# Kubernetes 上的 GPU 自动扩缩容 —— Karpenter、KAI Scheduler 与 Gang Scheduling（协同调度）

> 三层架构，而非单一机制。Karpenter 动态配置节点（耗时不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 Gang Scheduling（协同调度）、拓扑感知和分层队列——它避免了“7/8 部分分配”陷阱，即七个节点因缺少一块 GPU 而空转烧钱。应用级自动扩缩容器（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理特定信号进行扩缩容——如请求队列深度、KV cache 利用率——而非 CPU/DCGM 占用率。经典的 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是一种占用率度量指标：100% 可能意味着 10 个请求或 100 个请求。vLLM 会预分配 KV cache 内存，因此内存永远不会触发缩容。本课程教你如何组合这三层架构，并避免使用默认的 Karpenter `WhenEmptyOrUnderutilized` 策略，该策略会在推理过程中断正在运行的 GPU 作业。

**类型：** 学习
**语言：** Python（标准库、玩具级队列深度自动扩缩容模拟器）
**前置知识：** Phase 17 · 02（推理平台经济学）、Phase 17 · 04（vLLM 服务内部原理）
**预计时间：** 约 75 分钟

## 学习目标

- 绘制三层自动扩缩容架构（节点配置、协同调度、应用级），并指出每层使用的工具。
- 解释为何 `DCGM_FI_DEV_GPU_UTIL` 不适合作为 vLLM 的 HPA 信号，并列出两个替代方案（队列深度、KV cache 利用率）。
- 描述 Gang Scheduling（协同调度）以及 KAI Scheduler 所防止的部分分配故障模式（8 块 GPU 中 7 块空闲）。
- 指出会导致中断正在运行 GPU 作业的 Karpenter 合并策略（`WhenEmptyOrUnderutilized`），并说明 2026 年的安全替代方案。

## 问题所在

你的团队在 Kubernetes 上部署了 LLM 推理服务。你使用 `DCGM_FI_DEV_GPU_UTIL` 作为信号配置了 HPA。服务在工作时间内利用率一直卡在 100%。HPA 从未触发扩容——因为它认为你已经满载。你手动添加了一个副本；TTFT 下降了。但 HPA 仍然没有扩容。这个信号在欺骗你。

另外，你使用 Cluster Autoscaler 管理节点。凌晨 2 点到达一个 1M token 的提示词请求；集群花费 3 分钟配置节点，导致请求超时。

再次单独来看，你部署了一个需要跨 2 个节点使用 8 块 GPU 的 70B 模型。集群有 7 块空闲 GPU，另有 1 块分布在 3 个节点上。Cluster Autoscaler 为缺失的那块 GPU 配置了一个节点。七个节点空转等待了 4 分钟烧钱，直到 Kubernetes 搞定最后一块 GPU。

三层架构，三种不同的故障模式。2026 年的 GPU 感知型自动扩缩容不是“打开 HPA”。它是将节点配置、协同调度和应用信号自动扩缩容组合在一起。

## 核心概念

### 第一层 —— 节点配置（Karpenter）

Karpenter 监控 Pending 状态的 Pod，并在约 45-60 秒内配置节点（Cluster Autoscaler 为 GPU 节点通常需 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型——如果你的 Pod 需要 8 块 H100，且集群中没有匹配的节点，Karpenter 会直接配置一个新节点，而不是扩展现有的节点组。

**合并陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 资源池非常危险。它会终止正在运行的 GPU 节点，将 Pod 迁移到更便宜、规格更合适的实例上。对于推理工作负载而言，这意味着驱逐正在处理的请求，并在新节点上重新加载 70B 模型。损失包括数分钟的容量浪费和请求失败。

GPU 资源池的安全配置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在一小时后合并真正空的节点，但绝不会驱逐正在运行的作业。

### 第二层 —— 协同调度（KAI Scheduler）

KAI Scheduler（原项目名为“Karp”，后更名）处理默认 kube-scheduler 无法完成的任务：

**Gang Scheduling（协同调度）** —— 全有或全无。一个需要 8 块 GPU 的分布式推理 Pod，要么 8 个全部同时启动，要么一个都不启动。如果没有此机制，你会陷入部分分配陷阱：8 个 Pod 中的 7 个启动，无限期等待，白白烧钱。

**拓扑感知** —— 了解哪些 GPU 共享 NVLink，哪些位于同一机架，哪些之间通过 InfiniBand 连接。据此放置 Pod。DeepSeek-V3 67B 张量并行工作负载必须保持在同一个 NVLink 域内；KAI Scheduler 会尊重这一限制。

**分层队列** —— 多个团队竞争同一个 GPU 资源池，通过优先级和配额进行管理。只有当优先级规则允许时，Team B 的训练任务才会抢占 Team A 的生产任务。

KAI 作为二级调度器与 kube-scheduler 并行部署；你只需通过注解指定工作负载使用它即可。Ray 和 vLLM 生产栈均已集成支持。

### 第三层 —— 应用级信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是一种占用率指标——它测量每个采样间隔内 GPU 是否在执行任务。100% 利用率可能意味着 10 个并发请求或 100 个；无论如何 GPU 都在忙碌。基于占用率进行扩缩容等于盲目操作。

更糟的是，vLLM 及类似引擎会预分配 KV cache 内存（最高可达 `--gpu-memory-utilization`）。即使只有一个请求，内存使用率也维持在 90% 左右。基于内存的 HPA 永远不会触发缩容。

**2026 年替代信号**：

- 队列深度（等待执行 Prefill 的请求数量）。
- KV cache 利用率（已分配给活跃序列的 Block 比例）。
- 单副本 P99 TTFT（你的 SLA 信号）。
- 有效吞吐量 Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并调整副本数量。它们完全取代了 HPA 在 LLM 推理服务中的作用。

### 何时使用何种工具

| 扩缩容决策 | 工具 |
|----------------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 作业 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级任务 | KAI Scheduler 队列 |

### 分离式 Prefill/Decode 使一切复杂化

如果你运行分离式 Prefill/Decode（Phase 17 · 17），你将拥有两类具有不同扩缩容触发条件的 Pod：Prefill Pod 基于队列深度扩缩容，Decode Pod 基于 KV cache 压力扩缩容。llm-d 将这些暴露为独立的 `Services`，并为每个角色配置单独的 HPA。切勿尝试在两者前面共用一个 HPA。

### 冷启动同样关键

冷启动缓解措施（Phase 17 · 10）是节点配置时间变得对用户可见的地方。Karpenter 的 45-60 秒预热加上 20GB 模型加载和引擎初始化，意味着从零开始的请求需要 2-5 分钟。为 SLO 关键路径保留一个热资源池（`min_workers=1`），或在应用层使用类似 Modal 的检查点机制。

### 需要记住的关键数据

- Karpenter 节点配置：约 45-60 秒，而 Cluster Autoscaler 为约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配造成的浪费——解决“7/8 陷阱”。
- 将 `DCGM_FI_DEV_GPU_UTIL` 用作 HPA 信号：无效；请使用队列深度或 KV cache 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：会终止正在运行的 GPU 作业。推理服务请使用 `WhenEmpty + consolidateAfter: 1h`。

## 实践演练

`code/main.py` 模拟了在突发型 GPU 工作负载下的三层自动扩缩容器。对比了朴素 HPA（占用率）、基于队列深度的 HPA 以及经 KAI 协同调度的扩缩容。报告未满足的请求数、GPU 空闲分钟数及综合评分。

## 交付成果

本课程将产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形态和 SLO，它将设计一份三层自动扩缩容方案。

## 练习

1. 运行 `code/main.py`。在突发型工作负载下，朴素占用率 HPA 丢弃了多少请求，而基于队列深度的 HPA 却能接住？差异从何而来？
2. 为在 H100 SXM5 上提供 Llama 3.3 70B FP8 服务的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个用于阻止非 GPU 工作负载调度到这些节点的 Taint。
3. 你的团队报告部署卡在 Pending 状态，原因是“GPU 可用但 Pod 无法调度”。请诊断——这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以证实？
4. 分别为分离式的 Prefill Pod 和 Decode Pod 选择一个自动扩缩容信号。为两者的选择提供理由。
5. 计算 `WhenEmptyOrUnderutilized` 合并陷阱对一项 24x7 生产服务的成本影响，该服务平均每天发生 60 次因 P99 TTFT > 10s 而丢弃请求的事件。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | “节点配置器” | Kubernetes 节点自动扩缩容器；亚分钟级配置 |
| Cluster Autoscaler | “旧版扩缩容器” | Kubernetes 节点自动扩缩容器前身；速度较慢，基于节点组 |
| KAI Scheduler | “GPU 调度器” | 用于 Gang Scheduling + 拓扑感知 + 队列的二级调度器 |
| Gang Scheduling（协同调度） | “全有或全无” | 原子性调度 N 个 Pod，否则全部推迟 |
| 拓扑感知 | “机架感知” | 基于 NVLink/IB/机架布局放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | “GPU 利用率” | 占用率指标；NOT LLM 的扩缩容信号 |
| 队列深度 | “等待中的请求” | 适用于 Prefill 瓶颈扩缩容的正确 HPA 信号 |
| KV cache 利用率 | “内存压力” | 适用于 Decode 瓶颈扩缩容的正确 HPA 信号 |
| 合并（Consolidation） | “Karpenter 合并” | 终止节点以切换到更便宜的实例类型 |
| `WhenEmpty + 1h` | “安全合并” | 不会驱逐正在运行的 GPU 作业的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) —— 设计文档与配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) —— 合并策略语义与 GPU 安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) —— Dynamo Planner 扩缩容信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) —— Ray 集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) —— 托管 Kubernetes 专属指南。
- [llm-d GitHub](https://github.com/llm-d/llm-d) —— Workload Variant Autoscaler 设计。
