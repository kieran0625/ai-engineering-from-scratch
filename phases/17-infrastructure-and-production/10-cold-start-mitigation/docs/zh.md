# 无服务器大语言模型的冷启动缓解策略

> 一个 20 GB 的模型镜像从冷启动到提供服务需要 5-10 分钟（7B 参数）到 20 分钟以上（70B 参数）。在真正的无服务器环境中，这根本不是预热——而是服务中断。缓解措施涵盖五个层次：预置节点镜像（AWS 上的 Bottlerocket、双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal 检查点，重启速度提升高达 10 倍）、热池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 流水线，延迟降低 10-200 倍），以及仅迁移输入 token（KB）而非 KV cache（GB）的热迁移技术。Modal 将 2-4s 的冷启动时间作为基准；Baseten 默认为 5-10s，配合预热可低于 1s。本课程将教你如何测量、预算并组合使用这五层缓解策略。

**类型：** 学习
**语言：** Python（标准库、简易冷启动路径模拟器）
**前置知识：** 第 17 阶段 · 02（推理平台经济学）、第 17 阶段 · 03（GPU 自动扩缩容）
**预计时间：** 约 60 分钟

## 学习目标

- 列举冷启动缓解的五层架构，并说出每一层的一个工具或模式。
- 计算 70B 模型总冷启动时间，即（节点供应）+（权重下载）+（权重加载至 HBM）+（引擎初始化）之和。
- 解释为何热迁移传输的是输入 token（KB）而非 KV cache（GB），以及其代价是什么（重新计算）。
- 说明热池的权衡（支付空闲 GPU 费用或接受冷启动长尾延迟），以及 `min_workers > 0` 成为强制要求的 SLA 阈值。

## 问题所在

你的无服务器 LLM 端点在夜间缩容至零。早上 8 点流量激增。首个请求需等待以下过程完成：

1. Karpenter 供应 GPU 节点：45-60s。
2. 容器拉取包含权重的 30 GB 镜像：120-300s。
3. 引擎将权重加载至 HBM：根据模型大小和存储速度不同，需 45-120s。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache pool、tokenizer：10-30s。

总计：220-510s（约 3-8 分钟）后才能返回第一个 token。你的 SLA 是 2s。你部署了一个热池（`min_workers=1`），问题似乎消失了——但现在你需要为 1 个空闲 GPU 支付 24x7 的费用。如果你的服务有 5 个产品，每个产品维持 1 个热副本，那么无论是否有用户调用，每月都将消耗 5 × 24 × 30 = 3,600 GPU-hours/month。

冷启动缓解策略的目标是在保持无服务器经济性的同时，逼近始终在线服务的延迟表现。

## 核心概念

### 第一层 —— 预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。对已预拉取容器镜像的数据卷进行快照；在你的 `EC2NodeClass` 中引用该快照 ID。新节点启动时，权重已位于本地 NVMe 上——步骤 2 和部分步骤 3 被消除。原生支持与 Karpenter 配合使用。典型节省：大型模型每次冷启动可节省 2-4 分钟。

GCP 等效方案：使用预构建容器层的自定义 VM 镜像。Azure 等效方案：采用相同模式的 managed disk snapshots。

### 第二层 —— 模型流式加载（Run:ai Model Streamer）

无需在响应首个请求前加载完整文件，而是逐层将权重流式传输至 GPU 内存，并在首个 transformer block 驻留后立即开始处理。NVIDIA Run:ai Model Streamer 已原生集成于 vLLM 2026 版本。兼容 S3、GCS 和本地 NVMe。通过将 I/O 与 compute setup 重叠，可将大型模型的权重加载时间缩短约一半。

### 第三层 —— GPU 内存快照（Modal）

Modal 会在首次加载后对 GPU 状态（weights、CUDA graphs、KV cache region）进行检查点保存。后续重启时直接反序列化至 HBM——比重新初始化快 10 倍。这最接近“2 秒内启动热 GPU”的效果。权衡点：快照与特定 GPU topology 绑定，因此如果 Karpenter 将你迁移至不同 SKU，则需要重新生成 checkpoint。

### 第四层 —— 热池（min_workers=1）

最简单的缓解方案：始终保持一个 replica 处于就绪状态。成本为单个 GPU 的小时费率 × 24x7。对于小模型而言这笔账很残酷（支付 $0.85-$1.50/hr 以避免 30s 的冷启动），而对于大模型则较为友好（支付 $4/hr 以避免 5 分钟的冷启动）。热池成为强制要求的 SLA 阈值通常为：70B+ 模型的 TTFT P99 < 60s。

### 第五层 —— 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为层级结构：NVMe（速度快但容量大）、DRAM（中等且分层）、HBM（容量极小但即时访问）。权重预先加载至 DRAM；按需加载至 HBM。论文报告显示，相较于直接将磁盘加载至 HBM 的朴素做法，冷启动延迟可降低 10-200 倍。生产环境采用尚处早期，但已有与 vLLM 的集成方案。

### 第六层 —— 热迁移（附加模式）

当节点不可用时（如 spot eviction、node drain），传统模式是冷启动另一个 replica 并排空请求队列。热迁移将输入 token（kilobytes 级）移至已加载模型的目的地，并在目的地重新计算 KV cache。重新计算的成本远低于通过网络传输 GB 级的 KV cache。适用于 disaggregated deployments。

### 热池成本核算

对于 P99 TTFT SLA 为 2s 的服务，问题不在于“是否使用热池”，而在于“需要多少热副本，以及哪些路径分配它们”。

- 高价值交互路径（live chat、voice agent）：`min_workers=1-2`。
- 后台批处理路径（夜间分类任务）：接受 scale-to-zero，可容忍 5-10 分钟的冷启动。
- 高级订阅层：`min_workers`，为每个 tenant 提供 dedicated capacity。

### 优化前的度量

全新节点上 70B 模型的冷启动解剖（示例数据）：

| 阶段 | 耗时 | 缓解措施 |
|-------|------|-----------|
| 节点供应 | 50s | Bottlerocket + pre-seeded image、warm pool |
| 镜像拉取 | 180s | Pre-seeded data volume（消除） |
| 权重加载至 HBM | 75s | Model streamer（减半）；GPU snapshot（消除） |
| 引擎初始化 | 20s | Persistent CUDA graph cache |
| 首次前向传播 | 3s | Min inherent latency |
| **冷启动总计** | **328s** | |
| **缓解后总计** | **~15s** | 降低 22 倍 |

### 关键数据速记

- Modal 冷启动：2-4s（配合 GPU snapshots）。
- Baseten 默认冷启动：5-10s；配合 pre-warming 可低于 1s。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载提速约 2 倍。
- ServerlessLLM 分层加载：延迟降低 10-200 倍（论文数据）。

## 实践应用

`code/main.py` 模拟了开启与关闭各项缓解措施的冷启动路径。报告总冷启动时间、warm-pool 成本，以及 warm pool 成本得以收回的 break-even request rate。

## 交付成果

本课程将产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和 traffic shape，它将选择应叠加使用的缓解措施。

## 练习

1. 运行 `code/main.py`。计算 break-even request rate：当请求速率高于此值时，维持热副本的成本将低于因冷启动导致额外请求失败（违反 SLO）所付出的代价。
2. 你部署了一个 13B 模型，P99 TTFT SLA 为 3s。请选择能达到该目标的最小缓解栈（最少层数）。
3. Bottlerocket pre-seeding 消除了 image pull 时间，但权重仍需从 snapshot 加载至 HBM。若 snapshot-backed NVMe 读取速率为 7 GB/s，请计算 70B 模型的 wall-clock 时间。
4. 你的无服务器提供商提供 GPU snapshots（Modal），但团队以“snapshots 会泄露 PII”为由拒绝使用。请论证双方观点——实际风险是什么？缓解措施有哪些（ephemeral snapshots、encryption、namespace isolation）？
5. 设计分层 warm-pool 策略：付费用户、试用用户和 batch workloads 分别需要多少个热副本？请展示计算过程。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Cold start | “长时间停顿” | 全新 replica 从接收到请求到返回首个 token 的时间 |
| Warm pool | “始终运行的最小配置” | `min_workers >= 1`，用于确保至少有一个 replica 处于就绪状态 |
| Pre-seeded image | “预烘焙 AMI” | 容器 weights 已预置的 node image |
| Bottlerocket | “AWS 节点操作系统” | 支持 dual-volume snapshot 的 AWS container-optimized OS |
| Model streamer | “流式加载” | 将 weights I/O 与 compute setup 重叠执行 |
| GPU snapshot | “检查点至 HBM” | 序列化加载后的 GPU state；重启时反序列化 |
| Tiered loading | “NVMe + DRAM + HBM” | 存储层级结构；按需加载 |
| Live migration | “迁移 token” | 传输输入数据（KB 级），在目的地重新计算 KV cache |
| `min_workers` | “热副本” | 无服务器环境的最小 keep-alive count |
| Scale-to-zero | “完全无服务器” | 空闲时无成本；接受完整的冷启动代价 |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 发布的 benchmark 与 checkpoint 架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot 模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将 weights load 与 compute setup 重叠。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — pre-warming playbook。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — tiered loading 设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — disaggregated deployments 的热迁移方案。
