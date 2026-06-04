# 分离式 Prefill/Decode —— NVIDIA Dynamo 与 llm-d

> Prefill 是计算密集型任务；Decode 是内存带宽密集型任务。在同一块 GPU 上同时运行两者会浪费一种资源。分离式架构将它们拆分到独立的资源池中，并通过 NIXL（RDMA/InfiniBand，或回退至 TCP）在它们之间传输 KV cache。NVIDIA Dynamo（GTC 2025 发布，1.0 GA 版本）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler 与 SLA Planner 会自动调整 prefill 与 decode 的比例以匹配 SLO。NVIDIA 公布了该架构下的吞吐量提升数据——developer.nvidia.com（2025年6月）显示，在中等延迟场景下，GB200 NVL72 + Dynamo 为 DeepSeek-R1 MoE 带来了约 6 倍的吞吐提升；Dynamo 产品页（developer.nvidia.com，未注明日期）宣称在 GB300 NVL72 + Dynamo 相比 Hopper 架构下，MoE 吞吐量最高可达 50 倍。“30 倍”这一数据是社区对全栈 Blackwell + Dynamo + DeepSeek-R1 报告的汇总；我们尚未找到单一原始来源明确指出恰好为 30 倍，因此请将其视为一个方向性声明。llm-d（Red Hat + AWS）原生支持 Kubernetes：将 prefill / decode / router 作为独立 Service 部署，并针对各角色配置 HPA。llm-d 0.5 增加了分层 KV offloading、感知缓存的 LoRA routing、UCCL 网络以及 scale-to-zero 能力。经济账方面：综合多家客户披露的内部数据表明，在保持相同 SLA 的前提下，从共置服务切换至使用 Dynamo 的分离式架构，可在 200 万美元级别的推理支出中节省 30%–40%（即每年节省 60 万–80 万美元）；具体的 $2M→$600-800K 数据为内部综合估算，并非单一公开案例研究——请将其作为数量级参考锚点，而非引用文献。短提示词（<512 tokens，短输出）不值得承担传输成本。

**类型：** Learn
**语言：** Python（标准库，用于模拟分离式与共置服务的玩具程序）
**前置知识：** Phase 17 · 04（vLLM Serving 内部原理）、Phase 17 · 08（推理指标）
**预计时间：** 约 75 分钟

## 学习目标

- 解释为何 prefill 和 decode 的最优 GPU 分配比例不同，并量化共置时的资源浪费。
- 绘制分离式架构图：prefill 池、decode 池、通过 NIXL 的 KV 传输、router。
- 指出分离式架构不划算的条件（短提示词、短输出）。
- 区分 NVIDIA Dynamo（上层编排器）与 llm-d（Kubernetes 原生），并为各自匹配适用的运维场景。

## 问题所在

你在 8 块 H100 上运行 Llama 3.3 70B。在混合负载下（长提示词 + 短输出），由于大部分计算已消耗在 prefill 阶段，GPU 在 decode 阶段处于空闲状态。而在另一种负载下（短提示词 + 长输出），情况则相反。共置的 prefill + decode 意味着你必须同时为两者过度配置资源。

预算影响：20%–40% 的 GPU 时间被浪费在不匹配的资源上。你购买的是 H100 的计算能力来运行内存带宽受限的 decode，或是购买 H100 的 HBM 带宽来运行计算受限的 prefill。两者都是昂贵的浪费。

分离式架构将 prefill 和 decode 拆分到独立的资源池中，并按各自的瓶颈进行容量规划。KV cache 通过高带宽互连从 prefill 池传输至 decode 池。

## 核心概念

### 瓶颈差异的原因

**Prefill** —— 在一次前向传播中处理完整的输入提示词。矩阵乘法占主导；属于计算密集型。H100 的 FP8 可提供约 2000 TFLOPS 的有效吞吐量。批处理效率较高——一次前向传播可处理大量 token。

**Decode** —— 每次生成一个 token，每次迭代读取完整权重。属于内存带宽密集型。HBM3 提供约 3 TB/s 的带宽。仅在并发度极高时批处理效率才较好——权重读取开销可分摊到整个 batch 中。

共置运行：你需要购买同时优化了两者的 GPU。H100 两者兼顾，但无论侧重哪边成本都一样。在大规模部署时，你希望 prefill 池使用 H100（计算密集型）；decode 池使用 H200（内存密集型），或采用激进的量化策略。

### 架构设计

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输协议。在有条件时使用 RDMA/InfiniBand，否则回退至 TCP。传输延迟是真实存在的——对于 70B FP8 模型处理 4K token 提示词的 KV cache，传输延迟通常在 20–80 ms 之间。这就是为什么短提示词不值得采用分离式架构：传输开销会超过节省的成本。

### Dynamo 与 llm-d 对比

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA 版本）：
- 作为编排器，位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比例。
- 核心由 Rust 编写，支持 Python 扩展。
- 吞吐量提升：NVIDIA 报告在中等延迟场景下，GB200 NVL72 + Dynamo 为 DeepSeek-R1 MoE 带来 6 倍提升（developer.nvidia.com，2025年6月）；社区关于全栈 Blackwell + Dynamo + DeepSeek-R1 “最高 30 倍”的报告缺乏单一原始来源，应视为方向性声明。
- GB300 NVL72 + Dynamo：根据 Dynamo 产品页（developer.nvidia.com，未注明日期），相比 Hopper 架构，MoE 吞吐量最高可达 50 倍。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- 将 prefill / decode / router 作为独立的 Kubernetes Service 部署。
- 针对各角色配置 HPA，基于队列深度（prefill）/ KV 利用率（decode）信号进行扩缩容。
- `topologyConstraint packDomain: rack` 将 prefill+decode 集群打包在同一机架内，以实现高带宽 KV 传输。
- llm-d 0.5（2026年）：支持分层 KV offloading、感知缓存的 LoRA routing、UCCL 网络以及 scale-to-zero。

如果你希望获得托管的上层编排器，请选择 Dynamo。如果你需要 Kubernetes 原生组件且致力于 CNCF 生态，请选择 llm-d。

### 经济效益

内部综合估算（非单一公开案例研究——仅作数量级参考锚点）：

- 共置服务年推理支出为 200 万美元。
- 切换至使用 Dynamo 的分离式架构。
- 请求量不变，P99 延迟 SLA 保持一致。
- 报告节省金额：每年 60 万–80 万美元（减少 30%–40%）。
- 无需新增硬件。

该数据综合自多家客户披露信息，而非单一可引用的案例研究；最接近的公开数据点是 Baseten 使用 Dynamo KV routing 实现 TTFT 快 2 倍 / 吞吐量提升 61%（baseten.co，2025年10月），以及 VAST + CoreWeave 预测在 40%–60% KV 命中率下每美元可多生成 60%–130% 的 token（vastdata.com，2025年12月）。节省成本源于对各资源池的精准定容；偏重 prefill 的负载（如带有 8K+ prefix 的 RAG）比均衡负载受益更多。

### 何时不应采用分离式架构

- 提示词 < 512 tokens 且输出 < 200 tokens：传输开销将抵消收益。
- 小型集群（< 4 块 GPU）：资源池多样性不足。
- 团队无法按角色管理两个 GPU 池并进行弹性伸缩：Dynamo 虽有帮助，但并非开箱即用。
- 无 RDMA 网络：TCP 传输开销更大。

### Router 与 Phase 17 · 11 的集成

分离式 router 具备 KV cache 感知能力（见 Phase 17 · 11）。请求会优先路由至持有其 prefix 的 decode 池——若无匹配，则依次经过 prefill → decode。命中率与分离式架构的效果会叠加——cache-aware router 决定了是否真的需要执行新的 prefill。

### Blackwell 上的 MoE 才是真正体现性能的地方

GB300 NVL72 + Dynamo 相比 Hopper 基线实现了 50 倍的 MoE 吞吐量提升。MoE expert routing 在 prefill 阶段计算密集，但在 decode 阶段内存密集（需加载 expert caches），因此分离式架构能带来双重收益。2026 年的前沿模型推理将以 MoE 为主导（如 DeepSeek-V3、未来的 GPT-5 变体）。

### 需要记住的关键数据

基准测试数据会随时间变化——NVIDIA 及推理栈每季度都会发布更新结果。引用前请务必重新核对。

- GB200 NVL72 + Dynamo 上的 DeepSeek-R1：中等延迟场景下吞吐量约为基线的 6 倍（developer.nvidia.com，2025年6月）；社区关于全栈 Blackwell + Dynamo “最高 30 倍”的说法缺乏单一原始来源，仅为方向性汇总。
- GB300 NVL72 + Dynamo：相比 Hopper，MoE 吞吐量最高可达 50 倍（developer.nvidia.com，未注明日期）。
- 节省锚点（内部综合估算，非单一案例）：在恒定 SLA 下，年度 200 万美元支出可节省 60 万–80 万美元。
- 分离式适用阈值：提示词 >512 tokens 且输出 >200 tokens。
- 通过 NIXL 传输 KV cache：70B FP8 模型处理 4K 提示词的 KV 传输耗时 20–80 ms。

## 实践应用

`code/main.py` 模拟了共置与分离式服务的对比。它会报告吞吐量、单次请求成本以及提示词长度的盈亏平衡点。

## 交付成果

本课程将产出 `outputs/skill-disaggregation-decider.md`。它将根据工作负载和集群配置，判断是否应采用分离式架构。

## 练习

1. 运行 `code/main.py`。在何种提示词长度下，分离式架构的性能会优于共置架构？
2. 为一个 P99 prefix 长度为 8K、输出长度为 300 的 RAG 服务设计 prefill 池和 decode 池。
3. Dynamo 与 llm-d 选型：对于一个纯 Kubernetes 环境且无 Python 运行时偏好的团队，你会选择哪一个？
4. 计算 KV 传输成本：70B FP8 模型的 4K prefill 产生约 500 MB KV。在 RDMA 100 GB/s 下，传输耗时 5 ms；在 TCP 10 GB/s 下为 50 ms。这对你的 SLA 哪个更关键？
5. MoE expert routing 会改变 KV 访问模式。当 MoE 每个 token 激活不同的 expert 时，分离式架构的表现如何？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|------------------------|------------------------------------------|
| Disaggregated serving | “拆分 prefill/decode” | 为各阶段分配独立的 GPU 资源池 |
| NIXL | “NVIDIA 传输协议” | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | “编排器” | 面向 vLLM/SGLang/TRT-LLM 的上层协调器 |
| llm-d | “Kubernetes 原生” | Red Hat + AWS 的 K8s 分离式技术栈 |
| Planner Profiler | “Dynamo 自动配置” | 测量工作负载，配置资源池比例 |
| SLA Planner | “Dynamo 策略” | 自动调整 prefill:decode 比例以满足 SLO |
| `packDomain: rack` | “llm-d 拓扑结构” | 将 prefill+decode 打包于同一机架以实现快速 KV 传输 |
| UCCL | “统一集合通信” | llm-d 0.5 的 networking 层，支持 scale-to-zero |
| MoE expert routing | “每个 token 对应一个 expert” | DeepSeek-V3 模式；分离式架构对此有益 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
