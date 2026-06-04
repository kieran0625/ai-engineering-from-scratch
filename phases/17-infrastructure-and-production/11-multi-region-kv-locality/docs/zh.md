# 多区域 LLM 推理服务与 KV Cache 局部性

> 轮询负载均衡对缓存型 LLM 推理具有明显的破坏性。未能命中持有其前缀的节点的请求，将承担完整的预填充（prefill）成本——在长提示词下，P50 延迟约为 800 ms，而缓存命中时仅需约 80 ms。2026 年的生产环境模式是采用感知缓存的路由器（基于 Rust 的 vLLM Router、llm-d router），它们消费 KV-cache 事件并根据前缀哈希匹配进行路由。最新的研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业“跨区域推理”方案（如 Bedrock 跨区域推理、GKE 多集群网关）将推理视为黑盒——它们处理的是可用性，而非首字延迟（TTFT）。摩根大通和梅奥诊所于 2024 年 11 月进行了 us-east-1 故障转移演练，耗时约 22 分钟。DR（灾难恢复）的现实情况是：32% 的 LLM DR 失败是因为团队备份了模型权重，却忘记了 tokenizer 文件或量化配置。

**类型：** 学习
**语言：** Python（标准库，用于模拟前缀缓存感知路由器的玩具示例）
**前置知识：** 阶段 17 · 04（vLLM 推理服务）、阶段 17 · 06（SGLang RadixAttention）
**预计时间：** 约 60 分钟

## 学习目标

- 解释轮询负载均衡为何会破坏缓存推理，并量化 TTFT 惩罚。
- 绘制感知缓存路由器的架构图：输入（KV-cache 事件）、算法（前缀哈希匹配）、决胜条件（GPU 利用率）。
- 指出导致 LLM 32% DR 失败的原因（缺失 tokenizer 文件/量化配置），并列出三份文件的 DR 检查清单。
- 区分商业跨区域方案（Bedrock CRI、GKE 多集群网关）与 KV 感知路由。

## 问题所在

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前端放置了一个采用轮询策略的 ALB。生产环境中的前缀缓存命中率降至 8%。TTFT P50 翻了三倍。你的 vLLM 日志显示每个请求都在支付完整的预填充成本。

轮询对无状态服务是最优的。LLM 推理从设计上是状态相关的——KV cache 编码了模型见过的所有内容。盲目路由等于将请求导向错误的缓存。

此外，你的团队有一份 DR 计划。你将模型权重跨区备份到了 S3。某区域发生故障；你尝试故障转移；副本拒绝启动。你忘了 tokenizer.json、量化配置和 RoPE scaling 配置存放在一个未同步的独立 Bucket 中。

多区域 LLM 推理服务是一个缓存问题、路由问题和 DR 规范实践问题——而不是负载均衡器问题。

## 核心概念

### 感知缓存路由

请求携带提示词到达。路由器对前缀进行哈希计算（例如前 512 个 token）；它向每个副本询问“你是否缓存了这个前缀？”。副本在分配和驱逐块时，会在发布/订阅通道上发布 KV-cache 事件。路由器选择匹配的副本，如果无人匹配，则回退到基于 GPU 利用率的决胜条件。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希到副本索引的映射，通过 O(1) 查找进行路由。无匹配时回退到队列深度最小的节点。

**llm-d router**：相同模式，原生支持 Kubernetes。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（阶段 17 · 06）是副本内部的等效实现。跨副本路由严格属于上游逻辑。

### 关键数据

在 2K token 提示词、Llama 3.3 70B FP8、H100 上的 TTFT P50：
- 缓存命中（同一副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷预填充）：约 800 ms。

差距达 10 倍。如果你的路由器在跨副本场景下能命中 60-80% 的前缀缓存，你就能以 N 个副本的容量逼近单副本的性能。如果仅命中 10%，则近似于朴素扩展。

### 跨区域面临新的约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由器将来自 us-east-1 的请求路由到 ap-southeast-1 的热前缀，节省的预填充时间（800 → 80 ms）将被 440 ms 的往返延迟完全掩盖。GORGO（2026 年研究）明确指出这一点——需联合最小化 `prefill_time + network_latency`，而非仅优化预填充。通常的解决方案是保持区域内路由，除非面对巨大的多 MB 级前缀且预填充占主导时才跨区域。

### 商业“跨区域推理”在此场景下并无帮助

AWS Bedrock 跨区域推理会在容量压力期间自动将请求路由到其他区域。它优化的是可用性而非 TTFT，并将推理视为黑盒。GKE 多集群网关同理——属于服务级故障转移，不感知 KV cache。

即使使用这些方案，你仍然需要一个应用层的感知缓存路由器。它们处理的是“us-east-1 彻底宕机”的情况。感知缓存路由处理的是 TTFT 问题。

### DR 规范实践——32% 的文件缺失问题

广泛引用的 2026 年统计数据：32% 的 LLM DR 失败是因为团队备份了权重，却忘记了：
- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE scaling、attention masks、chat templates）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA adapter manifests）

解决方案是最小化的三份文件 DR 清单：
1. HF 模型仓库下的所有文件（权重 + 配置 + tokenizer）。
2. 引擎特定的推理配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定文件）。

补充：每季度进行一次 DR 演练。摩根大通 2024 年 11 月的 us-east-1 演练之所以能在 22 分钟内完成恢复，仅仅是因为演练剧本经过了反复排练。

### 数据驻留性是独立约束

欧盟客户的 PHI（个人健康信息）不得离开欧盟。如果你的感知缓存路由器为了前缀匹配，将源自巴黎的请求发送到 us-east-1，无论 TTFT 收益如何，你都违反了 GDPR。在优化缓存之前，必须按数据驻留边界划分路由器。

### 你需要记住的数据

- 缓存命中与未命中的 TTFT 差距：约 10 倍（2K 提示词下 80 ms vs 800 ms）。
- 美欧区域间 RTT：约 75 ms。
- DR 失败原因：32% 缺失 tokenizer/量化配置。
- 摩根大通 us-east-1 故障转移（2024 年 11 月）：22 分钟（SLA 为 30 分钟）。

## 动手实践

`code/main.py` 在多区域工作负载上模拟三种路由策略（轮询、区域感知缓存、全局感知缓存）。报告缓存命中率、TTFT P50/P99 以及跨区域流量费用。

## 交付成果

本课程产出 `outputs/skill-multi-region-router.md`。给定区域、数据驻留约束和 SLA，设计一份路由方案。

## 练习

1. 运行 `code/main.py`。在 75 ms RTT 条件下，提示词长度达到多少时，跨区域路由会优于纯本地路由？
2. 你的缓存命中率从 70% 骤降至 12%。诊断三种可能的原因，并说明确认每种原因的观测指标。
3. 为 vLLM 服务的 70B AWQ 量化模型（含 5 个 LoRA adapter）设计一份 DR 清单。列出所有文件和配置。
4. 论证对于具有严格 TTFT SLO 的金融科技公司而言，Bedrock 跨区域推理是否“足够”。引用具体行为进行说明。
5. 一条源自巴黎的请求匹配到了 us-east-1 中的前缀。你会路由它吗？写出相应的策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Cache-aware routing | “智能负载均衡” | 根据前缀哈希匹配，将请求路由至持有 KV cache 的副本 |
| KV-cache events | “缓存发布/订阅” | 副本发布块的添加/驱逐事件；路由器建立索引 |
| Prefix hash | “缓存键” | 前 N 个 token 的哈希值，用作路由器查找依据 |
| GORGO | “跨区域路由研究” | arXiv 2602.11688；将网络延迟作为显式项 |
| Cross-region inference | “Bedrock CRI” | AWS 产品；可用性故障转移，非 TTFT 感知 |
| DR manifest | “备份清单” | 恢复所需的所有文件——不仅限于权重 |
| Data residency | “GDPR 边界” | 限制哪些区域可访问用户数据的法律约束 |
| RTT | “往返时间” | 网络延迟；美欧约 75 ms，美亚约 220 ms |
| LLM-aware LB | “缓存命中负载均衡” | 将感知缓存路由器作为一个产品类别 |

## 延伸阅读

- [BentoML — 多云与跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 结合网络延迟项的跨区域 KV-cache 复用。
- [TianPan — 多区域 LLM 推理服务缓存局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock 跨区域推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM 生产栈路由器](https://github.com/vllm-project/production-stack) — 感知缓存路由器源码。
