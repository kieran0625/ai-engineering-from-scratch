# 托管式大语言模型平台 —— Bedrock、Vertex AI、Azure OpenAI

> 三大云厂商，三种截然不同的策略。AWS Bedrock 是一个模型市场——通过单一 API 提供 Claude、Llama、Titan、Stability 和 Cohere 等服务。Azure OpenAI 是独家 OpenAI 合作方案，并提供预留吞吐量单元（PTU）以保障专属容量。Vertex AI 则以 Gemini 为核心，在长上下文和多模态方面表现最佳。根据 2026 年 Artificial Analysis 的测量数据，在等效的 Llama 3.1 405B 模型上，Azure OpenAI 的中位延迟约为 50 ms，而 Bedrock 约为 75 ms——PTU 解释了这一差距，因为专属容量在性能上优于共享按需实例。决策原则不是“哪个最快”，而是“哪个模型目录和 FinOps 成本归因体系最匹配我的产品需求”。本课程将教你基于明确权衡做出选择，而非凭直觉。

**类型：** 学习
**语言：** Python（标准库，简易成本与延迟对比器）
**前置要求：** 第 11 阶段（LLM 工程），第 13 阶段（工具与协议）
**耗时：** 约 60 分钟

## 学习目标

- 说出三种平台策略（市场模式 vs 独家合作 vs Gemini 优先），并将每种策略匹配到相应的产品用例。
- 解释在 Azure OpenAI 中预留吞吐量单元（PTU）能带来什么优势，以及为何在 405B 规模下，按需调用的 Bedrock 通常慢约 25 ms。
- 为每个平台绘制 FinOps 成本归因体系图（Bedrock 应用推理配置档 vs Vertex 按团队划分项目 vs Azure 作用域 + PTU 预留）。
- 制定一份“双供应商最低要求”政策，并解释为何在 2026 年，单一供应商锁定是一项代价高昂的错误。

## 问题背景

你为你的产品选择了 Claude 3.7 Sonnet。现在你需要部署它来提供服务。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者走网关。直接 API 最简单；Bedrock 增加了业务伙伴协议（BAA）、VPC 终端节点、IAM 权限和 CloudWatch 成本归因。网关则增加了跨供应商的故障转移、统一账单和速率限制。

更深层的问题在于模型目录。如果你的产品需要同时使用 Claude、Llama 和 Gemini，除非你同时接入 Bedrock、Vertex AI 和 Azure OpenAI，否则无法从单一渠道购买所有模型。这些超大规模云厂商并非可互换的——它们各自押注了不同的模型层归属权。

本课将梳理这三种押注、延迟差距、FinOps 差距以及供应商锁定风险。

## 核心概念

### 三种策略

**AWS Bedrock** —— 模型市场。包含 Claude（Anthropic）、Llama（Meta）、Titan（AWS 自研）、Stability（图像）、Cohere（嵌入）、Mistral，以及图像和嵌入子目录。一套 API、一套 IAM 权限面、一份 CloudWatch 导出记录。Bedrock 的押注在于：客户更看重选项的丰富性，而非单一模型。

**Azure OpenAI** —— 独家合作。提供 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心内对 OpenAI 模型进行微调。在“Azure OpenAI Service”目录中没有非 OpenAI 模型——这些会进入 Azure AI Foundry（独立产品）。Azure 的押注在于：OpenAI 仍将保持前沿地位，且客户希望在该特定合作关系中获得企业级控制力。

**Vertex AI** —— Gemini 优先，其他其次。包括 Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，以及 Model Garden（第三方模型）。Vertex 的押注在于多模态长上下文——100 万 token 的 Gemini 上下文是其差异化优势。

### 规模下的延迟差距

Artificial Analysis 运行着持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需实例）上，Azure OpenAI 的首字延迟（TTFT）中位数约为 50 ms；Bedrock 约为 75 ms。这一差距并非 AWS 的失败，而是容量模型的差异。Azure 销售 PTU（预留吞吐量单元），为您的租户预留 GPU 容量。Bedrock 的对应方案（预留吞吐量）也存在，但起步价约为每单元 21 美元/小时，大多数客户仍停留在共享按需模式。

共享按需容量需与其他所有客户的流量竞争。专属容量则不会。如果您的产品 SLA 要求 P99 级别的 TTFT < 100 ms，您要么在 Azure 购买 PTU，要么购买 Bedrock 预留吞吐量，要么接受默认的性能波动。

### 预留吞吐量经济学

Azure PTU：一块预留的推理计算资源块。对于可预测的工作负载，相比按需实例最高可节省约 70% 的成本。费用按小时固定收取——即使空闲也需支付预留费用。盈亏平衡点通常在 40%-60% 的持续利用率左右。

Bedrock 预留吞吐量：根据模型和区域不同，每小时 21-50 美元。逻辑类似——盈亏平衡点约为峰值利用率的一半。需要月度承诺。

Vertex 预留容量按 Gemini SKU 出售；定价因模型和区域而异，且公开信息较少。

### FinOps 体系——真正的差异化因素

**Bedrock 应用推理配置档**是市场中最为清晰的归因方案。用 `team`、`product`、`feature` 标记一个配置档；将所有模型调用路由至其中；CloudWatch 无需后处理即可按配置档拆分成本。该功能于 2025 年推出，目前仍是超大规模云厂商中最细粒度的原生方案。

**Vertex** 的归因采用“按团队划分项目”加“处处打标签”的模式。您将每个团队建模为一个 GCP 项目，在所有资源上打上标签，并使用 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 允许您对成本数据执行任意 SQL 查询。

**Azure** 依赖订阅/资源组作用域加标签，并将 PTU 预留作为一等公民的成本对象。标签继承自资源组而非请求级别，因此要实现按请求归因，需要借助 Application Insights 自定义指标或一个能加盖请求头戳的网关。

总结规律：Bedrock 原生归因最清晰，Vertex 通过 BigQuery 最灵活，Azure 若不自行埋点则最为不透明。

### 锁定是 2026 年的核心风险

当单一模型占据主导时，绑定单一超大规模云厂商尚可接受。但在 2026 年，前沿技术每月都在迭代——一季度是 Claude 3.7，下一季度是 Gemini 2.5，再下一季度是 GPT-5。锁定单一平台意味着您将失去三分之二的的前沿模型访问权。

当前成熟团队采用的模式是：任何关键产品级的 LLM 调用必须满足“双供应商最低要求”。Bedrock 加上 Azure OpenAI 是最常见的组合——从一个供应商获取 Claude，从另一个获取 GPT，通过同一网关实现故障转移。成本增幅微乎其微，因为网关会自动路由最优路径；而在发生中断事件（如 2025 年 1 月的 Azure OpenAI 事故、AWS us-east-1 宕机）时，可用性提升则是决定性的。

### 数据驻留、BAA 与受监管行业

Bedrock：多数区域支持 BAA；支持 VPC 终端节点；内置护栏。常见金融科技首选。
Azure OpenAI：符合 HIPAA、SOC 2、ISO 27001；支持欧盟数据驻留；是企业合规场景的默认选择。
Vertex：符合 HIPAA、GDPR；按区域支持数据驻留；依托 Google Cloud 的合规栈。

三者均能满足基础合规要求。差异主要体现在数据保留策略、日志处理方式，以及是否由提供商读取您的流量用于滥用监控（大多数默认开启，企业版可提供关闭选项）。

### 需要记住的关键数据

- Azure OpenAI 在等效 Llama 3.1 405B 上的中位 TTFT：~50 ms（启用 PTU）。
- Bedrock 按需调用的中位 TTFT：~75 ms。
- Bedrock 预留吞吐量：每单元 21-50 美元/小时。
- Azure PTU 盈亏平衡点：~40-60% 持续利用率。
- 高利用率下 PTU 相比按需实例的节省幅度：最高达 70%。

## 实践应用

`code/main.py` 针对合成工作负载对比了这三个平台——它模拟了按需与 PTU 的经济学差异、TTFT 波动性以及成本归因的保真度。运行它以观察 PTU 何时能体现价值，以及何时市场的模型广度能够抵消 TTFT 的差距。

## 交付成果

本课将产出 `outputs/skill-managed-platform-picker.md`。给定工作负载配置文件（所需模型、TTFT SLA、日调用量、合规要求），它将推荐主用平台、备用平台以及 FinOps 埋点方案。

## 练习

1. 运行 `code/main.py`。对于 70B 级别的模型，Azure PTU 在何种持续利用率下会优于按需实例？计算盈亏平衡点，并与宣传的 40-60% 区间进行对比。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署方案——哪些模型分配给哪个超大规模云厂商，前端放置什么网关，故障转移策略是什么？
3. 一家受监管的医疗客户要求具备 BAA、美国东部数据驻留能力，以及 P99 TTFT < 100 ms。选择一个平台并用三个具体特性加以论证。
4. 你发现本月 Bedrock 账单上涨了 4 倍，但流量并无变化。在没有应用推理配置档的情况下，你将如何找出罪魁祸首？如果有配置档，需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页面。对于一个每月 1 亿 token 的 Claude 工作负载，哪种方式更便宜——直接 Anthropic API、Bedrock 按需调用，还是 Bedrock 预留吞吐量？

## 关键术语

| 术语 | 业内说法 | 实际含义 |
|------|----------|----------|
| Bedrock | “AWS 的 LLM 服务” | 覆盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | “Azure 的 ChatGPT” | 在 Azure 数据中心运行的独家 OpenAI 模型，附带企业级控制 |
| Vertex AI | “Google 的 LLM” | 以 Gemini 为核心的平台，Model Garden 提供第三方模型 |
| PTU | “专属容量” | 预留吞吐量单元（Provisioned Throughput Unit）——预留推理 GPU，按小时计价 |
| Application Inference Profile | “Bedrock 标签” | 带标签的按产品划分的成本/用量配置档，原生支持 CloudWatch |
| Model Garden | “Vertex 目录” | Vertex AI 的第三方模型专区，与 Gemini 独立 |
| Two-provider minimum | “LLM 冗余” | 策略要求将每条关键 LLM 路径部署在 ≥2 个超大规模云厂商之上 |
| BAA | “HIPAA 文件” | 商业伙伴协议（Business Associate Agreement）；涉及 PHI 时必须签署；三家均提供 |
| Abuse monitoring | “日志监控者” | 提供商侧对提示词/输出的安全扫描；企业版可关闭 |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) —— 权威费率表及预留吞吐量定价。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) —— PTU 经济学与费率表。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) —— Gemini 层级与 Model Garden 附加费。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) —— 跨供应商的持续延迟与吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) —— 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) —— 归因机制横向对比。
