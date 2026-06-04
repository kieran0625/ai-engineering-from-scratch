# 推理平台经济学 —— Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026年的推理市场已不再是简单的GPU时间租赁。它分化为三大阵营：定制芯片（Groq、Cerebras、SambaNova）、GPU平台（Baseten、Together、Fireworks、Modal）以及API优先的市场（Replicate、DeepInfra）。Fireworks于2026年5月1日将每GPU每小时租金上调了$1，其日处理量超10万亿token且估值达40亿美元，印证了以规模驱动的模式行之有效。Baseten于2026年1月完成3亿美元E轮融资，估值50亿美元。竞争定位法则很简单：Fireworks优化延迟，Together优化模型目录广度，Baseten优化企业级打磨程度，Modal优化原生Python开发者体验（DX），Replicate优化多模态覆盖范围，Anyscale优化分布式Python。本课将为你提供一份可交付给创始人的选型矩阵。

**类型：** 学习
**语言：** Python（标准库，简易单次调用经济成本对比脚本）
**前置知识：** 第17阶段 · 01（托管LLM平台）、第17阶段 · 04（vLLM服务内部原理）
**预计耗时：** 约60分钟

## 学习目标

- 说出三大细分市场（定制芯片、GPU平台、API优先），并将各厂商对应到相应细分市场。
- 解释为何“按token计费”的API定价模式会向推理引擎的成本曲线收敛，而非硬件成本曲线。
- 计算至少三家供应商的单请求有效成本，并说明在何种情况下按分钟计费（Baseten、Modal）优于按token计费。
- 针对特定工作负载（无服务器突发型、稳定高吞吐型、微调变体型、多模态型），识别最合适的默认平台。

## 问题背景

你评估过托管型云厂商平台后，决定需要更专注、更快的提供商——追求延迟选Fireworks，追求广度选Together，追求微调自定义模型选Baseten。现在你有六个真实选择，但它们的定价页面完全无法直接对齐。Fireworks显示的是 $/M tokens；Baseten显示的是 $/minute；Modal显示的是 $/second；Replicate显示的是 $/prediction。若不对工作负载进行建模，你就无法将它们进行直观对比。

更糟的是，每个定价页面背后的商业模式各不相同。Fireworks在共享GPU上运行其自研引擎（FireAttention）；按token费率反映了其利用率曲线。Baseten提供Truss + 专属GPU；按分钟计费体现了独占性。Modal是真正的Python无服务器架构——按秒计费且冷启动低于1秒。相同的输出（一个LLM响应），三种不同的成本函数。

本课将对这六家平台进行建模，并告诉你各自在什么场景下胜出。

## 核心概念

### 三大细分市场

**定制芯片** —— Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在相同模型下，其解码速度通常比基于GPU的集群快5-10倍。按token单价较高（2025年底Llama-70B约为~$0.99/M），但在对延迟敏感的场景中无可匹敌。Groq是语音代理和实时翻译的生产环境首选。

**GPU平台** —— Baseten、Together、Fireworks、Modal、Anyscale。运行于NVIDIA（2026年的H100、H200、B200）或偶尔的AMD芯片上。处于“裸GPU租赁”（RunPod、Lambda）与“超大规模云托管服务”（Bedrock）之间的经济层。

**API优先市场** —— Replicate、DeepInfra、OpenRouter、Fal。模型目录广泛，按预测次数或按秒计费，强调首次调用时间。

### Fireworks —— 延迟优化的GPU平台

- FireAttention引擎（自研）；官方宣称在同等配置下延迟比vLLM低4倍。
- 批量层级价格约为无服务器价格的50%，适用于非交互式工作负载。
- 微调模型的服务费率与基础模型相同——相较于那些对你的LoRA收取溢价的服务商，这是一个真正的差异化优势。
- 2026年中：按需GPU租赁费用上调$1/小时，于2026年5月1日生效。大规模用量可协商阶梯定价。
- 财务信号：估值40亿美元，日处理量超10万亿token。

### Together —— 广度优化的平台

- 拥有200+款模型，包括上游发布后数天内上线的开源模型。
- 在同等LLM模型上比Replicate便宜50%-70%——其“AI原生云”的定位核心在于规模与目录。
- 推理 + 微调 + 训练集成于单一API。

### Baseten —— 企业级打磨优化的平台

- Truss框架：将依赖项、密钥、服务配置打包进单一清单文件。
- GPU型号涵盖从T4到B200。按分钟计费，并提供合理的冷启动缓解机制。
- 符合SOC 2 Type II标准，支持HIPAA合规。是金融科技和医疗行业的常见选择。
- 估值50亿美元，2026年1月完成E轮融资（CapitalG、IVP、NVIDIA等投资3亿美元）。

### Modal —— 原生Python优化的平台

- 纯Python实现的Infrastructure-as-code。使用 ``@modal.function(gpu="A100")`` 装饰器修饰函数即可一键部署。
- 按秒计费。预热状态下冷启动为2-4秒；小模型冷启动<1秒。
- 2025年完成8700万美元B轮融资，估值11亿美元。在独立调查中开发者体验（DX）评分最高。

### Replicate —— 多模态广度平台

- 按预测次数计费。图像、视频和音频模型的默认平台。
- 集成生态丰富（Zapier、Vercel、CMS插件等）。
- 在LLM按token费率上竞争力较弱，但在多模态多样性上占据优势。

### Anyscale —— Ray原生平台

- 基于Ray构建；RayTurbo是Anyscale的专有推理引擎（与vLLM竞争）。
- 最适合分布式Python工作负载，其中推理步骤仅是更大计算图中的一个节点。
- 托管Ray集群；与Ray AIR和Ray Serve深度集成。

### 按token计费 vs 按分钟计费 —— 何时胜出

当工作负载对延迟不敏感且具有突发性时，按token计费更合理——你只为实际使用的部分付费。当利用率高且可预测时，按分钟计费更合理——一旦GPU达到饱和，按分钟计费就会优于按token计费。

经验法则：对于专用GPU持续利用率超过~30%的工作负载，按分钟计费（Baseten、Modal）开始优于按token计费（Fireworks、Together）。低于该阈值时，按token计费胜出，因为你无需为空闲时间付费。

### 自研引擎才是真正的护城河

除vLLM和SGLang外，每家平台都宣称拥有自研引擎。FireAttention、RayTurbo、Baseten的推理栈皆是如此。所谓“自研引擎”的说法带有营销水分——客观来看，vLLM + SGLang占据了约80%的生产环境开源推理份额，平台层的真正差异化在于开发者体验（DX）、归因与监控（attribution）和服务等级协议（SLA）。

### 需要记住的关键数据

- Fireworks GPU租赁费：2026年5月1日起上调$1/小时。
- Fireworks宣称：同等配置下延迟比vLLM低4倍。
- Together：LLM模型比Replicate便宜50%-70%。
- Baseten估值：50亿美元（2026年1月E轮，3亿美元融资）。
- Modal估值：11亿美元（2025年B轮）。
- 持续利用率高于~30%时，按分钟计费优于按token计费。

## 实践应用

`code/main.py` 基于合成工作负载，跨定价模型对比这六家供应商。报告每日成本（$/day）及等效每百万token成本（$/M tokens）。运行它以找出按token计费与按分钟计费的盈亏平衡点。

## 交付成果

本课将产出 `outputs/skill-inference-platform-picker.md`。根据工作负载特征、SLA要求和预算，选择主推理平台并列出第二选择。

## 练习

1. 运行 `code/main.py`。在单张H100上运行70B模型时，Baseten（按分钟计费）在何种持续利用率下会优于Fireworks（按token计费）？自行推导交叉点并与经验法则进行对比。
2. 你的产品同时提供图像生成、聊天和语音转文本服务。为每种模态选择合适的平台，并指出统一它们的网关模式。
3. Fireworks对你主力模型的租赁费上调了$1/小时。若40%的流量迁移至批量层级（享受50%折扣），请建模分析混合成本的变动影响。
4. 某受监管客户要求具备SOC 2 Type II认证、HIPAA合规支持及专属GPU。哪三家平台符合要求？哪家在FinOps方面表现最佳？
5. 对比Llama 3.1 70B在Fireworks无服务器、Together按需、Baseten专属以及Replicate API上的每千次预测成本。在每天10次预测时哪个最便宜？每天10,000次时呢？

## 关键术语

| 术语 | 行业说法 | 实际含义 |
|------|----------|----------|
| Custom silicon | “非GPU芯片” | Groq LPU、Cerebras WSE、SambaNova RDU —— 针对解码优化 |
| FireAttention | “Fireworks引擎” | 自研注意力算子；官方宣称延迟比vLLM低4倍 |
| Truss | “Baseten格式” | 模型打包清单；包含依赖项 + 密钥 + 服务配置 |
| Per-token | “API定价” | 按消耗的token收费；不为空闲时间付费 |
| Per-minute | “专属定价” | 按GPU物理时钟时间收费；高利用率时占优 |
| Per-prediction | “Replicate定价” | 按模型调用次数收费；常用于图像/视频模型 |
| RayTurbo | “Anyscale引擎” | 基于Ray的专有推理；在Ray集群中与vLLM竞争 |
| Batch tier | “5折优惠” | 非交互式队列，享受折扣费率；Fireworks、OpenAI常见 |
| Fine-tuned at base rate | “Fireworks LoRA” | 按基础模型费率收取LoRA服务请求的费用（差异化优势） |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) —— 按token费率、批量层级、GPU租赁。
- [Baseten Pricing](https://www.baseten.co/pricing/) —— 按分钟费率、预留容量、企业层级。
- [Modal Pricing](https://modal.com/pricing) —— 按秒GPU费率及免费层级。
- [Together AI Pricing](https://www.together.ai/pricing) —— 模型目录与按token费率。
- [Anyscale Pricing](https://www.anyscale.com/pricing) —— RayTurbo与托管Ray定价。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) —— 对比评估。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) —— 厂商格局。
