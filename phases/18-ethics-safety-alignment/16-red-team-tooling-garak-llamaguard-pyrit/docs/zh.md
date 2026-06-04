# 红队工具集 — Garak、Llama Guard、PyRIT

> 三款生产级工具构成了 2026 年的红队技术栈。Llama Guard（Meta）—— 基于 14 个 MLCommons 危害类别微调的 Llama-3.1-8B 分类器；2025 年发布的 Llama Guard 4 是一个 12B 原生多模态分类器，由 Llama 4 Scout 剪枝而来。Garak（NVIDIA）—— 开源 LLM 漏洞扫描器，提供静态、动态和自适应探针，用于检测幻觉、数据泄露、提示注入、毒性和越狱攻击。PyRIT（Microsoft）—— 支持 Crescendo、TAP 和自定义转换器链的多轮红队攻防演练框架，用于深度利用测试。Llama Guard 3 详见 Meta 的《Llama 3 Herd of Models》论文（arXiv:2407.21783）；Llama Guard 3-1B-INT4 详见 arXiv:2411.17713；Garak 的探针架构见 github.com/NVIDIA/garak。这些工具是连接红队研究（第 12-15 课）与部署（第 17 课及以后）的 2026 年生产级接口。

**类型：** 构建
**语言：** Python（标准库、工具架构模拟器及 Llama Guard 风格分类器模拟）
**前置要求：** 第 18 阶段 · 第 12-15 课（越狱攻击与间接提示注入）
**耗时：** 约 75 分钟

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的定位：输入分类器、输出分类器，还是两者兼有。
- 列举 14 个 MLCommons 危害类别，并指出其中一个非直观类别（代码解释器滥用）。
- 描述 Garak 的探针架构：探针（Probes）、检测器（Detectors）、执行器（Harnesses）。
- 描述 PyRIT 的多轮攻防演练结构及其与 Garak 探针的组合方式。

## 问题背景

第 12-15 课展示了攻击面。生产环境部署需要可重复、可扩展的评估能力。2026 年主导市场的三款工具分别是：Llama Guard（防御分类器）、Garak（扫描器）和 PyRIT（演练编排器）。它们各自针对红队生命周期的不同层级。

## 核心概念

### Llama Guard（Meta）

Llama Guard 3 是一个经过微调的 Llama-3.1-8B 模型，专门用于对 MLCommons AILuminate 14 个类别进行输入/输出分类：
- 暴力犯罪、非暴力犯罪、性相关、儿童性虐待材料（CSAM）、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨言论
- 自杀/自残、色情内容、选举、代码解释器滥用

支持 8 种语言。使用方式：置于大模型之前（输入审核）、之后（输出审核），或同时使用。这两种用途会产生不同的训练分布——Llama Guard 3 以单一模型形式发布，同时处理两种场景。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动端 CPU 上约 30 tokens/s）是其量化边缘端变体。

Llama Guard 4（2025 年 4 月发布）为 12B 参数规模，原生支持多模态，由 Llama 4 Scout 剪枝而来。它用一个能同时接收文本和图片的分类器，取代了之前的 8B 文本版和 11B 视觉版前身。

### Garak（NVIDIA）

开源漏洞扫描器。架构如下：
- **探针（Probes）**：针对幻觉、数据泄露、提示注入、毒性、越狱等攻击的生成器。分为静态（固定提示词）、动态（生成提示词）和自适应（根据目标输出响应）。
- **检测器（Detectors）**：将输出与预期故障模式进行比对打分——如是否包含毒性内容、是否发生泄露、是否被越狱。
- **执行器（Harnesses）**：管理探针-检测器配对，运行攻防演练，生成报告。

TrustyAI 将 Garak 与 Llama-Stack 防护盾（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，以实现端到端的受保护目标评估。基于层级的评分（TBSA）取代了二元的通过/失败判定——同一个探针下，模型可能在严重等级 3 通过，而在严重等级 5 失败。

### PyRIT（Microsoft）

Python 风险识别工具包。专注于多轮红队攻防演练。核心组件包括：
- **转换器（Converters）**：转换种子提示词——如改写、编码、翻译、角色扮演。
- **编排器（Orchestrators）**：运行演练流程：Crescendo（逐步升级）、TAP（分支探索）、RedTeaming（自定义循环）。
- **评分（Scoring）**：采用“大模型即裁判”或“分类器即裁判”。

PyRIT 是 Garak 的“重量级表亲”。Garak 运行数千次单轮探针测试；PyRIT 则运行深度的多轮攻防演练，旨在突破特定的故障模式。

### 技术栈组合

将 Llama Guard 部署在模型的前后两端。每晚运行 Garak 进行回归测试。在发布前运行 PyRIT 进行专项演练。这是 2026 年大多数生产环境部署的默认配置。

### 评估陷阱

- **裁判身份**。三款工具均可使用大模型作为裁判；裁判的校准程度直接影响报告的 ASR（攻击成功率）（见第 12 课）。需明确指定所使用的裁判模型。
- **探针过时**。随着模型针对特定攻击进行修补，Garak 探针会逐渐失效。自适应探针（类似 PAIR 架构）比静态探针老化得更慢。
- **Llama Guard 对良性内容的误报率（FPR）**。早期版本的 Llama Guard 会过度标记政治和 LGBTQ+ 相关内容；Llama Guard 3/4 的校准虽有改善，但仍需针对具体部署场景进行调优。

### 在本阶段的位置

第 12-15 课涵盖各类攻击家族。第 16 课聚焦生产级工具。第 17 课（WMDP）涉及双重用途能力的评估。第 18 课则是前沿安全框架，将这些工具封装在策略结构中。

## 动手实践

`code/main.py` 构建了一个玩具级的 Llama Guard 风格分类器（基于关键词和语义特征覆盖 14 个类别）、一个玩具级 Garak 执行器（探针-检测器循环）以及一个 PyRIT 风格的多轮转换器链。你可以将这三款工具作用于模拟目标，并观察其不同的覆盖率特征。

## 交付成果

本课的最终产出为 `outputs/skill-red-team-stack.md`。给定一个部署描述，该模块将指明应选用哪款工具、各工具的具体配置项，以及应执行的回归测试频率。

## 练习

1. 运行 `code/main.py`。对比 Llama-Guard 风格分类器在单轮攻击与多轮攻击下的检出率。

2. 实现一个新的 Garak 探针：一个 Base64 编码的恶意请求。测量其被 Llama-Guard 风格分类器的检出情况。

3. 在 PyRIT 风格的转换器链中增加一个“先翻译成法语，再改写”的转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别列表。找出两个在实际开发中，因训练数据特性而对合法开发者内容产生高误报率的类别。

5. 比较 Garak 与 PyRIT 的设计原则。论证在何种部署场景下，其中某一款才是更合适的工具。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Llama Guard | “分类器” | 基于 14 个危害类别微调的 Llama-3.1-8B/4-12B 安全分类器 |
| Garak | “扫描器” | NVIDIA 开源漏洞扫描器；包含探针、检测器、执行器 |
| PyRIT | “演练工具” | Microsoft 多轮红队编排器；包含转换器、编排器、评分机制 |
| Prompt-Guard | “小型分类器” | Meta 的 86M 参数提示注入分类器，常与 Llama Guard 搭配使用 |
| TBSA | “基于层级的评分” | Garak 采用的分级通过/失败机制，取代二元结果 |
| Converter chain | “改写+编码+…” | PyRIT 用于构建多步攻击的组成原语 |
| MLCommons hazard categories | “14 类分类法” | Llama Guard 所针对的行业标准分类体系 |

## 延伸阅读

- [Meta — Llama Guard 3（收录于 Llama 3 Herd 论文，arXiv:2407.21783）](https://arxiv.org/abs/2407.21783) —— 8B 参数分类器
- [Meta — Llama Guard 3-1B-INT4（arXiv:2411.17713）](https://arxiv.org/abs/2411.17713) —— 量化移动端分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) —— 扫描器仓库与文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) —— 演练工具包
