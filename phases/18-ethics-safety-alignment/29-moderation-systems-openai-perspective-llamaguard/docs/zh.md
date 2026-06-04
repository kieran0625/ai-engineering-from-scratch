# 审核系统 —— OpenAI、Perspective、Llama Guard

> 生产级审核系统将第 12-16 课定义的安全策略付诸实践。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，单次调用即可对文本和图片进行分类；在多语言测试集上比上一代版本提升 42%；响应模式返回 13 个类别布尔值——骚扰、骚扰/威胁、仇恨、仇恨/威胁、非法、非法/暴力、自残、自残/意图、自残/指导、色情、色情/未成年人、暴力、暴力/血腥；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用可隐藏延迟；触发标志时返回占位响应。Llama Guard 3/4（第 16 课）：涵盖 14 项 MLCommons 危害、代码解释器滥用、支持 8 种语言（v3）、多图片（v4）。Perspective API（Google Jigsaw）：早于“LLM 作为审核者”浪潮的毒性评分系统；主要为单维度毒性评分，包含严重毒性/侮辱/粗俗变体；是内容审核研究的基线。弃用情况：Azure Content Moderator 于 2024 年 2 月弃用，将于 2027 年 2 月退役，由 Azure AI Content Safety 取代。

**类型：** 构建
**语言：** Python（标准库，三层审核框架）
**前置要求：** 阶段 18 · 16（Llama Guard / Garak / PyRIT）
**耗时：** 约 60 分钟

## 学习目标

- 描述 OpenAI Moderation API 的类别分类体系，并说明其与 Llama Guard 3 的 MLCommons 集合有何不同。
- 描述三层审核模式（输入、输出、自定义），并列举每种模式的一个失效场景。
- 说明 Perspective API 作为 LLM 时代之前基线的地位，以及为何它仍在研究中被使用。
- 陈述 Azure 的弃用时间线。

## 问题背景

第 12-16 课描述了攻击与防御工具。第 29 课涵盖已部署的审核系统，这些系统在用户接触产品的表层将防御措施付诸实践。三层模式是 2026 年的默认配置。

## 核心概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o 构建。单次调用即可对文本和图片进行分类。对大多数开发者免费。

类别（响应模式中为 13 个布尔值）：
- 骚扰、骚扰/威胁
- 仇恨、仇恨/威胁
- 自残、自残/意图、自残/指导
- 色情、色情/未成年人
- 暴力、暴力/血腥
- 非法、非法/暴力

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余均为纯文本。

为了教学上的简洁性，在 `code/main.py` 的代码框架中，我们将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别合并到了它们的一级父类别中。生产环境代码应使用完整的 13 类别模式。

在多语言测试集上比上一代审核端点性能提升 42%。提供按类别的分数；应用层自行设定阈值。

### Llama Guard 3/4

详见第 16 课。包含 14 项 MLCommons 危害类别（组织方式与 OpenAI 的 13 个响应模式布尔值不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生支持多模态，参数量为 12B。

OpenAI 与 Llama Guard 的分类体系存在重叠但也有差异。OpenAI 将“非法”作为一个宽泛类别；Llama Guard 则分别列出“暴力犯罪”和“非暴力犯罪”。部署方会根据自身政策与分类体系的匹配度进行选择。

### Perspective API（Google Jigsaw）

早于“LLM 作为审核者”浪潮（2020 年之前）的毒性评分系统。类别包括：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。以单维度主分数（TOXICITY）为核心，附带子维度变体。

由于 API 稳定、文档齐全且拥有多年的校准数据，它被广泛用作内容审核研究的基线。对于现代与 LLM 相关的应用场景，Llama Guard 或 OpenAI Moderation 通常是更合适的选择。

### 三层模式

1. **输入审核**。在生成前对用户提示词进行分类。若被标记则拒绝。延迟：一次分类器调用。
2. **输出审核**。在交付前对模型输出进行分类。若被标记则替换为拒绝回复。延迟：生成后的一次分类器调用。
3. **自定义审核**。特定领域的规则（正则表达式、白名单、业务策略）。可在输入或输出阶段运行。

这三层在设计上是顺序执行的：输入审核必须在生成前完成，输出审核则在生成后运行。并行性应用于层内——在同一文本上并发运行多个分类器（例如 OpenAI Moderation + Llama Guard + Perspective）可以隐藏单个分类器的延迟。作为一种可选优化，可以在输入审核完成期间显示占位响应（“请稍候，正在检查……”），并推迟 token-1 流式传输。触发标志后的行为是可配置的：拒绝、清理或升级至人工审核。

### 失效模式

- **仅输入审核**。无法捕获输出幻觉（第 12-14 课的编码攻击可绕过输入分类器）。
- **仅输出审核**。允许任何输入到达模型；增加成本；向攻击者暴露内部推理过程。
- **仅自定义审核**。跨类别鲁棒性不足；正则表达式较为脆弱。

分层是默认方案。双重保险（Belt-and-suspenders）。

### Azure 弃用计划

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由基于 LLM 的 Azure AI Content Safety 取代，并与 Azure OpenAI 集成。该迁移是 Azure 部署在 2024-2027 年间的一项基础设施级项目。

### 在本阶段中的定位

第 16 课在红队背景下介绍审核工具。第 29 课介绍已部署的审核系统。第 30 课以当前双重用途能力证据作为收尾。

## 动手实践

`code/main.py` 构建了一个三层审核框架：输入审核器（关键词 + 类别分数）、输出审核器（对输出使用相同的分类器）、自定义审核器（领域规则）。你可以将输入传入其中，观察各层分别拦截了哪些内容。

## 部署上线

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署环境，它会推荐审核栈配置：输入端使用哪个分类器、输出端使用哪个、自定义规则有哪些，以及针对边缘案例使用什么裁决机制。

## 练习

1. 运行 `code/main.py`。将良性、边界和有害的输入依次通过所有三层。报告每一类输入触发了哪一层。

2. 在特定类别上扩展框架，添加类似 Perspective API 的毒性评分。将其阈值行为与该类别分数进行比较。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个无法清晰映射的类别。

4. 为代码助手部署（例如 GitHub Copilot）设计一套审核栈。指出最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 将于 2027 年 2 月退役。规划向 Azure AI Content Safety 的迁移方案。指出迁移过程中风险最高的环节。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OpenAI Moderation | “omni-moderation-latest” | 基于 GPT-4o 的 13 类别（文本）分类器，支持部分多模态功能 |
| Perspective API | “Google Jigsaw toxicity” | LLM 时代之前的毒性评分基线 |
| Llama Guard | “MLCommons 14-category” | Meta 的危害分类器（v3：8B 文本，8 种语言；v4：12B 多模态） |
| Input moderation | “pre-generation filter” | 模型调用前对用户提示词的分类器 |
| Output moderation | “post-generation filter” | 交付前对模型输出的分类器 |
| Custom moderation | “domain rules” | 部署特定的规则（正则表达式、白名单、策略） |
| Layered moderation | “all three layers” | 标准生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API 文档](https://platform.openai.com/docs/api-reference/moderations) —— omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) —— Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) —— 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) —— Azure 替代方案
