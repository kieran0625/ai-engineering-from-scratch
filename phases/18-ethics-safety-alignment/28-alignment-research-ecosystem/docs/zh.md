# 对齐研究生态系统 —— MATS、Redwood、Apollo、METR

> 五家机构定义了 2026 年的非实验室对齐研究层。MATS（机器学习对齐与理论学者）：自 2021 年底以来已有 527 名以上研究人员，发表 180 余篇论文，引用量超 1 万次，h-index 为 47；2024 年夏季批次已注册为 501(c)(3) 非营利组织，拥有约 90 名学者和 40 名导师；2025 年之前的校友中 80% 从事安全/安全工作，其中 200 余人任职于 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入了 AI Control（第 10 课内容）；与 UK AISI 合作开展控制安全案例研究。Apollo Research：为前沿实验室提供部署前的图谋评估；撰写了《上下文中的图谋》（第 8 课内容）与《迈向 AI 图谋的安全案例》。METR（模型评估与威胁研究）：基于任务的能力评估、自主任务时间跨度研究；《前沿 AI 安全政策的共同要素》对比了各实验室的框架。Eleos AI Research：模型福祉的部署前评估（第 19 课内容）；完成了 Claude Opus 4 的福祉评估。

**类型：** 学习
**语言：** 无
**前置条件：** 第 18 阶段 · 01-27（第 18 阶段的前置课程）
**时间：** 约 45 分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五家核心机构及其主要产出。
- 描述 MATS 的规模（学者数量、论文数、h-index）及其作为人才管道的角色。
- 描述 Redwood 的 AI Control 议程及其与 UK AISI 的合作关系。
- 描述 METR 的基于任务的评估方法。

## 问题背景

前沿实验室（第 18 课）在内部进行安全评估并发布部分结果。而实验室外部的生态系统则是验证这些评估、首次发现新型失效模式以及培养人才的场所。理解该生态系统有助于解读哪些研究成果被谁所信任。

## 核心概念

### MATS（机器学习对齐与理论学者）

始于 2021 年底。一项研究导师计划；学者将在资深研究员的指导下，用 10-12 周时间专注于解决特定的对齐问题。

规模（2026 年）：
- 自成立以来已有 527 名以上研究人员。
- 发表 180 余篇论文。
- 引用量超 1 万次。
- h-index 为 47。
- 2024 年夏季批次：90 名学者 + 40 名导师；已注册为 501(c)(3) 组织。

职业去向：约 80% 的 2025 年之前校友正在从事安全/安全工作。200 余人任职于 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI Control 议程（第 10 课内容）。与 UK AISI 合作开展控制安全案例研究。就评估设计为 DeepMind 和 Anthropic 提供咨询。

代表性论文：Greenblatt、Shlegeris 等人，《AI Control》（arXiv:2312.06942，ICML 2024）；Alignment Faking（Greenblatt、Denison、Wright 等人，arXiv:2412.14093，与 Anthropic 联合发表）。

风格：具体的威胁模型、最坏情况下的对手、可经受压力测试的具体协议。

### Apollo Research

为前沿实验室提供部署前的图谋评估。撰写了《上下文中的图谋》（第 8 课内容，arXiv:2412.04984）。参与 2025 年 OpenAI 反图谋训练合作项目。发布了《迈向 AI 图谋的安全案例》（2024 年）。

风格：在智能体环境中进行可能产生欺骗行为的评估；采用三支柱分解法（不对齐、目标导向性、情境意识）。

### METR（模型评估与威胁研究）

基于任务的能力评估。自主任务时间跨度研究。《前沿 AI 安全政策的共同要素》（metr.org/common-elements，2025 年）对比了各实验室的框架。

与 Apollo 联合撰写了 AI 图谋安全案例草案。

风格：长周期任务评估、实证能力测量、框架综合。

### Eleos AI Research

模型福祉的部署前评估。完成了系统卡片第 5.3 节中记录的 Claude Opus 4 福祉评估。为第 19 课中关于福祉相关的主张提供外部方法论审查。

### 运作流程

MATS 培养研究人员。毕业生流向 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估机构）。外部评估者与实验室及 UK AISI / CAISI 合作。出版物成果反馈至 MATS，供下一期学员使用。

### 为何这一层级至关重要

单一来源的评估并不可靠：实验室自我评估其模型存在结构性利益冲突。外部评估者能够提出并验证实验室可能低估或隐瞒的失效模式。2024 年的休眠代理论文（第 7 课内容）由 Anthropic 与 Redwood 合作完成；Alignment Faking 由 Anthropic 与 Redwood 合作完成；In-Context Scheming 由 Apollo 完成；Anti-Scheming 由 Apollo 与 OpenAI 合作完成。多机构协作结构构成了质量控制机制。

### 在本阶段中的定位

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架对比；第 19 课引用了 Eleos。第 28 课为本阶段其余课程所依赖的生态系统提供了明确的组织架构图谱。

## 实践应用

无需编写代码。阅读 METR 的《前沿 AI 安全政策的共同要素》，以此了解外部综合研究如何为实验室内部的政策工作增添价值。

## 交付成果

本课产出 `outputs/skill-ecosystem-map.md`。给定一个对齐主张或评估结果，该工具将识别涉及的机构、发表渠道，以及方法论风格，并与已知对应机构进行交叉核对。

## 练习

1. 从第 7-15 课中挑选一篇论文，识别参与的机构。将作者名单与 MATS 校友及当前生态系统隶属关系进行交叉核对。

2. 阅读 METR 的《前沿 AI 安全政策的共同要素》。指出他们强调的三个跨实验室共识，以及两个最大的分歧点。

3. MATS 的职业去向显示约 80% 从事安全/安全工作。请论证这种选择压力是适应性的（培养了领域人才）还是存在偏差的（过滤掉了非主流观点）。

4. Redwood 和 Apollo 都从事控制/图谋相关工作，但风格不同。请选择一种失效模式，并描述两者将如何分别对其进行调查。

5. Eleos AI 是唯一纯粹专注于模型福祉的机构。请设计一家假设的第二家机构，聚焦于不同的福祉相关问题（如认知自由、机器人具身化等），并阐明其方法论。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| MATS | “导师计划” | 机器学习对齐与理论学者；自 2021 年以来已有 527 名以上研究人员 |
| Redwood Research | “控制实验室” | 应用对齐；AI Control 作者方；UK AISI 合作伙伴 |
| Apollo Research | “图谋评估机构” | 为前沿实验室提供部署前的图谋评估 |
| METR | “任务跨度评估机构” | 基于任务的能力评估；框架综合 |
| Eleos AI | “福祉实验室” | 模型福祉的部署前评估 |
| 人才管道 | “MATS -> 实验室” | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | “非实验室核查” | 不由模型生产方执行的评估；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) —— 导师计划
- [Redwood Research](https://www.redwoodresearch.org/) —— AI Control 论文
- [Apollo Research](https://www.apolloresearch.ai/) —— 图谋评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) —— 框架对比
- [Eleos AI Research](https://www.eleosai.org/research) —— 模型福祉方法论
