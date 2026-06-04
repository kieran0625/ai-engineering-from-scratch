# LLM 功能 A/B 测试 —— GrowthBook、Statsig 与“凭感觉”问题

> 传统的 A/B 测试并非为非确定性 LLM 而设计。关键区别在于：评估（Evals）回答“模型能否完成任务？”，A/B 测试回答“用户是否在意？”。两者缺一不可；依靠“凭感觉”来发布功能的日子已经结束了。2026 年需要测试的内容包括：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源模型；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：某聊天机器人奖励模型变体使对话长度增加 70%，留存率提升 30%；Nextdoor AI 标题行实验在优化奖励函数后点击率（CTR）提升 1%；Khan Academy Khanmigo 则在延迟与数学准确率之间进行了迭代权衡。平台对比：**Statsig**（OpenAI 于 2025 年 9 月以 11 亿美元收购）——支持序贯检验、CUPED、一站式方案。**GrowthBook** —— 开源、原生仓库架构，支持贝叶斯+频率学派+序贯引擎，具备 CUPED、SRM 检查、Benjamini-Hochberg 与 Bonferroni 校正。根据你的仓库-SQL 技术栈倾向以及企业对收购背景的接受程度来选择。

**类型：** 学习
**语言：** Python（标准库、简易序贯检验模拟器）
**前置条件：** 第 17 阶段 · 13（可观测性），第 17 阶段 · 20（渐进式部署）
**耗时：** 约 60 分钟

## 学习目标

- 区分评估（Evals，“模型能否完成任务”）与 A/B 测试（“用户是否在意”）。
- 列举三个可测试维度（提示词、模型、参数），并为每个维度选择合适的指标。
- 解释 CUPED、序贯检验以及 Benjamini-Hochberg 多重比较校正。
- 根据仓库-SQL 技术栈倾向及企业对收购背景的接受程度，选择 Statsig 或 GrowthBook。

## 问题所在

你手动调优了系统提示词。感觉更好了。你发布了它。转化率因噪声发生了波动。你归咎于指标。或者你上线了新模型，但转化率毫无变化——是模型性能下降了，还是变化太小无法检测？你不知道答案，因为你没有经过 A/B 测试就发布了。

评估（Evals）回答的是模型在标注数据集上能否完成任务。它们并不回答用户是否更喜欢该输出。只有受控的在线实验才能给出答案，且前提是实验具有足够的统计功效、控制了非确定性因素，并对多重比较进行了校正。

## 核心概念

### 评估（Evals）与 A/B 测试

**评估（Evals）** —— 离线、标注数据集、裁判（评分标准、LLM-as-judge 或人工）。回答：“在该固定分布下，输出是否正确/有帮助/安全？”

**A/B 测试** —— 在线、真实用户、随机分配。回答：“新变体是否推动了关键的用户级指标？”

两者缺一不可。评估在暴露前捕获回归问题；A/B 测试在上线后确认产品影响。

### 测试什么

1. **提示词工程** —— 措辞、系统提示词结构、示例。指标：任务成功率、用户留存率、单次请求成本。
2. **模型选择** —— GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务相关）+ 单次请求成本 + P99 延迟。多目标优化。
3. **生成参数** —— temperature、top-p、max_tokens。指标：任务特定指标（输出多样性与确定性的权衡）。

### CUPED —— 方差缩减

Controlled-experiments Using Pre-Experiment Data（利用实验前数据进行的控制实验）。在比较实验后数据之前，通过回归消除实验前期的方差。典型方差缩减幅度：30%-70%。有效样本量免费提升。

实现情况：Statsig 和 GrowthBook 均支持。

### 序贯检验

传统 A/B 测试假设固定样本量。序贯检验（“随时查看并决策”）在多次查看时能控制假阳性率。始终有效的序贯程序（如 mSPRT、Howard 置信序列）允许你在出现明显胜者时提前停止。

### 多重比较校正

以 95% 置信度运行 20 次 A/B 测试，按概率会产生一次假阳性。Bonferroni 校正收紧每次测试的 α 值；Benjamini-Hochberg 控制错误发现率（FDR）。GrowthBook 同时实现了这两种方法。

### SRM —— 样本比例不匹配

分配哈希将用户随机分配到各变体。如果是 50/50 分流却得到 47/53 的结果，说明出了问题——SRM 检查会标记此异常。两个平台均内置此检查。

### Statsig 与 GrowthBook

**Statsig**：
- 由 OpenAI 于 2025 年 9 月以 11 亿美元收购。托管型 SaaS。
- 支持序贯检验、CUPED、预留测试人群。
- 一体化方案：功能开关 + 实验平台 + 可观测性。
- 最佳适用场景：团队已希望获得捆绑产品，且不介意 OpenAI 的所有权背景。

**GrowthBook**：
- 开源（MIT 协议）；原生仓库架构（直接从 Snowflake/BigQuery/Redshift 读取数据）。
- 多引擎支持：贝叶斯、频率学派、序贯。
- 支持 CUPED、SRM、Bonferroni 与 BH 校正。
- 支持自托管或托管云。
- 最佳适用场景：重度依赖仓库-SQL 的团队，数据团队掌控指标层，且偏好开源软件。

### 非确定性增加统计功效计算的复杂度

相同的提示词会产生不同的输出。传统的功效计算假设观测值是独立同分布（IID）的。由于 LLM 的非确定性，有效样本量会低于名义样本量。建议将所需样本量乘以约 1.3-1.5 倍作为安全余量。

### 真实案例结果

- 聊天机器人奖励模型变体：对话长度增加 70%，留存率提升 30%。
- Nextdoor 邮件主题行：优化奖励函数后点击率（CTR）提升 1%。
- Khan Academy Khanmigo：在延迟与数学准确率之间进行迭代权衡。

### 反模式：凭感觉发布

每位资深工程师都能举出一个仅因“感觉更好”就发布且未经过 A/B 测试的功能案例。其中大多数都导致了产品指标下滑，而团队数月未察觉。A/B 测试正是强制约束这一行为的机制。

### 需要记住的关键数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：开源 MIT 协议；支持贝叶斯 + 频率学派 + 序贯。
- CUPED 方差缩减：30%-70%。
- LLM 非确定性 → 需增加 30%-50% 的样本量缓冲。

## 动手实践

`code/main.py` 模拟了带有固定边界和序贯边界的序贯 A/B 测试。展示了序贯检验如何让你提前停止测试。

## 交付成果

本课产出 `outputs/skill-ab-plan.md`。给定功能变更、工作负载和基线数据，它将自动选择平台、设置发布闸门（gates）及计算样本量。

## 练习

1. 运行 `code/main.py`。若预期提升为 5%，基线转化率为 3%，达到 80% 统计功效所需的样本量是多少？
2. 为一家受医疗监管的本地部署（on-prem）客户选择 Statsig 或 GrowthBook。
3. 设计一个 A/B 测试，用于对比 GPT-4 与 GPT-3.5 在“每张解决工单的成本”上的表现。主指标、护栏指标和次要指标分别是什么？
4. 金丝雀发布（canary）已通过，但 A/B 测试显示转化率下降 1.2%。你会发布吗？请写出升级与处置标准。
5. 将 CUPED 应用于前期方差占后期 60% 的数据集。计算有效样本量的提升倍数。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| Eval | “离线测试” | 基于标注数据集的模型能力评估 |
| A/B test | “实验” | 面向真实用户的随机化对比 |
| CUPED | “方差缩减” | 利用前期数据进行回归以降低方差 |
| Sequential test | “随时查看测试” | 始终有效的允许提前终止的程序 |
| Multiple comparison | “族系误差” | 运行大量测试会放大假阳性率 |
| Bonferroni | “严格校正” | 将 α 除以测试次数 |
| Benjamini-Hochberg | “BH FDR” | 控制错误发现率，相对不那么保守 |
| SRM | “糟糕的分流” | 样本比例不匹配；分配逻辑存在缺陷 |
| Statsig | “OpenAI 旗下” | 商业一体化方案，2025 年被收购 |
| GrowthBook | “那个开源的” | MIT 协议的仓库原生平台 |
| mSPRT | “序贯概率比检验” | 经典序贯检验程序 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
