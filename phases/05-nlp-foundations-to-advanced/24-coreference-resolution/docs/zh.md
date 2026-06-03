# 指代消解

> "她给他打了电话。他没有接。医生去吃午饭了。" 三句话提到了两个人，但没有人被点名。指代消解要弄清楚谁是谁。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 5 · 06（命名实体识别），阶段 5 · 07（词性标注与句法分析）
**时间：** ~60 分钟

## 问题

从一篇 300 词的文章中提取 Apple Inc. 的每一次提及。当文章说 "Apple" 时很简单。但当它说 "the company"、"they"、"Cupertino's technology giant" 或 "Jobs's firm" 时就难了。如果不将这些提及解析为同一实体，你的 NER 流水线会漏掉 60-80% 的提及。

指代消解将指向同一现实世界实体的每个表达链接到一个簇中。它是表层 NLP（NER、句法分析）与下游语义任务（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

2026 年为何重要：

- 摘要生成："The CEO announced..." 与 "Tim Cook announced..." —— 摘要应该点名 CEO。
- 问答系统："Who did she call?" 需要解析 "she"。
- 信息抽取：知识图谱中 "PER1 founded Apple" 和 "Jobs founded Apple" 作为两个独立条目是错误的。
- 多文档信息抽取：跨文章合并同一事件的提及属于跨文档指代消解。

## 概念

![指代聚类：提及 → 实体](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：提及（span）的聚类，每个簇指向一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词短语。** "the CEO"、"the company"
- **代词。** "he"、"she"、"they"、"it"
- **同位语。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs, 1978）。** 使用语法规则的句法树代词解析。良好的基线。在代词上 surprisingly 难以超越。
2. **提及对分类器。** 对每一对提及 (m_i, m_j)，预测它们是否共指。通过传递闭包聚类。2016 年前的标准方法。
3. **提及排序。** 对每个提及，排序候选先行词（包括 "no antecedent"）。选择最高分。
4. **基于 Span 的端到端（Lee et al., 2017）。** Transformer 编码器。枚举所有长度上限内的候选 span。预测提及分数。预测每个 span 的先行词概率。贪心聚类。现代默认方法。
5. **生成式（2024+）。** 提示 LLM："列出这段文本中的每个代词及其先行词。" 简单案例表现良好，长文档和罕见指代上表现不佳。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为单一指标无法捕捉聚类质量。报告前三个的平均值作为 CoNLL F1。2026 年 CoNLL-2012 上的最先进水平：~83 F1。

**已知难点。**

- 指向数页前引入实体的定指描述。
- 桥接指代（"the wheels" → 先前提到的汽车）。
- 汉语、日语等语言中的零形回指。
- 逆指（代词在指代对象之前）："When **she** walked in, Mary smiled."

## 动手实现

### 步骤 1：预训练神经指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你会得到类似这样的结果：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 步骤 2：基于规则的代词解析器（教学用）

参见 `code/main.py` 获取仅使用标准库的纯 Python 实现：

1. 提取提及：命名实体（首字母大写的 span）、代词（字典查找）、定指描述（"the X"）。
2. 对每个代词，查看前 K 个提及并按以下规则打分：
   - 性别/数一致（启发式）
   - 新近度（越近越好）
   - 句法角色（主语优先）
3. 链接最高分的先行词。

与神经模型相比没有竞争力。但它展示了搜索空间以及端到端模型必须做出的决策。

### 步骤 3：使用 LLM 进行指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

注意两种失败模式。第一，LLM 过度合并（"him" 和 "her" 指向两个不同的人）。第二，LLM 在长文档中静默遗漏提及。始终用 span 偏移量检查验证。

### 步骤 4：评估

标准 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，先从标注测试集上的 span 级别精确率和召回率开始，然后添加提及链接 F1。

## 陷阱

- **单例爆炸。** 某些系统将每个提及报告为独立的簇。B³ 对此宽容。MUC 会惩罚。始终检查三个指标。
- **长上下文中的代词。** 文档超过 2,000 token 时性能下降约 15 F1。谨慎分块。
- **性别假设。** 硬编码的性别规则在非二元指代、组织、动物上失效。使用学习模型或中性打分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地聚类 50+ 段落中的提及。使用滑动窗口 + 合并。

## 应用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP neural coref |
| 多语言 | 在 OntoNotes 或多语言 CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件指代 | 专用端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude 配合结构化输出指代消解 prompt |
| 生产对话系统 | 基于规则的兜底 + 神经主模型 + 关键槽位人工复核 |

2026 年实际部署的集成模式：先运行 NER，再运行指代消解，将指代簇合并到 NER 实体中。下游任务看到的是每个簇一个实体，而非每个提及一个实体。

## 交付

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 `code/main.py` 中的基于规则解析器上运行 5 段手工编写的段落。对照 ground truth 测量提及链接准确率。
2. **中等。** 在新闻文章上使用预训练神经指代模型。与手动标注的簇比较。哪里失败了？
3. **困难。** 构建指代增强的 NER 流水线：先 NER，再通过指代簇合并。在 100 篇文章上测量与纯 NER 相比的实体覆盖提升。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Mention（提及） | 一个指代 | 指向实体的文本 span（名称、代词、名词短语）。 |
| Antecedent（先行词） | "it" 指什么 | 后出现的提及与之共指的较早出现的提及。 |
| Cluster（簇） | 实体的提及 | 全部指向同一现实世界实体的提及集合。 |
| Anaphora（回指） | 向后指代 | 后出现的提及指向前面的（"he" → "John"）。 |
| Cataphora（逆指） | 向前指代 | 先出现的提及指向后面的（"When he arrived, John..."）。 |
| Bridging（桥接） | 隐式指代 | "I bought a car. The wheels were bad."（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 的 F1 分数平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) —— 经典教科书章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) —— 基于 span 的端到端方法。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) —— 提升指代消解的预训练方法。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) —— 基准数据集。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) —— 基于规则的经典方法。
