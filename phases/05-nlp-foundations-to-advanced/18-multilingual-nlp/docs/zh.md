# 多语言 NLP

> 一个模型，100 多种语言，其中大多数无需训练数据。跨语言迁移是 2020 年代的实际奇迹。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 04（GloVe、FastText、子词），Phase 5 · 11（机器翻译）
**时间：** ~45 分钟

## 问题

英语拥有数十亿标注样本。乌尔都语只有数千。迈蒂利语几乎为零。任何面向全球用户的实用 NLP 系统都必须能在长尾语言上工作，而这些语言往往缺乏任务特定的训练数据。

多语言模型通过同时训练一个模型来解决这个问题。共享的表示让模型能够将高资源语言学到的技能迁移到低资源语言。用英语情感分析数据微调模型，它就能直接对乌尔都语做出相当不错的情感预测。这就是零样本跨语言迁移，它已经重塑了 NLP 向世界交付的方式。

本课将介绍其中的权衡、经典模型，以及新手团队在多语言工作中最容易踩坑的一个决策：选择迁移的源语言。

## 概念

![通过共享多语言嵌入空间进行跨语言迁移](../assets/multilingual.svg)

**共享词表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表是共享的：相同的子词单元在相关语言中代表相同的词素。英语和意大利语中的 `anti-` 获得相同的 token。

**共享表示。** 在多种语言上进行掩码语言建模预训练的 transformer 学习到，不同语言中语义相似的句子会产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这一特性。英语中 "cat" 的嵌入与法语中 "chat" 和西班牙语中 "gato" 的嵌入聚类在一起，整句嵌入也是如此。

**零样本迁移。** 用一种语言（通常是英语）的标注数据微调模型。推理时，在模型支持的任何其他语言上运行。不需要目标语言的标注。对于类型学相关的语言效果较好，对于差异大的语言效果较弱。

**少样本微调。** 添加 100-500 条目标语言的标注样本。分类任务的准确率跃升至英语基线的 95-98%。这是多语言 NLP 中性价比最高的单一手段。

## 模型

| 模型 | 年份 | 覆盖范围 | 说明 |
|------|------|----------|------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。首个实用的多语言语言模型。低资源语言表现弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（远大于 Wikipedia）。设定跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R 配备 1M token 词表（对比 250k）。低资源语言表现更好。 |
| mT5 | 2020 | 101 种语言 | T5 架构的多语言生成模型。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开放 176B 多语言训练 LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。阿拉伯语、印地语、斯瓦希里语表现强劲。 |

按用例选择。分类任务用 XLM-R-base 作为合理的默认选择。生成任务根据翻译还是开放生成选择 mT5 或 NLLB。LLM 风格的工作配合 Aya-23 或 Claude，使用显式的多语言提示。

## 源语言决策（2026 年研究）

大多数团队默认用英语作为微调源语言。近期研究（2026）表明这往往是错的。

语言相似度比原始语料规模更能预测迁移质量。对于斯拉夫语系目标，德语或俄语往往胜过英语。对于印度语系目标，印地语往往胜过英语。**qWALS** 相似度指标（2026，基于世界语言结构地图集特征）对此进行了量化。**LANGRANK**（Lin 等，ACL 2019）是另一个更早的方法，结合语言相似度、语料规模和亲缘关系对候选源语言进行排序。

实用规则：如果你的目标语言有类型学相近的高资源亲缘语言，先尝试在该语言上微调，然后与英语微调对比。

## 动手实践

### 步骤 1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。在 NLI 数据上训练的 XLM-R 通过蕴含技巧很好地迁移到分类任务。

### 步骤 2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在嵌入空间中距离很近。不同的英语句子距离更远。这正是跨语言检索、聚类和相似度计算得以工作的基础。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认选择。过高的学习率会导致多语言对齐崩溃，最终得到一个仅支持英语的模型。

## 真正有效的评估

- **各语言在留出集上的准确率。** 不要聚合。聚合会隐藏长尾问题。
- **与单语基线对比。** 对于数据充足的语言，从头训练的单语模型有时能击败多语言模型。要测试。
- **实体级测试。** 目标语言的命名实体。多语言模型对于远离拉丁字母的文字往往分词能力较弱。
- **跨语言一致性。** 相同含义的两种语言应产生相同预测。测量差距。

## 应用

2026 年技术栈：

| 任务 | 推荐方案 |
|-----|----------|
| 分类，100 种语言 | XLM-R-base (~270M) 微调 |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上进行领域特定微调 |

如果性能重要，务必为微调目标语言预留预算。零样本是起点，不是最终答案。

### 分词税（低资源语言的问题所在）

多语言模型在所有语言间共享一个分词器。该词表在主要由英语、法语、西班牙语、汉语、德语主导的语料上训练。对于任何不在主导集合中的语言，三种税悄然叠加：

- **生育税。** 低资源语言的文本分词后每个词产生的 token 远多于英语。一个印地语句子可能需要同等英语句子 3-5 倍的 token。这 3-5 倍消耗你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、变音符号变体、Unicode 规范化不匹配或大小写变化，在嵌入空间中都成为冷启动的不相关序列。模型无法学习母语者视为显然的正字法对应关系。
- **容量溢出税。** 税 1 和税 2 消耗上下文位置、层深度和嵌入维度。剩余给实际推理的资源系统性地少于高资源语言从同一模型获得的资源。

实际症状：你的模型在印地语上训练正常，损失曲线看起来对，评估困惑度看起来合理，但生产输出却微妙地错误。形态学在句中崩溃。罕见屈折无法恢复。**你无法通过数据规模扩张来摆脱一个坏的分词器。**

缓解措施：选择对目标语言覆盖良好的分词器（XLM-V 的 1M token 词表是直接修复）；训练前在目标语言留出文本上验证分词生育率；对于真正的长尾文字使用字节级回退（SentencePiece `byte_fallback=True`、GPT-2 风格的字节级 BPE），确保没有任何东西成为 OOV。

## 部署

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语各 10 个句子上运行零样本分类流水线。报告每种语言的准确率。你应该看到法语表现强劲，印地语尚可，阿拉伯语不稳定。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言语料库上构建跨语言检索器。用英语查询，检索任意语言的文档。测量 recall@5。
3. **困难。** 对比英语源和印地语源微调在印地语分类任务上的表现。两种方案下各用 500 条目标语言样本进行少样本微调。报告哪种源语言产生更好的印地语准确率，差距多少。这是 LANGRANK 论题的微缩版。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| 多语言模型 | 一个模型，多种语言 | 跨语言共享词表和参数。 |
| 跨语言迁移 | 一种语言训练，另一种语言运行 | 在源语言微调，在目标语言评估，无需目标语言标注。 |
| 零样本 | 无目标语言标注 | 无需在目标语言上微调即可迁移。 |
| 少样本 | 少量目标标注 | 100-500 条目标语言样本用于微调。 |
| mBERT | 首个多语言语言模型 | 在 Wikipedia 上预训练的 104 语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言机器翻译 | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau 等 (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究线的分析论文。
- [Costa-jussà 等 (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün 等 (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文。
