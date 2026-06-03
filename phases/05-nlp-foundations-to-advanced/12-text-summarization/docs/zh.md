# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者想表达什么。不同的任务，不同的陷阱。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 02（词袋模型 + TF-IDF），Phase 5 · 11（机器翻译）
**时间：** ~75 分钟

## 问题

一篇 2000 字的新闻文章出现在你的信息流中。你需要 120 字来概括它。你可以从文章中挑选三个最重要的句子（抽取式），或者用自己的话重写内容（生成式）。两者都称为摘要。它们是完全不同的问题。

抽取式摘要是一个排序问题。为每个句子打分，返回前 `k` 个。输出总是符合语法的，因为它是逐字提取的。风险在于可能遗漏分散在文章各处的信息。

生成式摘要是一个生成问题。Transformer 基于输入产生新文本。输出流畅且压缩，但可能产生源文中不存在的事实幻觉。风险在于自信的虚构。

本节课将构建两种方法，并展示各自固有的失败模式。

## 概念

![抽取式 TextRank 与生成式 Transformer 对比](../assets/summarization.svg)

**抽取式。** 将文章视为图，节点是句子，边是相似度。在图上运行 PageRank（或类似算法），根据句子与其他内容的连接程度来评分。得分最高的句子构成摘要。经典实现是 **TextRank**（Mihalcea 和 Tarau, 2004）。

**生成式。** 在文档-摘要对上微调 Transformer 编码器-解码器（BART、T5、Pegasus）。推理时，模型通过交叉注意力逐 token 读取文档并生成摘要。Pegasus 特别使用了间隙句子预训练目标，使其在无需大量微调的情况下就能出色地完成摘要任务。

使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）进行评估。ROUGE-1 和 ROUGE-2 对一元组和二元组重叠进行评分。ROUGE-L 对最长公共子序列进行评分。分数越高越好，但 40 的 ROUGE-L 算"好"，50 算"优秀"。每篇论文都会报告这三个指标。使用 `rouge-score` 包。

## 构建

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两点值得说明。相似度函数使用对数归一化的词重叠，这是 TextRank 的原始变体。TF-IDF 向量的余弦相似度也有效。阻尼因子 0.85 和迭代次数是 PageRank 的默认值。

### 步骤 2：使用 BART 进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了微调。它能直接生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），使用相应的 Pegasus 检查点或在目标数据上进行微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取。没有它，"running" 和 "run" 会被视为不同的词，导致 ROUGE 低估。

### 超越 ROUGE（2026 年摘要评估）

ROUGE 二十年来一直是主导的摘要指标，但在 2026 年单独使用已不足够。一项对 NLG 论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）在 2023 年后逐渐普及，现在大多数摘要论文都会与 ROUGE 一起报告。
- **BARTScore** 将评估视为生成任务：通过预训练 BART 给定源文时赋予摘要的概率来评分。
- **MoverScore**（基于上下文嵌入的 Earth Mover 距离）在 2025 年摘要基准中登顶，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **基于问答的忠实度** 在 2021-2023 年很常见，现在常被 **G-Eval**（一种 GPT-4 提示链，通过思维链推理对连贯性、一致性、流畅性、相关性进行评分）取代。
- **G-Eval** 和类似的 LLM 评判方法在评分标准设计良好时，约 80% 的时间与人类判断一致。

生产建议：报告 ROUGE-L 用于遗留对比，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。用 50-100 个人工标注的摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险低得多，因为输出是从源文逐字提取的，但如果源句被断章取义、过时或顺序错乱，仍可能产生误导。这是生产系统仍偏好对合规敏感内容使用抽取式方法的最大原因。

需要了解的幻觉类型：

- **实体替换。** 源文说"John Smith"，摘要说"John Brown"。
- **数字漂移。** 源文说"25,000"，摘要说"25 million"。
- **极性翻转。** 源文说"rejected the offer"，摘要说"accepted the offer"。
- **事实捏造。** 源文未提及 CEO，摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在源句和摘要句之间进行蕴含关系训练的二分类器。预测是否事实。
- **基于问答的事实性。** 向 QA 模型提问，答案应在源文中。如果摘要支持不同的答案，则标记。
- **实体级 F1。** 比较源文和摘要中的命名实体。仅出现在摘要中的实体可疑。

对于任何面向用户且事实性重要的场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式需要在循环中进行事实性检查。

## 应用

2026 年技术栈：

| 应用场景 | 推荐方案 |
|---------|---------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或微调后的 T5 |
| 多文档，长文本 | 任何支持 32k+ 上下文的 LLM，通过提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，低幻觉风险，可保证 | TextRank 或 `sumy` 的 LSA / LexRank |

在计算资源不受限制时，2026 年长上下文 LLM 通常优于专用模型。权衡在于成本和可重复性；专用模型的输出更一致。

## 交付

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 对 5 篇新闻文章运行 TextRank。将前 3 个句子与参考摘要对比。测量 ROUGE-L。在 CNN/DailyMail 风格的文章上，你应该能看到 30-45 的 ROUGE-L。
2. **中等。** 实现实体级事实性：从源文和摘要中提取命名实体（spaCy），计算源文实体在摘要中的召回率以及摘要实体相对于源文的精确率。高精确率低召回意味着安全但简短；低精确率意味着存在幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上对比 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）和每条摘要的成本。记录各自的优势场景。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Extractive | 挑选句子 | 从源文中逐字返回句子。不会产生幻觉。 |
| Abstractive | 重写 | 基于源文生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的 N-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上运行 PageRank。 |
| Factuality | 是否正确 | 摘要中的主张是否得到源文支持。 |
| Hallucination | 编造内容 | 摘要中存在源文不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 与间隙句子目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性研究综述论文。
