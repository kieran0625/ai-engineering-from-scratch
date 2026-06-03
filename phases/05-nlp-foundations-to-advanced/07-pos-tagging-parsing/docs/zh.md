# 词性标注与句法分析

> 语法曾一度不受待见。后来每个 LLM 流水线都需要验证结构化抽取，它又回来了。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 01（文本处理），Phase 2 · 14（朴素贝叶斯）
**时间：** ~45 分钟

## 问题

第 01 课曾承诺，词形还原需要词性标注。如果不知道 `running` 是动词，词形还原器就无法将其还原为 `run`。如果不知道 `better` 是形容词，它也无法还原为 `good`。

这个承诺背后隐藏着一整个子领域。词性标注（POS tagging）为每个词分配语法类别。句法分析（syntactic parsing）还原句子的树状结构：哪个词修饰哪个，哪个动词支配哪些论元。经典 NLP 花了二十年时间精研两者。随后深度学习将它们压缩为基于预训练 transformer 的 token 分类任务，研究界便转向了别处。

但应用界没有。每个结构化抽取流水线仍在底层使用 POS 和依存树。LLM 生成的 JSON 要依据语法约束进行验证。问答系统利用依存分析来分解查询。机器翻译质量评估器检查分析树的对齐。

值得了解。本课介绍标注集、基线方法，以及你从何处停止从零实现、转而调用 spaCy。

## 概念

**词性标注** 为每个 token 标注语法类别。**Penn Treebank (PTB)** 标注集是英语的默认标准。36 个标签，包含 casual reader 觉得过于细分的区别：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，等等。**Universal Dependencies (UD)** 标注集更粗（17 个标签）且语言无关；它已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析** 生成一棵树。两种主要风格：

- **成分分析（Constituency parsing）。** 名词短语、动词短语、介词短语相互嵌套。输出是一棵非终结符类别（NP、VP、PP）的树，词作为叶子节点。
- **依存分析（Dependency parsing）。** 每个词有一个它所依存的核心词，并标注语法关系。输出是一棵树，每条边都是一个（核心词，依存词，关系）三元组。

依存分析在 2010 年代胜出，因为它能干净地跨语言泛化，尤其是自由语序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 动手实现

### 步骤 1：最频繁词性基线

最简陋但能工作的词性标注器。对每个词，预测它在训练集中出现最频繁的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线达到约 85% 的准确率。不算好，但任何严肃模型都不应跌破这个底线。

### 步骤 2：二元 HMM 标注器

对序列的联合概率建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（给定前一个标签的当前标签概率）和发射概率（给定标签的词概率）。用带 Laplace 平滑的计数估计两者。用 Viterbi 算法解码（在标签格子上进行动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 语料库上的二元 HMM 达到约 93% 准确率。从 85% 到 93% 的跃升主要来自转移概率——模型学到 `DET NOUN` 常见而 `NOUN DET` 罕见。

### 步骤 3：现代标注器为何能超越

转移概率 + 发射概率是局部的。它们无法捕捉 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie" 中是动词。引入任意特征的 CRF（后缀、词形、前后词、词本身）达到约 97%。BiLSTM-CRF 或 transformer 达到约 98%+。

这个任务的天花板由标注者分歧决定。人类标注者在 Penn Treebank 上的一致性约为 97%。超过 98% 的模型可能是在测试集上过拟合。

### 步骤 4：依存分析概览

从零完整实现依存分析超出本课范围；经典教材处理见 Jurafsky 和 Martin。需要了解的两类经典方法：

- **基于转移的分析器**（arc-eager、arc-standard）行为类似于移进-归约分析器：读取 token，将其移入栈，并应用创建弧的归约动作。贪心解码速度快。经典实现是 MaltParser。现代神经版本：Chen 和 Manning 的基于转移分析器。
- **基于图的分析器**（Eisner 算法、Dozat-Manning 双仿射）为每个可能的核心-依存边打分，并选取最大生成树。较慢但更准。

对于大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下到上阅读 `dep` 列，句子的语法结构便呈现出来。

## 应用

每个生产级 NLP 库都将 POS 和依存分析器作为标准流水线的一部分。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确，与分词 + NER + 词形还原集成。`token.tag_`（Penn）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP (stanza)**。Stanford 对 CoreNLP 的继任者。60 多种语言上的 state-of-the-art。
- **trankit**。基于 transformer，UD 准确率高。
- **NLTK**。`pos_tag`。可用，慢，较老。适合教学。

### 2026 年这仍然重要的场景

- **词形还原。** 第 01 课需要 POS 才能正确还原。始终如此。
- **LLM 输出的结构化抽取。** 验证生成的句子是否遵守语法约束（例如，主谓一致、必需的修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过分析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，支持对新语言进行零样本结构化分析。
- **低算力流水线。** 如果你无法部署 transformer，POS + 依存分析 + 地名词典能让你走得很远。

## 交付

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习

1. **简单。** 在小型标注语料库（如 NLTK 的 Brown 子集）上使用最频繁词性基线，在 held-out 句子上测量准确率。验证约 85% 的结果。
2. **中等。** 训练上述二元 HMM 并报告每类标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **困难。** 使用 spaCy 的依存分析从 1000 句样本中提取主语-动词-宾语三元组。在 50 个手动标注的三元组上评估。记录抽取失败的情况（通常是被动语态、并列结构和省略主语）。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| POS tag | 词的类别 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标注集 | 英语专用。细粒度的动词时态和名词数。 |
| Universal Dependencies | 多语言标注集 | 比 PTB 粗；语言中立；跨语言工作的默认选择。 |
| Dependency parse | 句子树 | 每个词有一个核心词，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 给定发射概率和转移概率，找出最高概率的标签序列。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 和分析的经典教材处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言分析器使用的跨语言标注集和树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个暴露属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经分析器带入主流的论文。
