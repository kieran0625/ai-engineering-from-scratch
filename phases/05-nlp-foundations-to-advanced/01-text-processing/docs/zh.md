# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**类型：** Build
**语言：** Python
**前置知识：** Phase 2 · 14（朴素贝叶斯）
**时间：** ~45 分钟

## 问题

模型读不懂 "The cats were running."。它读的是整数。

每个 NLP 系统开篇都要面对三个问题。单词从哪里开始。单词的词根是什么。如何在需要时将 "run"、"running"、"ran" 视为同一个词，在不需要时视为不同的词。

分词出错，模型就从垃圾中学习。如果你的分词器将 `don't` 视为一个 token，但将 `do n't` 视为两个，训练分布就会分裂。如果你的词干提取器将 `organization` 和 `organ` 压缩为同一个词干，主题建模就会失效。如果你的词形还原器需要词性上下文但你没有传入，动词就会被当作名词处理。

本节课从零构建三个预处理原语，然后展示 NLTK 和 spaCy 如何做同样的工作，让你看清其中的权衡。

## 概念

三个操作。每个都有职责和失效模式。

**分词（Tokenization）** 将字符串切分为 token。"Token" 故意模糊，因为正确的粒度取决于任务。词级别用于经典 NLP。子词级别用于 transformers。字符级别用于没有空格的语言。

**词干提取（Stemming）** 用规则砍掉后缀。快、激进、粗糙。`running -> run`。`organization -> organ`。第二个就是失效模式。

**词形还原（Lemmatization）** 利用语法知识将单词还原为词典形式。较慢、精确、需要查表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。当速度重要且能容忍噪声时使用词干提取（搜索索引、粗略分类）。当语义重要时使用词形还原（问答、语义搜索、任何用户会阅读的内容）。

## 动手实现

### 步骤 1：正则表达式分词器

最简单实用的分词器按非字母数字字符切分，同时将标点符号保留为独立的 token。不完美，不完整，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排序。带可选内部撇号的单词（`don't`、`it's`）。纯数字。任何单个非空白非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失效模式。`3pm` 被切分为 `['3', 'pm']`，因为我们在字母串和数字串之间交替。对大多数任务来说够用了。URL、邮箱、话题标签都会失效。生产环境中，在通用模式之前添加特定模式。

### 步骤 2：Porter 词干提取器（仅步骤 1a）

完整的 Porter 算法有五阶段规则。仅步骤 1a 就覆盖了最常见的英语后缀，足以展示其模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

自上而下阅读规则。`ies -> i` 规则解释了为什么 `ponies -> poni`，而不是 `pony`。真正的 Porter 有步骤 1b 会修复它。规则相互竞争。前面的规则优先。顺序比任何单条规则都重要。

### 步骤 3：基于查表的词形还原器

真正的词形还原需要形态学。一个适合教学的简化版本使用小型词元表加回退策略。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键的教学时刻。`watched` 不在我们的表中，而我们的回退只处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较级形容词、音变复数（`children -> child`）。这就是生产系统使用 WordNet、spaCy 的形态分析器或完整形态分析器的原因。

### 步骤 4：将它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的环节是词性标注器。Phase 5 · 07（POS 标注）会构建一个。目前，将所有内容默认为 `NOUN`，并承认这一局限性。

## 使用现成工具

NLTK 和 spaCy 提供了生产级版本。各需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩写形式、Unicode、你的正则表达式遗漏的边界情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将词性标签从 NLTK 的 Penn Treebank 体系转换为 WordNet 的缩写集合。上面的转换连接代码是大多数教程跳过的地方。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流程隐藏在 `nlp(text)` 后面。分词、词性标注、词形还原全部运行。大规模下比 NLTK 更快。开箱即用更准确。代价是你无法轻易替换单个组件。

### 如何选择

| 场景 | 选择 |
|------|------|
| 教学、研究、替换组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer 流程（反正你会用模型的分词器） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人提醒你的两种失效模式

大多数教程教完算法就结束。有两件事会咬到真正的预处理流程，而且几乎从未被提及。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原行为。spaCy 2.x 产生 `['do', "n't"]` 的地方，3.x 可能产生 `["don't"]`。你的模型在一个分布上训练。推理现在在另一个分布上运行。准确率悄然下降，没人知道为什么。在 `requirements.txt` 中固定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级时运行它。

**训练/推理不匹配。** 训练时使用激进的预处理（小写、停用词移除、词干提取），部署时处理原始用户输入，看着性能暴跌。这是生产 NLP 中最常见的失败。如果你在训练时预处理，推理时必须运行完全相同的函数。将预处理作为函数打包在模型包内部，而不是作为服务团队重写的 notebook 单元格。

## 交付

一个可复用的提示，帮助工程师无需阅读三本教科书就能选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize` 以将 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")` 应产生一个 URL token。
2. **中等。** 实现 Porter 步骤 1b。如果单词包含元音且以 `ed` 或 `ing` 结尾，则移除它。处理双辅音规则（`hopping -> hop`，而非 `hopp`）。
3. **困难。** 构建一个词形还原器，使用 WordNet 作为查表，但在 WordNet 无条目时回退到你的 Porter 词干提取器。在标注语料库上测量准确率，与纯 WordNet 和纯 Porter 对比。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Token | 一个词 | 模型消费的任何单位。可以是词、子词、字符或字节。 |
| Stem | 词根 | 基于规则的后缀剥离结果。不一定是真正的单词。 |
| Lemma | 词典形式 | 你会去查的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | 如 NOUN、VERB、ADJ 的类别。词形还原需要它才能准确。 |
| Morphology | 词形规则 | 单词如何根据时态、数、格改变形式。词形还原依赖它。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真正的流程如何连接。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边界情况。
