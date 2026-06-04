# Transformer 之前的文本生成 —— N-gram 语言模型

> 如果某个词令人惊讶，说明模型很差。困惑度（Perplexity）将“惊讶”量化为数值。平滑技术使其保持有限。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 01（文本处理）、Phase 2 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题所在

在 Transformer、RNN 和词嵌入出现之前，语言模型通过统计前 `n-1` 个词之后接某个词的频率来预测下一个词。例如统计 "the cat" → "sat" 出现 47 次，"the cat" → "jumped" 出现 12 次，"the cat" → "refrigerator" 出现 0 次。将其归一化即可得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，它驱动了所有的语音识别器、拼写检查器和基于短语的机器翻译系统。当你需要低成本的设备端语言建模时，它依然在被使用。

有趣的问题在于如何处理未见过（unseen）的 n-gram。基于原始计数的模型会给任何未见过的事物分配零概率，这非常致命，因为句子通常很长，而几乎每个长句子都至少包含一个训练集中未出现的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑便是其成果，现代深度学习也继承了它的经验主义传统。

## 核心概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。设定 `n`（三元组通常为 3，四元组通常为 4）。根据计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练集中未出现的任何 n-gram 都会获得零概率。一项针对 Brown 语料库的 2007 年研究发现，即使是 4-gram 模型，也有 30% 的预留测试集 4-gram 在训练中从未出现过。如果不进行平滑处理，你无法对任何真实文本进行评估。

**平滑方法（按复杂程度排序）：**

1. **拉普拉斯平滑（加一平滑）。** 给所有计数加 1。简单，但对罕见事件效果极差。
2. **Good-Turing 平滑。** 基于“频率的频率”，将高频率事件的概率质量重新分配给未见事件。
3. **插值法。** 结合 n-gram、(n-1)-gram 等估计值，并使用可调节的权重进行组合。
4. **回退法。** 如果 n-gram 计数为零，则回退到 (n-1)-gram。Katz 回退法对此进行了规范化。
5. **绝对折扣法。** 从所有计数中减去固定折扣值 `D`，并将这部分概率重新分配给未见事件。
6. **Kneser-Ney 平滑。** 绝对折扣法加上对低阶模型的巧妙设计：使用*延续概率*（一个词出现在多少种不同上下文中）而非原始频率。

Kneser-Ney 的洞察非常深刻。“San Francisco”是一个常见的大词组（bigram）。一元词（unigram）“Francisco”大多出现在“San”之后。朴素的绝对折扣法会赋予“Francisco”较高的一元词概率（因为它的总计数很高）。但 Kneser-Ney 注意到“Francisco”只出现在一种上下文中，因此相应地降低了它的延续概率。结果：以“Francisco”结尾的新颖 bigram 会得到恰当的较低概率。

**评估指标：困惑度（Perplexity）。** 在预留测试集上，每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型的不确定性等同于在 100 个词中均匀随机选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手实现

### 步骤 1：三元组计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是标记化句子的列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 表示句子边界。

### 步骤 2：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给所有计数加 1。虽然实现了平滑，但会向未见事件过度分配概率质量，同时也损害了已知罕见事件的概率。

### 步骤 3：Kneser-Ney（二元组，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

包含三个核心部分。`continuation_prob` 捕捉“这个词出现在多少种不同的上下文中？”（这是 Kneser-Ney 的创新点）。`lambda_prev` 是折扣释放出的概率质量，用于加权回退项。最终概率等于折扣后的主项加上加权的延续项。

### 步骤 4：使用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例进行采样。每次运行种子不同总会产生不同的输出。若要获得类似束搜索（beam search）的输出，可在每一步选择 argmax（贪心策略），并加入一个小随机性旋钮（温度参数 temperature）。

### 步骤 5：计算困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，经过良好调优的 4-gram KN 模型困惑度约为 140。Transformer 语言模型在同一测试集上的困惑度为 15-30。差距约为 10 倍。正是这个差距推动了整个领域向前发展。

## 应用场景

- **经典 NLP 教学。** 让你最直观地理解平滑、最大似然估计（MLE）和困惑度的途径。
- **KenLM。** 生产级 n-gram 库。在语音识别和机器翻译系统中用作重排序器（rescorer），适用于对延迟敏感的场景。
- **设备端自动补全。** 键盘中的三元组模型。至今仍在广泛使用。
- **基线对比。** 在宣称你的神经语言模型表现良好之前，务必先计算 n-gram 语言模型的困惑度作为基线。如果你的 Transformer 没有大幅超越 KN 模型，那说明出了问题。

## 交付代码

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在包含 1000 个句子的莎士比亚语料库上训练一个三元组语言模型。生成 20 个句子。它们会在局部看起来合理，但整体缺乏连贯性。这是该领域的标准演示案例。
2. **中等。** 在预留的莎士比亚数据集划分上为你的 KN 模型实现困惑度计算。与拉普拉斯平滑进行对比。你应该能看到 KN 将困惑度降低了 30%-50%。
3. **困难。** 构建一个三元组拼写纠错器：给定一个拼写错误的词及其上下文，生成纠错候选并根据语言模型下的上下文概率进行排序。在公开的 Birkbeck 拼写语料库上进行评估。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | `n` 个连续 token 组成的序列。 |
| Smoothing | 避免零概率 | 重新分配概率质量，使未见事件获得非零概率。 |
| Perplexity | 语言模型质量指标 | 在预留数据上的 `exp(-average log-prob)`。越低越好。 |
| Backoff | 回退到更短上下文 | 如果三元组计数为零，则使用二元组。Katz 回退法对此进行了形式化定义。 |
| Kneser-Ney | n-gram 最佳平滑方法 | 绝对折扣法 + 低阶模型的延续概率。 |
| Continuation probability | KN 特有概念 | `P(w)` 根据 `w` 出现的上下文数量进行加权，而非原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) —— n-gram 语言模型和平滑技术的权威论述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) —— 确立 Kneser-Ney 为最佳 n-gram 平滑方法的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) —— KN 平滑的原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) —— 快速的生产级 n-gram 语言模型，截至 2026 年仍广泛用于对延迟敏感的应用场景。
