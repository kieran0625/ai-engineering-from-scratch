# 词嵌入 —— 从零实现 Word2Vec

> 一个词的意义由它的上下文决定。在这个想法上训练一个浅层网络，几何结构就会自然涌现。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 02 (词袋模型 + TF-IDF), Phase 3 · 03 (从零实现反向传播)
**时间：** ~75 分钟

## 问题所在

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。但它不知道这两个词意思几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列举同义词来掩盖这个问题，但这在罕见术语、领域行话以及你未预料到的每种语言上都会失败。

你想要一种表示方式，让 `dog` 和 `puppy` 在空间中距离很近。让 `king - man + woman` 靠近 `queen`。让在 `dog` 上训练的模型能自动将部分信号迁移到 `puppy`。

Word2Vec 给了我们这样的空间。两层神经网络，万亿 token 的训练规模，2013 年发表。架构简单到近乎令人尴尬。结果却重塑了 NLP 十年。

## 核心概念

**分布假说** (Firth, 1957)："You shall know a word by the company it keeps." 如果两个词出现在相似的上下文中，它们可能意思相近。

Word2Vec 有两种变体，都利用了这一思想。

- **Skip-gram。** 给定中心词，预测周围的词。`cat -> (the, sat, on)`，窗口大小为 2。
- **CBOW (continuous bag of words，连续词袋)。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词效果更好。它成为了默认选择。

网络有一个隐藏层，没有非线性激活。输入是词汇表上的 one-hot 向量。输出是词汇表上的 softmax。训练完成后，丢弃输出层。隐藏层的权重就是嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

诀窍在于：100k 个词上的 softmax 计算成本过高。Word2Vec 使用**负采样**将其转化为二分类任务。预测"这个上下文词是否出现在这个中心词附近，是或否"。每个训练对只采样少量负例（非共现词），而不是计算整个词汇表的 softmax。

## 动手实现

### 步骤 1：从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每个 (中心词, 上下文词) 对都是一个正训练样本。

### 步骤 2：嵌入表

两个矩阵。`W` 是中心词嵌入表（保留的那个）。`W'` 是上下文词表（通常丢弃，有时与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小随机初始化。词汇量 10k、维度 100 是现实的设置；教学用途下，50 词汇量 × 16 维度足以观察几何结构。

### 步骤 3：负采样目标函数

对每个正样本对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负例。训练模型使得 `W[center] · W'[context]` 的点积在正例上很高，在负例上很低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：正样本对的 logistic 损失（希望 sigmoid 接近 1）加上负样本对的 logistic 损失（希望 sigmoid 接近 0）。梯度流向两个表。完整推导见原始论文；如果你想真正理解，拿纸笔推导一遍。

### 步骤 4：在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上训练足够轮次后，共享上下文的词会有相似的中心嵌入。在玩具语料库上，效果微弱。在数十亿 token 上，效果惊人。

### 步骤 5：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是王室。而是因为向量 `(king - man)` 捕捉了类似"王室的"含义，将其加到 `woman` 上就会落在王室女性的区域附近。

## 实际应用

从零写 Word2Vec 是教学。生产环境的 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

实际工作中，你几乎不会自己训练 Word2Vec。你会下载预训练向量。

- **GloVe** —— Stanford 的共现矩阵分解方法。50d、100d、200d、300d 检查点。通用覆盖好。第 04 课专门讲 GloVe。
- **fastText** —— Facebook 的 Word2Vec 扩展，嵌入字符 n-gram。通过组合子词处理未登录词。第 04 课。
- **Google News 预训练 Word2Vec** —— 300d，3M 词汇量，2013 年发布。至今每天仍有下载。

### Word2Vec 在 2026 年仍然胜出的场景

- 轻量级领域特定检索。在笔记本电脑上一小时内在医学摘要上训练，获得通用模型无法捕捉的专业向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它，得到性别中立轴。仍在公平性研究中使用。
- 可解释性。100 维足够小，可以通过 PCA 或 t-SNE 绘图，实际观察到聚类形成。
- 任何必须在无 GPU 设备上运行推理的场景。Word2Vec 查询就是单行读取。

### Word2Vec 的局限

多义词壁垒。`bank` 只有一个向量。`river bank` 和 `financial bank` 共享它。`table`（电子表格 vs. 家具）也共享它。下游分类器无法从向量中区分不同含义。

上下文嵌入（ELMo、BERT 及之后的每个 transformer）通过基于周围上下文为每个词的出现产生不同向量解决了这个问题。这就是从 Word2Vec 到 BERT 的飞跃：从静态到上下文。Phase 7 涵盖 transformer 部分。

未登录词问题是另一个失败点。如果 `Zoomer-approved` 不在训练数据中，Word2Vec 就从未见过它。没有回退机制。fastText 用子词组合修复了这个问题（第 04 课）。

## 交付

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习

1. **简单。** 在小型语料库（20 句关于猫和狗的句子）上运行训练循环。200 轮后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的前 3 个结果包含 `dog`。如果没有，增加轮数或词汇量。
2. **中等。** 添加高频词子采样。频率高于 `10^-5` 的词按与其频率成比例的概率从训练对中丢弃。测量对罕见词相似性的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两个偏置轴：`he - she` 和 `doctor - nurse`。将职业词投影到两个轴上。报告哪些职业的偏置差距最大。这是公平性研究者常用的探测方法。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Word embedding（词嵌入） | 词变成向量 | 从上下文学习得到的密集低维（通常 100-300 维）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| Negative sampling（负采样） | 训练捷径 | 用与 `k` 个随机词的二分类替代完整词汇表的 softmax。 |
| Static embedding（静态嵌入） | 一词一向量 | 无论上下文如何，同一词始终使用相同向量。多义词上失效。 |
| Contextual embedding（上下文嵌入） | 上下文敏感向量 | 根据周围词为每次出现产生不同向量。transformer 的输出。 |
| OOV | Out of vocabulary（未登录词） | 训练中未见过的词。Word2Vec 无法为这些词产生向量。 |

## 延伸阅读

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) —— 负采样论文。简短易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) —— 最清晰的梯度推导，如果原始论文的数学让你感到密集。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) —— 实际有效的生产训练设置。
