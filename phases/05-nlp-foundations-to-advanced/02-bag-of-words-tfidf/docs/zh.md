# 词袋模型、TF-IDF 与文本表示

> 先计数，后思考。到 2026 年，在定义明确的任务上，TF-IDF 仍然能打败嵌入模型。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 01（文本处理），Phase 2 · 02（从零实现线性回归）
**时间：** ~75 分钟

## 问题

模型需要数字。你手里的是字符串。

每个 NLP 流水线都必须回答同一个问题：如何将长度可变的 token 流转换为分类器可以消费的固定长度向量。这个领域给出的第一个答案是最笨但有效的那个：数词，造向量。

这个向量支撑的生产级 NLP 比任何嵌入模型都多。垃圾邮件过滤、主题分类、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、前十年的学术 NLP 基准测试。2026 年的从业者面对窄分类任务时仍然会首先想到它。它速度快、可解释，而且在“哪些词出现”起决定作用的任务上，往往与 4 亿参数的嵌入模型无法区分。

本节课从零构建词袋模型，然后是 TF-IDF。接着展示 scikit-learn 如何用三行代码做到同样的事。最后指出那个让你不得不转向嵌入模型的失效模式。

## 概念

**词袋模型（BoW）** 丢弃顺序。对每个文档，统计每个词汇词出现的次数。向量长度等于词汇表大小。位置 `i` 是词 `i` 的计数。

**TF-IDF** 对 BoW 重新加权。在每个文档中都出现的词信息量低，所以缩小其权重。在整个语料库中罕见但在单个文档中频繁出现的词是信号，所以放大其权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是词在文档中的词频，`df` 是文档频率（包含该词的文档数），`N` 是总文档数。`log` 使得无处不在的词的权重有界。

关键特性：两者都产生具有可解释轴的稀疏向量。你可以查看训练好的分类器的权重，读出哪些词推动文档朝向哪个类别。对于 768 维的 BERT 嵌入，你做不到这一点。

## 动手构建

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本节课的 `code/main.py` 使用简化的小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词索引 0 是在第一个文档中首次见到的词。惯例有所不同；scikit-learn 按字母顺序排序。

### 步骤 2：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词汇索引。条目 `[i][j]` 表示“词 `j` 在文档 `i` 中出现多少次”。文档 1 中 `cat` 出现两次，因为确实如此。文档 0 中 `ran` 出现零次，因为确实没有。

### 步骤 3：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得命名的平滑技巧。`(n+1)/(d+1)` 避免 `log(x/0)`。末尾的 `+1` 确保在每个文档中都出现的词 IDF 为 1（而非 0），与 scikit-learn 的默认设置一致。其他实现使用原始的 `log(N/df)`。两者都有效；平滑版本更友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三个文档中，所以其 IDF 低。`dog` 出现在一个文档中，所以其 IDF 高。向量是稀疏的（大多数条目很小），有区分度的词凸显出来。

### 步骤 5：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

如果不归一化，较长的文档会得到更大的向量并在相似度计算中占主导。L2 归一化将每个文档放到单位超球面上。行之间的余弦相似度现在就是点积。

## 使用它

scikit-learn 提供了生产级版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一步调用中完成分词、词汇表构建和 BoW。`TfidfVectorizer` 添加 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万文档，稠密版本放不进内存；在分类器要求稠密之前保持稀疏。

改变一切的参数：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元组。通常提升分类效果。 |
| `min_df=2` | 丢弃出现在少于 2 个文档中的词。在噪声数据上修剪词汇表。 |
| `max_df=0.95` | 丢弃出现在超过 95% 文档中的词。近似停用词移除，无需硬编码列表。 |
| `stop_words="english"` | scikit-learn 的内置停用词列表。任务相关——情感分析*不应*丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 替代原始 `tf`。当一个词在文档中重复多次时有用。 |

### TF-IDF 仍然获胜的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。词的出现与否才是关键；语义细微差别不重要。
- 低数据场景（数百标注样本）。TF-IDF 加逻辑回归没有预训练成本。
- 任何对延迟敏感的地方。TF-IDF 加线性模型在微秒级响应。用 transformer 嵌入文档需要 10-100ms。
- 必须解释预测的系统。检查分类器系数。最正向的词就是原因。

### TF-IDF 失效的场景

语义失明失效。考虑这两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是负面评价。一个是正面的。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。词袋分类器必须记住词 `not` 靠近 `good` 时标签会翻转。数据足够多时它能学会，但永远不如理解句法的模型优雅。

另一个失效：推理时的未登录词。在 IMDb 评论上训练的 BoW 模型不知道如何处理 `Zoomer-approved`，如果该 token 从未在训练中出现。子词嵌入（第 04 课）能处理这个。TF-IDF 不能。

### 混合方案：TF-IDF 加权嵌入

2026 年中数据分类的务实默认方案：用 TF-IDF 权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入获得语义能力，从 TF-IDF 获得罕见词强调。分类器在池化后的向量上训练。在情感、主题和意图分类中，对于约 5 万标注样本以下的场景，这单独使用任一方都表现更好。

## 部署

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇完全不重叠的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产生 `n`-gram 的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上是否产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 使用 GloVe 100d 向量（下载一次，缓存）构建上述 TF-IDF 加权嵌入混合方案。在 20 Newsgroups 数据集上比较与纯 TF-IDF 和纯均值池化嵌入的分类准确率。报告哪种方案在何处获胜。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BoW | 词频向量 | 单个文档中词汇词的计数。丢弃顺序。 |
| TF | 词频 | 词在文档中的计数，可选按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词一次的文档数。 |
| IDF | 逆文档频率 | `log(N / df)` 平滑。降低到处都出现的词的权重。 |
| 稀疏向量 | 大部分是零 | 词汇表通常为 10k-100k 词；大多数词在任何给定文档中都不出现。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 表示完全相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn — 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 权威 API 参考，以及每个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 让 TF-IDF 成为十年默认方法的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年关于旧方法何时获胜及原因的见解。
