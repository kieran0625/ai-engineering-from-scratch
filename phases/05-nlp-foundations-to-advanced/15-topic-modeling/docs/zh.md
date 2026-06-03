# 主题建模 — LDA 与 BERTopic

> LDA：文档是主题的混合体，主题是词语的分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。目标相同，底层原语不同。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 02（词袋模型 + TF-IDF），Phase 5 · 03（Word2Vec）
**时间：** ~45 分钟

## 问题

你有 10,000 条客户支持工单、50,000 篇新闻文章或 200,000 条推文。你需要在不逐篇阅读的情况下了解整个集合的内容。你没有标注的类别，甚至不知道存在多少类别。

主题建模无需监督即可回答这个问题。给它一个语料库，返回一组数量较少且语义连贯的主题，以及每篇文档在这些主题上的分布。

两大算法家族占据主导地位。LDA（2003）将每篇文档视为潜在主题的混合体，每个主题视为词语的分布。推断采用贝叶斯方法。它至今仍用于生产环境，特别是在需要混合成员主题分配和可解释的词语级概率分布时。

BERTopic（2020）使用 BERT 编码文档，用 UMAP 降维，用 HDBSCAN 聚类，并通过基于类别的 TF-IDF 提取主题词。它在短文本、社交媒体以及语义相似性比词语重叠更重要的任何场景下表现优异。一篇文档对应一个主题，这对长文本内容而言是一个局限。

本节课将建立对两者的直觉，并指明针对给定语料库应选择哪一种。

## 概念

![LDA 混合模型与 BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 生成故事。** 每个主题是词语的分布。每篇文档是主题的混合体。要生成文档中的一个词，先从文档的主题混合中采样一个主题，再从该主题的分布中采样一个词。推断则反向进行：给定观测到的词语，推断每篇文档的主题分布和每个主题的词语分布。折叠吉布斯采样或变分贝叶斯负责数学计算。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行求和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行求和为 1（主题的词语分布）。

**BERTopic 流程。**

1. 使用句子 Transformer 编码每篇文档（例如，`all-MiniLM-L6-v2`）。384 维向量。
2. 使用 UMAP 降维至约 5 维。BERT 嵌入维度过高，不适合直接聚类。
3. 使用 HDBSCAN 聚类。基于密度，产生可变大小的聚类和"离群"标签。
4. 对每个聚类，计算该聚类文档的基于类别的 TF-IDF，提取顶部词语。

输出为每篇文档一个主题（外加 -1 离群标签）。可选地，通过 HDBSCAN 的概率向量获得软成员关系。

## 动手实现

### 步骤 1：通过 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：去除停用词，min_df 和 max_df 过滤罕见和过于普遍的词项，使用 CountVectorizer（而非 TfidfVectorizer），因为 LDA 需要原始计数。

### 步骤 2：BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤去除了 BERTopic 的离群桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小；BERTopic 库的默认值是 10。本示例为配合课程规模显式设为 15。对于超过 10,000 篇文档的语料库，应增加至 50 或 100。

### 步骤 3：评估

两种方法都输出主题词。问题在于这些词语是否连贯。

- **主题连贯性（c_v）。** 结合滑动窗口上下文中顶部词对的 NPMI（归一化点互信息），将分数聚合为主题向量，并通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel` 配合 `coherence="c_v"`。
- **主题多样性。** 所有主题顶部词语中唯一词语的比例。越高越好（主题不重叠）。
- **定性检查。** 阅读每个主题的顶部词语。它们是否指代真实事物？人工判断仍是最后一道防线。

## 如何选择

| 场景 | 选择 |
|------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 包含主题混合的长文档 | LDA |
| 无 GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| 用于主题标注的 LLM 集成 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大化语义连贯性 | BERTopic |

最大的实际考量是文档长度。BERT 嵌入会截断；LDA 计数适用于任意长度。对于超过嵌入模型上下文长度的文档，要么分块 + 聚合，要么使用 LDA。

## 应用

2026 年技术栈：

- **BERTopic。** 短文本和语义重要的场景下的默认选择。
- **`gensim.models.LdaModel`。** 生产环境的经典 LDA，成熟、久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 实验用的简易 LDA。
- **NMF。** 非负矩阵分解。LDA 的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 设计与 BERTopic 类似。社区较小，但在部分基准测试中表现良好。
- **FASTopic。** 较新，在超大规模语料库上比 BERTopic 更快。
- **基于 LLM 的标注。** 运行任意聚类，然后提示模型为每个聚类命名。

## 部署

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA。打印每个主题的顶部 10 个词。手工标注每个主题。算法是否找到了真实类别？
2. **中等。** 在同一 20 Newsgroups 子集上拟合 BERTopic。比较发现的主题数量、顶部词语和定性连贯性与 LDA 的差异。哪种方法更清晰地呈现了真实类别？
3. **困难。** 在你的语料库上计算 LDA 和 BERTopic 的 c_v 连贯性。分别用 5、10、20、50 个主题运行。绘制连贯性 vs 主题数量的曲线。报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 主题（Topic） | 语料库所涉及的事物 | 词语的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| 混合成员（Mixed membership） | 文档属于多个主题 | LDA 为每篇文档分配在所有主题上的分布。 |
| UMAP | 降维 | 保持局部结构的流形学习；用于 BERTopic。 |
| HDBSCAN | 密度聚类 | 发现可变大小的聚类；为离群点生成"噪声"标签（-1）。 |
| c_v 连贯性 | 主题质量指标 | 滑动窗口内顶部主题词语的平均点互信息。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考文档。示例丰富。
