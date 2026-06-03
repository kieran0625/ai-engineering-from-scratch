# 信息检索与搜索

> BM25 精确但脆弱。Dense 覆盖面广但会遗漏关键词。Hybrid 是 2026 年的默认选择。其余都是调优。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**时间：** ~75 分钟

## 问题

用户输入 "what happens if someone lies to get money"，期望找到实际涵盖该行为的法条："Section 420 IPC"。关键词搜索完全无法匹配（无共享词汇）。如果嵌入未在法语文本上训练，语义搜索也会失败。真实的搜索必须同时处理这两种情况。

IR 是每个 RAG 系统、每个搜索框、每个文档站点模糊查找背后的管道。2026 年能在生产环境运行的架构不是单一方法，而是一系列互补方法的链条，每个方法弥补前一个方法的失败。

本课构建每个组件，并指出每个组件能捕获哪些失败。

## 概念

![混合检索：BM25 + dense + RRF + cross-encoder 重排](../assets/retrieval.svg)

四层。按需选择。

1. **稀疏检索 (BM25)。** 快速，精确匹配能力强，语义理解差。基于倒排索引运行。百万级文档查询延迟低于 10ms。能准确找到法条引用、产品代码、错误信息、命名实体。
2. **稠密检索。** 将查询和文档编码为向量。最近邻搜索。捕获同义改写和语义相似性。会遗漏仅差一个字符的精确关键词匹配。使用 FAISS 或向量数据库时查询延迟 50-200ms。
3. **融合。** 合并稀疏和稠密检索的排名列表。倒数排名融合 (RRF) 是简单的默认选择，因为它忽略原始分数（处于不同尺度），仅使用排名位置。当已知某个信号在特定领域占主导时，加权融合是可选项。
4. **Cross-encoder 重排。** 取融合后的前 30 个结果。运行 cross-encoder（将查询和文档一起输入，为每对打分）。保留前 5 个。Cross-encoder 比 bi-encoder 每对更慢，但准确率高得多。通过仅在 top-30 上运行来摊销成本。

三路检索（BM25 + dense + 学习式稀疏如 SPLADE）在 2026 年基准测试中优于两路，但需要学习式稀疏索引的基础设施。对大多数团队而言，两路加 cross-encoder 重排是最佳平衡点。

## 构建

### 步骤 1：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。`k1=1.5` 控制词频饱和度；越高意味着越重视词项重复。`b=0.75` 控制长度归一化；0 忽略文档长度，1 完全归一化。默认值来自原始论文中 Robertson 的建议，很少需要调整。

### 步骤 2：使用 bi-encoder 进行稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入做 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 为 384 维，速度快，对大多数英文检索足够强。多语言工作使用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高准确率使用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。更高的 `k` 会 flatten 排名差异的贡献；更低的 `k` 使高排名占主导。60 是发表的默认值，很少需要调整。

### 步骤 4：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找到词汇匹配。Dense 找到语义匹配。RRF 合并两个排名，无需分数校准。Cross-encoder 对 top-30 使用查询-文档对一起重排，捕获 bi-encoder 遗漏的细粒度相关性。保留 top-5。

### 步骤 5：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 在存在正确文档的查询中，正确文档出现在 top-k 中的比例 |
| MRR (Mean Reciprocal Rank) | 首个相关文档排名的倒数平均值 |
| nDCG@k | 考虑相关性分级，而非仅二元的相关/不相关 |

针对 RAG 场景，检索器的 **Recall@k** 是最重要的数字。如果正确的段落不在检索结果中，reader 无法回答。

调试技巧：对于失败的查询，对比稀疏和稠密排名的差异。如果其中一个找到了正确文档而另一个没有，说明存在词汇不匹配（修复：补充缺失的部分）或语义歧义（修复：更好的嵌入或重排器）。

## 应用

2026 年技术栈：

| 规模 | 技术栈 |
|------|--------|
| 1k-100k 文档 | 内存 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需单独数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于稠密 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus 支持混合检索。Cross-encoder 重排 top-30。 |
| 最高质量前沿 | 三路 (BM25 + dense + SPLADE) + ColBERT 晚期交互重排 |

无论选择什么，都要为评估预留预算。在基准测试端到端 RAG 准确率之前，先基准测试检索召回率。Reader 无法修复检索器遗漏的内容。

### 2026 年生产 RAG 的宝贵经验

- **80% 的 RAG 失败源于 ingestion 和分块，而非模型。** 团队花费数周更换 LLM 和调整 prompt，而检索器每三个查询就悄悄返回错误上下文。先修复分块。
- **分块策略比分块大小更重要。** 固定大小拆分破坏表格、代码和嵌套标题。句子感知是默认选择；语义或基于 LLM 的分块对技术文档和产品手册有回报。
- **父文档模式。** 检索小的 "子" 块以提高精度。当同一父章节的多个子块出现时，替换为父块以保留上下文。这能持续提升回答质量，无需重新训练。
- **k_rerank=3 通常最优。** 超过该值的每个额外块都会增加 token 成本和生成延迟，而不会提升回答质量。如果 k=8 仍优于 k=3，说明重排器表现不佳。
- **HyDE / 查询扩展。** 从查询生成假设答案，嵌入该答案，然后检索。弥合短问题与长文档之间的措辞差距。无需训练即可免费提升精度。
- **上下文预算低于 8K token。** 持续触及该限制意味着重排器阈值过松。
- **版本化一切。** Prompt、分块规则、嵌入模型、重排器。任何漂移都会悄悄破坏回答质量。在忠实度、上下文精度和未回答问题率上设置 CI 门禁，在用户看到之前阻止退化。
- **三路检索 (BM25 + dense + 学习式稀疏如 SPLADE) 优于两路** 在 2026 年基准测试中，尤其对混合专有名词和语义的查询。基础设施支持 SPLADE 索引时上线。

据 2026 年行业测量，合理的检索设计可减少 70-90% 的幻觉。大多数 RAG 性能提升来自更好的检索，而非模型微调。

## 交付

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在 500 文档语料库上实现上述 `hybrid_search`。测试 20 个查询。对比 BM25-only、dense-only 和 hybrid 的 recall@5。
2. **中等。** 添加 MRR 计算。对每个已知正确文档的测试查询，找到正确文档在 BM25、dense 和 hybrid 排名中的位置。报告各自的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss (Sentence Transformers) 在领域数据上微调稠密编码器。从 500 个查询-文档对构建训练集。对比微调前后的召回率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| BM25 | 关键词搜索 | Okapi BM25。根据词频、IDF 和文档长度打分。 |
| Dense retrieval | 向量搜索 | 将查询和文档编码为向量，找最近邻。 |
| Bi-encoder | 嵌入模型 | 独立编码查询和文档。查询时速度快。 |
| Cross-encoder | 重排模型 | 将查询和文档一起编码。慢但准确。 |
| RRF | 排名融合 | 通过求和 `1/(k + rank)` 合并两个排名。 |
| Recall@k | 检索指标 | 相关文档出现在 top-k 中的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典的 bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与稠密检索差距的学习式稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 晚期交互检索。
