# 高级 RAG（分块、重排序与混合搜索）

> 基础 RAG 仅检索最相似的 top-k 个文本块。这对简单问题有效。但在多跳推理、模糊查询和大型语料库面前就会失效。高级 RAG 的区别在于：一个是只能在 10 份文档上跑通的演示项目，另一个是能在 1000 万份文档上稳定运行的系统。

**类型：** 构建
**语言：** Python
**前置要求：** 第 11 阶段，第 06 课（RAG）
**耗时：** 约 90 分钟
**相关课程：** 第 5 阶段 · 第 23 课（RAG 的分块策略）涵盖了所有六种分块算法——递归、语义、句子、父文档、晚期分块、上下文检索，并附有 Vectara/Anthropic 的基准测试数据。本课在此基础上进阶：混合搜索、重排序、查询转换。

## 学习目标

- 实现高级分块策略（语义、递归、父子），以保留文档结构和上下文
- 构建混合搜索管道，结合 BM25 关键词匹配、语义向量搜索以及交叉编码器重排序器
- 应用查询转换技术（HyDE、多查询、退步查询）以提升对模糊或复杂问题的检索效果
- 诊断并修复常见的 RAG 故障：检索到错误的文本块、上下文中无答案、多跳推理断裂

## 问题所在

你在第 06 课中构建了基础 RAG 管道。它对小型语料库上的直白问题有效。现在试试这些场景：

**模糊查询**：“上个季度的营收是多少？”语义搜索返回了关于营收策略、营收预测以及 CFO 对营收增长看法的文本块。它们都与“revenue”一词在语义上相似，但都不包含实际数字。正确的文本块写着“$47.2M in Q3 2025”，但使用的是“earnings”而非“revenue”。嵌入模型认为“营收策略”比“Q3 收益为 $47.2M”更接近查询词。

**多跳问题**：“哪个团队在客户满意度评分提升方面表现最高？”这需要找出每个团队的满意度评分，进行比较，并找出最大值。没有单个文本块包含完整答案。信息分散在各个团队报告中。

**大型语料库问题**：你有 200 万个文本块。正确答案位于第 1,847,293 号文本块。你的 top-5 检索结果拉取了第 #14、#89,201、#1,200,000、#44 和 #901,333 号文本块。它们在嵌入空间中很接近，但都不包含答案。在这个规模下，近似最近邻搜索引入的误差足以将相关结果挤出 top-k。

基础 RAG 失败的原因是向量相似度不等于相关性。一个文本块可能在语义上与查询相似，但对回答问题毫无帮助。高级 RAG 通过四种技术解决此问题：混合搜索（增加关键词匹配）、重排序（更仔细地评估候选项）、查询转换（搜索前优化查询）以及更好的分块（在合适的粒度进行检索）。

## 核心概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似度）擅长理解含义。“如何取消我的订阅？”能匹配到“计划终止步骤”，即使两者没有共享词汇。但它会错过精确匹配。“错误代码 E-4021”可能无法匹配包含“E-4021”的文本块，如果嵌入模型将其视为噪声的话。

关键词搜索（BM25）则相反。它擅长精确匹配。“E-4021”能完美匹配。但如果文档写的是“terminate your plan”，那么“cancel my subscription”将返回零结果。

混合搜索同时运行两者，然后合并结果。

**BM25**（最佳匹配 25）是标准的关键词搜索算法。自 20 世纪 90 年代以来一直是搜索引擎的核心。公式如下：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是 t 在文档 d 中的词频，IDF(t) 是逆文档频率，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和（默认 1.2），b 控制长度归一化（默认 0.75）。

通俗来说：当文档包含查询词（尤其是稀有词）时，BM25 会给文档打高分，但对重复出现的词会有递减回报。包含 50 次“revenue”的文档，其相关性不会是一次包含该词的文档的 50 倍。

### 倒数排名融合 (RRF)

你拥有两个排名列表：一个来自向量搜索，一个来自 BM25。如何合并它们？倒数排名融合是标准做法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是一个常数（通常为 60），用于防止排名第一的结果占据绝对主导。

在向量搜索中排名第 1、在 BM25 中排名第 5 的文档得分：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

在向量搜索中排名第 3、在 BM25 中排名第 2 的文档得分：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 自然地平衡了这两种信号。在两个列表中排名都高的文档获得最高分。在一个列表中排名第 1 但在另一个列表中缺失的文档获得中等分数。这种方法很稳健，因为它使用排名而非原始分数，因此两个系统之间的分数分布差异无关紧要。

### 重排序

检索（无论是向量、关键词还是混合检索）速度快但不精确。它使用双编码器：查询和每个文档独立嵌入，然后进行比较。嵌入计算一次并缓存。这可以扩展到数百万文档。

重排序使用交叉编码器：查询和候选文档一起输入到一个输出相关性分数的模型中。模型同时看到两段文本，能够捕捉它们之间细粒度的交互。交叉编码器可以理解“What were Q3 earnings?”与包含“$47.2M in Q3”的文本块高度相关，即使双编码器错过了这种关联。

权衡之处：交叉编码器比双编码器慢 100 到 1000 倍，因为它们联合处理查询-文档对。你无法为一百万个文档预计算交叉编码器分数。解决方案：检索更大的候选集（混合搜索的 top-50），然后使用交叉编码器重排序以获得最终的 top-5。

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]
```

常见重排序模型（2026 年阵容）：
- Cohere Rerank 3.5：托管 API，多语言，在混合语料库上召回率提升最佳
- Voyage rerank-2.5：托管 API，托管选项中延迟最低
- Jina-Reranker-v2 Multilingual：开源权重，支持 100 多种语言
- bge-reranker-v2-m3：开源权重，强大的基线模型
- cross-encoder/ms-marco-MiniLM-L-6-v2：开源权重，可在 CPU 上运行以进行原型开发
- ColBERTv2 / Jina-ColBERT-v2：晚期交互多向量重排序器——评分时为 O(tokens) 而非 O(docs)

### 查询转换

有时问题不在于检索，而在于查询本身。“关于新政策变动的那件事是什么？”是一个糟糕的搜索查询。它不包含任何具体术语。嵌入结果模糊。没有任何检索系统能从这种查询中找到正确的文档。

**查询重写**：将用户的查询改写为更好的搜索查询。LLM 可以做到这一点：

```
User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"
```

**HyDE（假设文档嵌入）**：不使用查询进行搜索，而是生成一个假设性答案，对其进行嵌入，然后搜索与之相似的真实文档。

```
Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."
```

对假设性答案进行嵌入，并搜索与其相似的真实文档。直觉是：假设性答案在嵌入空间中比原始问题更接近真实答案。问题和答案具有不同的语言结构。通过生成假设性答案，你可以在嵌入空间中弥合“问题空间”和“答案空间”之间的差距。

HyDE 在检索前增加了一次 LLM 调用。这会将延迟增加 500-2000 毫秒。当原始查询的检索质量较差时，这是值得的。

### 父子分块

标准分块迫使你做出权衡：小文本块用于精确检索，大文本块用于提供充足上下文。父子分块消除了这种权衡。

对小文本块（128 个 token）建立索引以进行检索。当检索到一个小文本块时，返回其父文本块（512 个 token）用于提示词。小文本块精确匹配查询。父文本块为 LLM 生成良好答案提供足够的上下文。

```mermaid
graph TD
    P["Parent chunk (512 tokens)<br/>Full section about refund policy"]
    C1["Child chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child chunk (128 tokens)<br/>Enterprise: 60-day pro-rated"]
    C3["Child chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child chunk (128 tokens)<br/>How to submit a request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"matches child"| C2
    C2 -.->|"return parent"| P
```

查询“企业退款？”精确匹配子文本块 C2。但提示词接收的是完整的父文本块 P，其中包含有关处理时间和提交流程的周围上下文。

### 元数据过滤

在运行向量搜索之前，按元数据（日期、来源、类别、作者、语言）过滤语料库。这可以减少搜索空间并防止不相关的结果。

“上个月安全策略发生了什么变化？”应该只搜索过去 30 天内安全类别的文档。如果没有元数据过滤，你将搜索整个语料库，可能会检索到一份恰好语义相似但已有两年的旧安全文档。

生产级 RAG 系统会在每个文本块旁存储元数据：源文档、创建日期、类别、作者、版本。向量数据库支持在相似度搜索之前按元数据进行预过滤，这对于大规模下的性能至关重要。

### 评估

你构建了一个 RAG 系统。如何知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对于一组已知相关文档的测试问题，相关文档出现在 top-k 结果中的百分比是多少？如果问题的答案在第 47 号文本块中，第 47 号文本块是否出现在 top-5 中？

**忠实度（Faithfulness）**：生成的答案是否基于检索到的文档？如果检索到的文本块说“60 天退款窗口”，而模型说“90 天退款窗口”，那就是忠实度失败。模型在拥有正确上下文的情况下仍然产生了幻觉。

**答案正确性**：生成的答案是否与预期答案匹配？这是端到端指标。它结合了检索质量和生成质量。

一个简单的忠实度检查方法：提取生成答案中的每个主张，并验证其在检索到的文本块中是否存在（实质内容上）。如果答案包含任何检索文本块中都没有的事实，那很可能是幻觉。

```mermaid
graph TD
    subgraph "Evaluation Framework"
        Q["Test questions<br/>+ expected answers<br/>+ relevant doc IDs"]
        Q --> Ret["Retrieval evaluation<br/>Recall@k: are right<br/>docs retrieved?"]
        Q --> Faith["Faithfulness evaluation<br/>Is answer grounded<br/>in retrieved docs?"]
        Q --> Correct["Correctness evaluation<br/>Does answer match<br/>expected answer?"]
    end
```

## 动手实现

### 步骤 1：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 步骤 2：倒数排名融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 步骤 3：混合搜索管道

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 步骤 4：简易重排序器

在生产环境中，你会使用交叉编码器模型。这里我们构建一个重排序器，使用词重叠、术语重要性和短语匹配来评估查询-文档的相关性。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 步骤 5：HyDE（假设文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 步骤 6：父子分块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 步骤 7：忠实度评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 实际应用

使用真实的交叉编码器进行重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

配合真实 LLM 使用 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

配合 Weaviate 用于生产环境的混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡比例：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统使用 0.3 到 0.7 之间的 alpha 值。

## 交付成果

本课产出：
- `outputs/prompt-advanced-rag-debugger.md` —— 用于诊断和修复 RAG 质量问题的提示词
- `outputs/skill-advanced-rag.md` —— 用于构建具备混合搜索和重排序功能的生产级 RAG 的技能包

## 练习

1. 在示例文档上比较 BM25、向量搜索和混合搜索。针对 5 个测试查询中的每一个，记录哪种方法在位置 #1 返回了最相关的文本块。混合搜索应在至少 3 个查询中胜出。

2. 实现元数据过滤器。为每个文档添加“category”字段（security, billing, api, product）。在运行向量搜索之前，将文本块过滤为仅属于相关类别。使用“What encryption is used?”进行测试，并验证它仅搜索 security 类别的文本块。

3. 使用第 06 课中的简单 generate 函数构建完整的 HyDE 管道。在所有 5 个测试查询上，比较直接查询搜索和 HyDE 搜索的检索质量（top-3 相关性）。HyDE 应能改善模糊查询的结果。

4. 在示例文档上实现父子分块策略。使用 child_size=30 和 parent_size=100。使用子文本块进行搜索，但在提示词中返回父文本块。将生成的答案与 chunk_size=50 的标准分块生成的答案进行比较。

5. 创建评估数据集：10 个带有已知答案文本块的问题。分别测量 (a) 仅向量搜索、(b) 仅 BM25、(c) 混合搜索、(d) 混合搜索+重排序的 Recall@3、Recall@5 和 Recall@10。绘制结果图，并识别重排序在何处带来最大帮助。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| BM25 | “关键词搜索” | 一种概率排名算法，根据词频、逆文档频率和文档长度归一化对文档进行打分 |
| Hybrid search | “兼顾两者优势” | 并行运行语义（向量）和关键词（BM25）搜索，然后通过排名融合合并结果 |
| Reciprocal Rank Fusion | “合并排名列表” | 通过将每个文档在所有列表中的 1/(k + rank) 相加来合并多个排名列表 |
| Reranking | “第二遍打分” | 使用更昂贵的交叉编码器模型对初始检索得到的候选集重新打分 |
| Cross-encoder | “联合查询-文档模型” | 将查询和文档作为单一输入输入的模型，输出相关性分数；比双编码器更准确，但速度太慢，无法用于全量语料库搜索 |
| Bi-encoder | “独立嵌入模型” | 独立对查询和文档进行嵌入的模型；速度快因为嵌入已预计算，但准确性不如交叉编码器 |
| HyDE | “用假答案搜索” | 为查询生成一个假设性答案，对其进行嵌入，并搜索与其相似的真实文档 |
| Parent-child chunking | “小范围检索，大上下文” | 对小文本块建立索引以实现精确检索，但返回较大的父文本块以提供充足上下文 |
| Metadata filtering | “搜索前缩小范围” | 在运行向量搜索前按属性（日期、来源、类别）过滤文档，以减少搜索空间 |
| Faithfulness | “是否紧扣上下文” | 生成的答案是否由检索到的文档支持，而非源自模型训练数据的幻觉 |

## 延伸阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) -- BM25 的权威参考，解释了公式背后的概率基础
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) -- RRF 的原始论文，证明其优于更复杂的融合方法
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) -- HyDE 论文，证明假设文档嵌入无需任何训练数据即可提升检索效果
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) -- 证明了在 BM25 之上使用交叉编码器重排序可显著提升检索质量
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) -- 将提示词构建和权重选择视为检索管道的优化问题；阅读此文了解如何“编程 LLM”而非“提示 LLM”。
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) -- GraphRAG 论文：实体关系抽取 + Leiden 社区检测用于查询聚焦摘要；探讨全局与局部检索的区别。
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) -- 带有反思标记的自评估 RAG；处于静态检索后生成之前的智能体前沿。
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) -- 如何将自然语言查询转换为结构化数据库查询（Text-to-SQL、Cypher）作为检索前的步骤。
