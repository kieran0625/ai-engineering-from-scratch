# RAG 分块策略

> 分块配置对检索质量的影响与嵌入模型的选择同等重要（Vectara NAACL 2025）。分块策略出错时，再多的重排序也无法挽救。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 14（信息检索），Phase 5 · 22（嵌入模型）
**时间：** ~60 分钟

## 问题所在

你将一份 50 页的合同输入 RAG 系统。用户提问："终止条款是什么？"检索器却返回了封面页。为什么？因为模型训练时使用的是 512 token 的分块，而终止条款位于 20 页之后，跨页断开，且没有局部关键词与查询关联。

解决方案不是"买更好的嵌入模型"。解决方案是分块。多大？重叠多少？在哪里分割？是否包含周围上下文？

2026 年 2 月的基准测试显示了令人意外的结果：

- Vectara 2026 年研究：递归 512 token 分块以 69% 对 54% 的准确率击败了语义分块。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠部分未提供任何可测量的收益。
- 上下文悬崖：上下文达到约 2,500 token 时，响应质量急剧下降。

"显而易见"的答案（语义分块、20% 重叠、1000 token）往往是错的。本节课将建立对六种策略的直觉，并告诉你何时该用哪种。

## 核心概念

![六种分块策略在同一段文本上的可视化](../assets/chunking.svg)

**固定分块。** 每 N 个字符或 token 分割一次。最简单的基线。会在句子中间断开。压缩率高，连贯性差。

**递归分块。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试在 `\n\n` 处分割，然后是 `\n`，接着是 `.`，最后是空格。优雅降级。2026 年的默认选择。

**语义分块。** 嵌入每个句子。计算相邻句子的余弦相似度。在相似度低于阈值处分割。保持主题连贯性。速度较慢；有时会产生 40 token 的微小片段，损害检索效果。

**句子分块。** 在句子边界处分割。每个分块一个句子，或 N 个句子的窗口。在约 5k token 以内与语义分块效果相当，但成本仅为后者的一小部分。

**父文档分块。** 存储小的子分块用于检索，同时存储更大的父分块用于上下文。通过子分块检索，返回父分块。优雅降级：糟糕的子分块仍能返回合理的父分块。

**Late chunking（2024）。** 先在 token 级别嵌入整个文档，然后将 token 嵌入汇聚为分块嵌入。保留跨分块的上下文。适用于长上下文嵌入器（BGE-M3、Jina v3）。计算成本更高。

**上下文检索（Anthropic, 2024）。** 在每个分块前添加 LLM 生成的文档位置摘要（"该分块位于终止条款的第 3.2 节……"）。在 Anthropic 自己的基准测试中，检索提升 35-50%。索引成本高昂。

### 超越所有默认设置的规则

将分块大小与查询类型匹配：

| 查询类型 | 分块大小 |
|---------|---------|
| 事实型（"CEO 的名字是什么？"） | 256-512 token |
| 分析型 / 多跳推理 | 512-1024 token |
| 整节理解 | 1024-2048 token |

NVIDIA 2026 年基准测试。分块应足够大以包含答案及局部上下文，又足够小使得检索器的 top-K 结果聚焦于答案而非上下文噪声。

## 动手实现

### 步骤 1：固定分块与递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

根据你的领域调整 `threshold`。过高 → 碎片化。过低 → 一个巨大分块。

### 步骤 3：父文档分块

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞见：对父文档去重。多个子分块可能映射到同一父分块；返回所有父分块会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引带上下文的分块。查询时，检索受益于额外的周围信号。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

务必进行基准测试。对你的语料库而言，"最佳"策略可能与任何博客文章都不一致。

## 常见陷阱

- **仅在事实型查询上评估分块。** 多跳查询会揭示截然不同的优胜者。使用按查询类型分层的评估集。
- **语义分块不设最小尺寸。** 产生 40 token 的碎片损害检索。务必强制执行 `min_tokens`。
- **将重叠视为 Cargo Cult。** 2026 年研究发现重叠往往毫无收益且使索引成本翻倍。测量，不要假设。
- **不强制执行最小/最大尺寸。** 5 token 或 5000 token 的分块都会破坏检索。进行限制。
- **跨文档分块。** 绝不允许分块跨越两个文档。始终按文档分块，然后合并。

## 实际应用

2026 年技术栈：

| 场景 | 策略 |
|-----|------|
| 首次构建，未知语料库 | 递归分块，512 token，无重叠 |
| 事实型 QA | 递归分块，256-512 token |
| 分析型 / 多跳推理 | 递归分块，512-1024 token + 父文档 |
| 大量交叉引用（合同、论文） | Late chunking 或上下文检索 |
| 对话型 / 对话语料库 | 轮次级别分块 + 说话人元数据 |
| 短文本（推文、评论） | 一个文档 = 一个分块 |

从递归 512 开始。在 50 个查询的评估集上测量 recall@5。由此开始调优。

## 交付

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 用固定分块(512, 0)、递归分块(512, 0) 和递归分块(512, 100) 对一个 20 页文档进行分块。比较分块数量和边界质量。
2. **中等。** 在 5 个文档上构建 30 个查询的评估集。测量递归分块、语义分块和父文档分块的 recall@5。哪种胜出？是否与博客文章一致？
3. **困难。** 实现上下文检索。测量相对于基线递归分块的 MRR 提升。报告索引成本（LLM 调用次数）与准确率增益的对比。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|-----|----------|---------|
| Chunk（分块） | 文档的一部分 | 被嵌入、索引和检索的子文档单元。 |
| Overlap（重叠） | 安全余量 | 相邻分块间共享的 N 个 token；在 2026 年基准测试中往往无用。 |
| Semantic chunking（语义分块） | 智能分块 | 在相邻句子的嵌入相似度下降处分割。 |
| Parent-document（父文档） | 两级检索 | 检索小子分块，返回更大的父分块。 |
| Late chunking | 嵌入后分块 | 在 token 级别嵌入完整文档，汇聚为分块向量。 |
| Contextual retrieval（上下文检索） | Anthropic 的技巧 | 索引前在每个分块前添加 LLM 生成的摘要。 |
| Context cliff（上下文悬崖） | 2500 token 墙 | 2026 年 1 月在 RAG 中观察到约 2.5k 上下文 token 处的质量下降。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产环境中的默认选择。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 分块与嵌入选择同等重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — late chunking 论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 通过 LLM 生成的上下文前缀实现 35-50% 的检索提升。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型划分的分块大小。
