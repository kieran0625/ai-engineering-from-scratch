# 问答系统

> 三种系统塑造了现代问答。抽取式找到文本片段。检索增强式将其锚定在文档中。生成式产出答案。每个现代 AI 助手都是三者的混合。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 5 · 11（机器翻译），Phase 5 · 10（注意力机制）
**时间：** ~75 分钟

## 问题

用户输入 "When did the first iPhone launch?"，期望得到 "June 29, 2007."。不是 "Apple's history is long and varied."。也不是孤立存在的 "2007"。要的是直接、有依据、正确的答案。

过去十年中，三种架构主导了问答领域。

- **抽取式问答。** 给定一个问题和一段已知包含答案的文本，找出答案片段在文本中的起始和结束索引。SQuAD 是经典基准。
- **开放域问答。** 不给定文本。先检索相关文本，再抽取或生成答案。这是当今每个 RAG 流程的基石。
- **生成式 / 闭卷问答。** 大型语言模型从其参数化记忆中回答。无需检索。推理最快，事实可靠性最低。

2026 年的趋势是混合式：检索最佳的几段文本，然后提示生成式模型基于这些文本作答。这就是 RAG，第 14 课将深入讲解检索部分。本课构建问答部分。

## 概念

![问答架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 用 transformer（BERT 家族）将问题和文本一起编码。训练两个预测答案起始和结束 token 索引的头部。损失是有效位置上的交叉熵。输出是文本中的片段。按构造不会幻觉，按构造无法处理文本无法回答的问题。

**检索增强式（RAG）。** 两个阶段。首先，检索器从语料库中找到前 `k` 段文本。其次，阅读器（抽取式或生成式）使用这些文本生成答案。检索器-阅读器的拆分使各自可以独立训练和评估。现代 RAG 通常在两者之间添加重排序器。

**生成式。** 仅解码器 LLM（GPT、Claude、Llama）从学习到的权重中回答。无检索步骤。对常识性知识表现优异，对罕见或最新事实灾难性失败。幻觉率与预训练数据中事实出现频率呈负相关。

## 构建

### 步骤 1：使用预训练模型进行抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，包含无法回答的问题。默认情况下，`question-answering` 流水线即使当模型的空分数获胜时也会返回最高分的片段——它*不会*自动返回空答案。要获得明确的"无答案"行为，向流水线调用传入 `handle_impossible_answer=True`：此时流水线仅在空分数超过所有片段分数时返回空答案。无论如何都要检查 `score` 字段。

### 步骤 2：检索增强式流水线（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。密集检索器（Sentence-BERT）通过语义相似度找到相关文本。抽取式阅读器（RoBERTa-SQuAD）从合并后的 top 文本中抽取答案片段。适用于小型语料库。对于百万文档级别的语料库，使用 FAISS 或向量数据库。

### 步骤 3：使用 RAG 进行生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式很重要。明确告诉模型要基于上下文作答，并在上下文不足时返回 "I don't know"，相比朴素提示可将幻觉率降低 40-60%。更复杂的模式会添加引用、置信度分数和结构化抽取。

### 步骤 4：反映真实世界的评估

SQuAD 使用**精确匹配（EM）**和**token 级 F1**。EM 是规范化（小写、去除标点、去掉冠词）后的严格匹配——预测与答案要么完全匹配得 1 分，要么得 0。F1 基于预测与参考答案之间的 token 重叠计算，给予部分分数。两者都低估改写："June 29, 2007" 与 "June 29th, 2007" 通常得到 0 EM（序数词破坏了规范化），但仍因重叠 token 获得可观的 F1。

生产环境问答：

- **答案准确性**（LLM 评判或人工评判，因为指标无法捕捉语义等价）。
- **引用准确性。** 引用的文本是否真正支持答案？通过生成引用与检索文本之间的字符串匹配即可自动检查。
- **拒绝校准。** 当答案不在检索到的文本中时，系统是否正确地说 "I don't know"？衡量虚假置信率。
- **检索召回率。** 在评估阅读器之前，衡量检索器是否将正确的文本放入前 `k` 中。阅读器无法修复缺失的文本。

### RAGAS：2026 年生产评估框架

`RAGAS` 专为 RAG 系统设计，是 2026 年的默认交付标准。它在无需黄金参考答案的情况下对四个维度进行评分：

- **忠实度。** 答案中的每个主张是否都来自检索到的上下文？通过基于 NLI 的蕴含关系衡量。你的主要幻觉指标。
- **答案相关性。** 答案是否回应了问题？通过从答案生成假设问题并与真实问题比较来衡量。
- **上下文精确度。** 检索到的片段中，实际相关的比例是多少？精确度低 = 提示中的噪声。
- **上下文召回率。** 检索到的集合是否包含了所有需要的信息？召回率低 = 阅读器无法成功。

无需参考的评分让你可以在实时生产流量上进行评估，无需 curated 的黄金答案。对于精确匹配指标失效的开放式问题，在其上叠加 LLM-as-judge。

`pip install ragas`。接入你的检索器 + 阅读器。每个查询获得四个标量。对退化情况发出告警。

## 使用

2026 年的技术栈。

| 用例 | 推荐方案 |
|---------|-------------|
| 给定文本，找到答案片段 | `deepset/roberta-base-squad2` |
| 基于固定语料库，不可接受闭卷 | RAG：密集检索器 + LLM 阅读器 |
| 文档存储上的实时问答 | 采用混合检索（BM25 + 密集）+ 重排序器的 RAG（第 14 课） |
| 对话式问答（追问） | 带对话历史的 LLM + 每轮 RAG |
| 高度事实性、受监管领域 | 基于权威语料库的抽取式；绝不单独使用生成式 |

抽取式问答在 2026 年已不流行，因为基于 LLM 的 RAG 能处理更多场景。但在需要逐字引用的场景中仍有应用：法律研究、监管合规、审计工具。

## 交付

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在上述 10 段 Wikipedia 文本上搭建 SQuAD 抽取式流水线。手工设计 10 个问题。衡量答案正确的频率。如果文本和问题都清晰，你应该能看到 7-9 个正确。
2. **中等。** 添加拒绝分类器。当 top 检索分数低于阈值（如余弦相似度 0.3）时，返回 "I don't know" 而不是调用阅读器。在留出集上调优阈值。
3. **困难。** 在你选择的 10,000 份文档语料库上构建 RAG 流水线。实现混合检索（BM25 + 密集）配合 RRF 融合（见第 14 课）。测量有无混合步骤时的答案准确性。记录哪些问题类型受益最大。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 抽取式 QA | 找到答案片段 | 预测给定文本中答案的起始和结束索引。 |
| 开放域 QA | 基于语料库的问答 | 无给定文本；必须先检索再回答。 |
| RAG | 检索再生成 | 检索增强生成。检索器 + 阅读器流水线。 |
| SQuAD | 经典基准 | Stanford Question Answering Dataset。EM + F1 指标。 |
| 幻觉 | 编造的答案 | 阅读器输出未被检索到的上下文支持。 |
| 拒绝校准 | 知道何时闭嘴 | 当无法回答时，系统正确地说 "I don't know"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 的经典密集检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
