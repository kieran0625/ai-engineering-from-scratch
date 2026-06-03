# 关系抽取与知识图谱构建

> NER 发现了实体。实体链接将它们锚定。关系抽取找出它们之间的边。知识图谱是节点、边及其来源的总和。

**类型：** Build
**语言：** Python
**前置条件：** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**时间：** ~60 分钟

## 问题

分析师读到："Tim Cook became CEO of Apple in 2011." 四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）将自由文本转换为结构化三元组 `(subject, relation, object)`。在语料库上聚合，你就得到了知识图谱。聚合并查询，你就有了用于 RAG、分析或合规审计的推理基底。

2026 年的问题：LLM 热情地抽取关系。过于热情了。它们会幻觉出源文本不支持的三元组。没有来源，你就无法区分真实三元组和看似合理的虚构。2026 年的答案是 AEVS 风格的锚定-验证流水线。

## 概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，任意关系）。

**三种抽取方法。**

1. **规则 / 模式匹配。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工编写的正则表达式。脆弱、精确、可解释。
2. **监督分类器。** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准方法。
3. **生成式 LLM。** 提示模型输出三元组。开箱即用。需要来源，否则会幻觉出看似合理的垃圾信息。

**AEVS（Anchor-Extraction-Verification-Supplement，2026）。** 当前的幻觉缓解框架：

- **Anchor（锚定）。** 识别每个实体片段和关系短语片段的精确位置。
- **Extract（抽取）。** 生成与锚定片段关联的三元组。
- **Verify（验证）。** 将每个三元组元素匹配回源文本；拒绝任何无支持的内容。
- **Supplement（补充）。** 覆盖性检查确保没有锚定片段被遗漏。

幻觉大幅下降。需要更多计算，但可审计。

**开放与封闭的权衡。**

- **封闭本体。** 固定属性列表（例如，Wikidata 的 11,000+ 属性）。可预测。可查询。难以发明。
- **Open IE。** 任何动词短语都成为关系。高召回。低精确。查询混乱。

生产知识图谱通常混合使用：Open IE 用于发现，然后将关系规范化到封闭本体，再合并入主图。

## 构建

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见 `code/main.py` 获取完整的玩具抽取器。Hearst 模式仍在领域特定流水线中使用，因为它们可调试。

### 步骤 2：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：文本输入，三元组输出，已使用 Wikidata 属性 id。在远程监督数据上微调。标准的开源权重基线。

### 步骤 3：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

针对每个返回的片段与源文本进行验证。拒绝任何 `text[start:end] != triple_entity` 的情况。这是 AEVS "verify" 步骤的最简形式。

### 步骤 4：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占工程工作的 60-80%。为此预留预算。

### 步骤 5：构建小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个基于 KG 的 RAG 系统的原子单元。使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强的图存储进行扩展。

## 陷阱

- **RE 前进行共指消解。** "He founded Apple" — RE 需要知道 "he" 是谁。先运行共指消解（第 24 课）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析为同一节点。先进行实体链接（第 25 课）。
- **幻觉三元组。** LLM 会输出文本不支持的三元组。强制执行片段验证。
- **关系规范化漂移。** Open IE 关系不一致（"was born in," "came from," "is a native of"）。折叠到规范 id，否则图谱无法查询。
- **时间错误。** "Tim Cook is CEO of Apple" — 现在为真，2005 年为假。许多关系有时间边界。使用限定词（Wikidata 中的 `P580` start time、`P582` end time）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域微调的 RE 模型。

## 应用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred + Wikidata 规范化 |
| 领域特定（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| LLM 提示，审计输出 | AEVS 流水线：anchor → extract → verify → supplement |
| 大规模新闻信息抽取 | 基于模式 + 监督混合 |
| 从零构建知识图谱 | Open IE + 手动规范化 |
| 时序知识图谱 | 带限定词抽取（开始/结束时间、时间点） |

集成模式：NER → coref → entity linking → relation extraction → ontology mapping → graph load。每个阶段都是潜在的质量关卡。

## 交付

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 5 个新闻句子上运行 `code/main.py` 中的模式抽取器。手工检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或小型 LLM）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 流水线：LLM 抽取 + 针对源文本验证片段。在 50 个 Wikipedia 风格句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | `(s, r, o)` 元组，知识图谱的原子单元。 |
| Open IE | Extract anything | 开放词汇关系短语；高召回，低精确。 |
| Closed ontology | Fixed schema | 有限的关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | Normalize everything | 将表面名称 / 关系映射到规范 id。 |
| AEVS | Grounded extraction | Anchor-Extraction-Verification-Supplement 流水线（2026）。 |
| Provenance | Source-of-truth link | 每个三元组携带文档 id + 字符跨度指向其来源。 |
| Distant supervision | Cheap labels | 将文本与现有知识图谱对齐以创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力工具。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉缓解设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 标准图谱查询。
