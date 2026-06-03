# 命名实体识别

> 把名字抽出来。听起来简单，直到你面对模糊的边界、嵌套实体和专业领域术语。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (词嵌入)
**时间：** ~75 分钟

## 问题

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple (ORG)、Google (ORG)、iPhone (PRODUCT)、search deal (可能算)、US (GPE)。一个好的 NER 系统能提取出所有实体并标注正确类型。差的系统会漏掉 iPhone，把 Apple 水果和公司混淆，把 "US" 标成 PERSON。

NER 是每个结构化抽取流水线背后的主力。简历解析、合规日志扫描、医疗记录脱敏、搜索查询理解、聊天机器人回复的 grounding、法律合同抽取。你几乎看不到它，但永远依赖它。

本课沿着经典路径（基于规则、HMM、CRF）走向现代方法（BiLSTM-CRF、再到 transformers）。每一步解决前一步的特定局限。这个演进模式本身就是教训。

## 概念

**BIO 标注**（或 BILOU）将实体抽取转化为序列标注问题。为每个 token 标注 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（任何实体之外）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 实体链式连接：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽取任意跨度。

架构演进：

- **基于规则。** 正则 + 词典查找。已知实体精度高，新实体零覆盖。
- **HMM。** 隐马尔可夫模型。给定标签的发射概率，标签到标签的转移概率。Viterbi 解码。基于标注数据训练。
- **CRF。** 条件随机场。类似 HMM 但为判别式，可混合任意特征（词形、大小写、邻近词）。2026 年仍是低资源部署的经典生产主力。
- **BiLSTM-CRF。** 神经网络特征替代手工特征。LSTM 双向读取句子，顶层 CRF 强制标签序列一致性。
- **基于 Transformer。** 微调 BERT 加 token 分类头。最佳精度。最高计算量。

## 动手实现

### 步骤 1：BIO 标注辅助函数

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤 2：手工特征

对于经典（非神经网络）NER，特征就是一切。有用的特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式对专有名词是高信号特征。

### 步骤 3：简单的基于规则 + 词典基线

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产环境词典有数百万条目，从 Wikipedia 和 DBpedia 抓取。覆盖率好。消歧（`Apple` 公司 vs 水果）很差。这就是统计模型获胜的原因。

### 步骤 4：CRF 步骤（草图，非完整实现）

50 行从零写 CRF 没有概率论基础并不 enlightening。改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学习非法序列（如 `I-ORG` 在 `O` 之后）概率很低，这就是 CRF 如何在不写约束的情况下强制 BIO 一致性。

### 步骤 5：BiLSTM-CRF 增加了什么

特征变为学习得到。输入：token 嵌入（GloVe 或 fastText）。LSTM 从左到右、从右到左读取。拼接的隐状态通过 CRF 输出层。CRF 仍强制标签序列一致性；LSTM 用学习特征替代手工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，使用 `torchcrf.CRF`（pip install pytorch-crf）。相比手工 CRF 的提升是可测量的，但除非你有数万条标注句子，否则比你预期的小。

## 使用

spaCy 开箱即提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 将 `ORG` 标注为 `PRODUCT` — spaCy 的小模型产品实体覆盖弱。大模型（`en_core_web_lg`）表现更好。transformer 模型（`en_core_web_trf`）还要更好。

Hugging Face 用于基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X token 合并为跨度。没有它，你得到 token 级别标签，需要自己合并。

### 基于 LLM 的 NER（2026 年的选择）

零样本和少样本 LLM NER 目前在许多领域已与微调模型竞争，在标注数据稀缺时显著更好。

- **零样本提示。** 给 LLM 一个实体类型列表和示例 schema，要求 JSON 输出。开箱即用；在新领域上精度中等。
- **ZeroTuneBio 风格提示。** 将任务分解为候选提取 → 含义解释 → 判断 → 重新检查。多阶段提示（非一次性）在生物医学 NER 上大幅提升精度。相同模式适用于法律、金融和科学领域。
- **RAG 动态提示。** 从少量标注种子集中为每次推理检索最相似的标注示例；动态构建少样本提示。在 2026 年基准测试中，这比静态提示提升 GPT-4 生物医学 NER 的 F1 达 11-12%。
- **按实体类型分解。** 对于长文档，单次调用提取所有实体类型会随着长度增长而丢失召回率。每个实体类型运行一次提取。推理成本更高，精度显著更高。这是临床笔记和法律合同的标准模式。

2026 年的生产建议：在收集训练数据之前，先用 LLM 零样本基线。通常 F1 已经足够好，无需微调。

### 经典 NER 仍胜出的场景

即使有 LLM，经典 NER 在以下情况仍胜出：

- 延迟预算低于 50ms。
- 有数千标注示例且需要 98%+ F1。
- 领域有稳定本体，预训练 CRF 或 BiLSTM 迁移良好。
- 监管约束要求本地部署、非生成式模型。

### 经典方法的局限

- **领域迁移。** CoNLL 训练的 NER 在法律合同上表现比词典还差。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠跨度。需要嵌套 NER（多轮或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." Token 级别模型有时会拆分。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医学 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不懂。ScispaCy 和 BioBERT 是那里的起点。

## 部署

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作）并在 10 个句子上验证往返一致性。
2. **中等。** 在上述 sklearn-crfsuite CRF 上训练 CoNLL-2003 英语 NER 数据集。使用 `seqeval` 报告各实体 F1。典型结果：~84 F1。
3. **困难。** 在领域特定 NER 数据集（医学、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小模型对比。记录数据泄漏检查并写下让你惊讶的发现。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| NER | 提取名字 | 用类型标注 token 跨度（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 开始，`I-X` 继续，`O` 外部。 |
| BILOU | 更好的 BIO | 增加 `L-X`（末尾）、`U-X`（单元）以获得更清晰的边界。 |
| CRF | 结构化分类器 | 建模标签间转移，而非仅发射概率。强制有效序列。 |
| 嵌套 NER | 重叠实体 | 一个跨度与其子跨度是不同实体。BIO 无法表达。 |
| 实体级 F1 | 正确的 NER 指标 | 预测跨度必须与真实跨度完全匹配。Token 级 F1 高估精度。 |

## 延伸阅读

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。经典。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入成为标准的 token 分类模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库。永远用它。
