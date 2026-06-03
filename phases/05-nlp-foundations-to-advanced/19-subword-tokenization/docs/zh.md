# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词级分词器遇到未登录词就束手无策。字符级分词器让序列长度爆炸。子词分词器取两者之长。每个现代 LLM 都基于其中一种。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 5 · 01（文本处理）、Phase 5 · 04（GloVe / FastText / 子词）
**时间：** ~60 分钟

## 问题所在

你的词表有 50,000 个词。用户输入了 "untokenizable"。你的分词器返回 `[UNK]`。模型现在对这个词没有任何信息。更糟的是：你语料中第 90 百分位的文档有 40 个罕见词，这意味着每篇文档丢失 40 处信息。

子词分词解决了这个问题。常见词保持为单个词元。罕见词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据可以覆盖所有内容，因为任何字符串最终都是字节序列。

2026 年的每个前沿 LLM 都采用三种算法之一（BPE、Unigram、WordPiece），封装在三个库之一中（tiktoken、SentencePiece、HF Tokenizers）。不选择其中之一，你就无法部署语言模型。

## 核心概念

![BPE vs Unigram vs WordPiece，逐字符对比](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding，字节对编码）。** 从字符级词表开始。统计每对相邻字符。将最频繁的字符对合并为新词元。重复此过程直到达到目标词表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE。** 相同算法，但在原始字节（256 个基础词元）而非 Unicode 字符上运行。保证零个 `[UNK]` 词元——任何字节序列都能编码。GPT-2 使用 50,257 个词元（256 字节 + 50,000 次合并 + 1 个特殊词元）。

**Unigram。** 从一个巨大的词表开始。为每个词元分配一个 unigram 概率。迭代地剪除那些移除后使语料库对数似然增加最少的词元。推理时具有概率性：可以采样多种分词方式（通过子词正则化进行数据增强时很有用）。T5、mBART、ALBERT、XLNet、Gemma 使用。

**WordPiece。** 合并能最大化训练语料库似然的字符对，而非原始频率。BERT、DistilBERT、ELECTRA 使用。

**SentencePiece vs tiktoken。** SentencePiece 是在原始 Unicode 文本上*训练*词表的库（BPE 或 Unigram），将空白编码为 `▁`。tiktoken 是 OpenAI 针对预构建词表的快速*编码器*；它不能训练。

经验法则：

- **训练新词表：** SentencePiece（多语言，无需预分词）或 HF Tokenizers。
- **针对 GPT 词表的快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者兼顾：** HF Tokenizers——一个库，训练 + 服务。

## 动手实现

### 步骤 1：从零实现 BPE

参见 `code/main.py`。循环逻辑：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

该算法编码了三个关键事实。`</w>` 标记词尾，因此 "low"（后缀）和 "lower"（前缀）保持区分。频率加权使高频字符对优先合并。合并列表是有序的——推理时按训练顺序应用合并。

### 步骤 2：用学习到的合并规则进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现为 O(n·|merges|)。生产环境实现（tiktoken、HF Tokenizers）使用带优先队列的合并秩查找，运行时间接近线性。

### 步骤 3：实际使用 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为 `▁`，`character_coverage` 控制对罕见字符的保留程度与映射到 `<unk>` 之间的平衡。

### 步骤 4：用 tiktoken 处理 OpenAI 兼容词表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust 后端）。与 GPT-4/5 分词完全匹配，用于字节计数、成本估算、上下文窗口预算。

## 2026 年仍在犯的陷阱

- **分词器漂移。** 在词表 A 上训练，在词表 B 上部署。词元 ID 不同；模型输出乱码。在 CI 中检查 `tokenizer.json` 哈希。
- **空白歧义。** BPE 中 "hello" 和 " hello" 产生不同的词元。始终显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足。** 英语为主的语料产生的词表会将非拉丁文字拆分为 5-10 倍更多的词元。相同提示在日语/阿拉伯语中 GPT-3.5 的成本高 5-10 倍。o200k_base 部分修复了此问题。
- **表情符号拆分。** 单个表情符号可能占用 5 个词元。在预算上下文时检查表情符号处理。

## 实际应用

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 从零训练单语模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供 OpenAI 兼容 API | tiktoken（`o200k_base`，用于 GPT-4+） |
| 领域特定词表（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，与基础词表合并 |
| 边缘推理、小模型 | Unigram（更小的词表效果更好） |

词表大小是扩展决策，不是固定常数。粗略经验：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿模型用 200k+。

## 部署

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习

1. **简单。** 在 `code/main.py` 的小型语料上训练一个 500 次合并的 BPE。编码三个 held-out 词。有多少词恰好产生 1 个词元，多少产生 >1 个？
2. **中等。** 比较 `cl100k_base`、`o200k_base` 和你训练的 SentencePiece BPE（vocab=32k）在 100 句英文维基百科句子上的词元数量。报告各自的压缩比。
3. **困难。** 用 BPE、Unigram 和 WordPiece 训练相同语料。在小型情感分类器上分别使用它们，测量下游准确率。选择是否能让 F1 变化超过 1 个点？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| BPE | Byte-Pair Encoding | 贪婪合并最频繁的字符对，直到达到目标词表大小。 |
| 字节级 BPE | 永远不会出现未知词元 | 在原始 256 字节上运行 BPE；GPT-2 / Llama 使用此方案。 |
| Unigram | 概率分词器 | 从大型候选集中基于对数似然进行剪枝；T5、Gemma 使用。 |
| SentencePiece | 那个处理空白的 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 那个快的 | OpenAI 的 Rust 后端 BPE 编码器，用于预构建词表。不能训练。 |
| 合并列表 | 那些魔法数字 | 有序的 `(a, b) → ab` 合并列表；推理时按顺序应用。 |
| 字符覆盖率 | 多罕见算太罕见？ | 分词器必须覆盖的训练语料字符比例；~0.9995 为典型值。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) —— Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) —— 该库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) —— 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) ——  cookbook + 编码列表。
