# 预训练数据流水线

> 模型是一面镜子。它反映你喂给它的任何数据。喂它垃圾，它就以完美的流畅度反映垃圾。

**类型：** 构建
**语言：** Python
**前置条件：** Phase 10, Lessons 01-02（分词器、构建分词器）
**时间：** ~90 分钟

## 学习目标

- 构建一个流式数据流水线，能够在不将所有数据加载到内存的情况下，对 TB 级文本进行分词、分块、打乱和分批
- 实现真实预训练流水线中使用的数据质量过滤器（去重、语言检测、内容过滤）
- 创建具有正确注意力掩码和文档边界处理的固定长度训练序列
- 分析流水线吞吐量，确保数据加载器跟上 GPU 训练速度

## 问题

你有了分词器。现在你需要数据。

不是数据集。不是 CSV 文件。而是 TB 级的文本——经过清洗、去重、质量过滤、分词成固定长度序列，并以足够快的速度提供随机批次，让你的 8-GPU 集群无需等待下一批数据。

大多数人认为训练 LLM 是关于模型架构的。并非如此。Llama 3 使用了 15.6 万亿个 token。GPT-3 使用了 3000 亿个。DeepSeek-V2 使用了 8.1 万亿个。这三者的架构大致相同：堆叠的 transformer 块，带有注意力和前馈层。输出质量的差异主要来自数据。

DeepMind 的 Chinchilla 论文精确地说明了这一点。对于给定的计算预算，模型参数与训练 token 之间存在最优比例。Chinchilla 表明，2022 年的大多数模型都严重训练不足——它们相对于看到的数据量来说参数过多。一个训练了 1.4 万亿 token 的 70B 参数模型（Chinchilla 最优）表现优于一个训练了 3000 亿 token 的 280B 参数模型（Gopher）。

你的数据流水线决定了你的模型是学习语言还是学习噪声。

## 概念

### 数据来源

每个大型语言模型都是在混合数据源上训练的。确切的组成对大多数实验室来说是严格保密的，但我们了解足够多来理解这些类别。

| 来源 | 规模 | 质量 | 使用者 |
|--------|------|---------|---------|
| Common Crawl | ~250 TB 原始数据 | 低（需要大量过滤） | GPT-3、Llama、大多数开源模型 |
| Wikipedia | ~20 GB | 高 | 每个主要 LLM |
| GitHub 代码 | ~1 TB+ | 中（大量重复、死代码） | StarCoder、CodeLlama、DeepSeek-Coder |
| 书籍（BookCorpus、Pile） | ~100 GB | 高 | GPT-2、GPT-3、早期模型 |
| 学术论文（arXiv、S2ORC） | ~100 GB | STEM 领域高 | Llama、Galactica |
| StackOverflow、Reddit | ~100 GB | 中 | Llama、Falcon |
| 精选网页（C4、RefinedWeb） | ~5 TB | 中高（预过滤） | T5、Falcon |

Llama 3 披露了其数据混合比例：大约 50% 网页数据、25% 代码、13% 书籍和学术论文、8% 数学数据、4% 多语言网页数据。总计 15.6 万亿 token，来自超过 5 TB 的原始文本。

比例与总大小同样重要。网页数据太多，模型会变成 Reddit 复读机。代码太少，它无法编程。数学太少，它在推理方面表现不佳。调好这个比例是训练 LLM 最难的部分之一，而且没有公式可循——需要实验和评估。

### 数据清洗

原始网页数据非常脏。典型的 Common Crawl 转储包含：

- HTML 标签和 JavaScript
- 样板页眉、页脚、导航菜单
- 重复页面（完全重复和近似重复）
- 机器生成的垃圾信息
- 个人身份信息（PII）
- 低质量文本（关键词列表、SEO 垃圾）
- 编码为文本的非文本内容

清洗这不是可选的。这是模型生成连贯段落与输出混合 HTML 标签和产品列表之间的区别。

```mermaid
graph TD
    A[Raw Text] --> B[HTML Strip]
    B --> C[Language Detection]
    C --> D[Quality Filter]
    D --> E[Deduplication]
    E --> F[PII Removal]
    F --> G[Clean Text]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

每一步消除一类噪声：

**HTML 剥离：** 移除所有标记。只保留可见文本内容。像 `trafilatura` 或 `readability` 这样的库可以提取文章内容，同时丢弃导航、广告和样板内容。

**语言检测：** 使用 fastText 的语言识别模型（lid.176.bin）对每个文档进行分类。过滤到你的目标语言。一个被分类为英语但置信度低于 0.8 的文档很可能不是干净的英语。

**质量过滤：** 这里变得有趣。RefinedWeb（Falcon 背后的数据集）使用基于困惑度的过滤器：在 Wikipedia 上训练一个小型语言模型，然后对每个文档打分。高困惑度意味着该文档不像 Wikipedia——可能是垃圾信息、关键词列表或机器生成内容。困惑度超过阈值的文档被移除。

**去重：** 单一最具影响力的清洗步骤。Common Crawl 包含大量重复页面——法律免责声明、Cookie 通知、服务条款。在重复数据上训练浪费计算资源，还可能导致模型逐字记忆和复述特定段落。

**PII 移除：** 姓名、电子邮件地址、电话号码、社会安全号码。结构化 PII 使用基于正则的检测，上下文中的姓名使用 NER 模型。

### 使用 MinHash 去重

精确去重很简单：对每个文档哈希，移除重复。但近似重复才是真正的问题。同一篇新闻文章的两个副本，周围广告略有不同，就是近似重复。内容 95% 相同，但字节级不同。

MinHash + 局部敏感哈希（LSH）高效地解决了这个问题。

```mermaid
graph LR
    A[Document] --> B[Shingling]
    B --> C[MinHash Signature]
    C --> D[LSH Buckets]
    D --> E[Candidate Pairs]
    E --> F[Jaccard Similarity]
    F --> G[Deduplicated Set]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

思路：

1. **Shingling：** 将每个文档转换为 n-gram 集合（例如，5-gram 词或字符）。"the quick brown fox" 使用 3 词 shingles 变为 {"the quick brown", "quick brown fox"}。

2. **MinHash：** 对于每个文档的 shingle 集合，计算 k 个哈希值。每个哈希值是所有 shingle 在不同哈希函数下的最小哈希。这创建了一个固定大小的"签名"，近似任意两个文档之间的 Jaccard 相似度。

3. **LSH：** 根据 MinHash 签名的 bands 将文档分组到桶中。同一桶中的文档是候选近似重复。这避免了比较每一对——只比较候选。

4. **验证：** 对于每个候选对，计算精确 Jaccard 相似度。如果相似度超过阈值（通常为 0.8），移除一个副本。

Llama 团队报告通过去重移除了约 38% 的网页数据。这不是一个小数字。超过三分之一的 Common Crawl 是重复或近似重复内容。

### 序列打包

你的模型期望固定长度的输入序列。你的文档长度不一。有些 50 个 token。有些 50,000 个 token。

朴素方法：将每个文档填充到最大序列长度。这在填充 token 上浪费大量计算，而这些 token 对学习毫无贡献。

更好的方法：将多个文档打包到单个序列中，用序列结束 token 分隔。一个 2048 token 的序列可能包含三个短文档，用特殊 token 连接。

```mermaid
graph TD
    subgraph Naive Packing
        A1["Doc A (200 tokens)"] --> P1["[PAD] x 1848"]
        A2["Doc B (500 tokens)"] --> P2["[PAD] x 1548"]
        A3["Doc C (100 tokens)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing
        B1["Doc A (200) | Doc B (500) | Doc C (100) | Doc D (400) | Doc E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff
```

注意力掩码必须正确设置。同一打包序列中，文档 A 的 token 不应关注文档 B 的 token。这需要块对角注意力掩码。

长文档在序列边界处被截断或分块。分割点很重要：在句子中间分割会迫使模型看到不完整的想法。一些流水线尽可能按段落或句子边界对齐分割。

### Chinchilla 缩放定律

对于固定计算预算 C（以 FLOPs 计），最优模型大小 N 和数据集大小 D 遵循：

```
N_opt ~ C^0.5
D_opt ~ C^0.5
```

在实践中，这意味着你应该大致相等地缩放模型大小和数据集大小。参数多 10 倍的模型需要大约多 10 倍的训练 token 才能达到相同的损失。

| 模型 | 参数量 | 训练 token | Chinchilla 最优？ |
|-------|-----------|----------------|-------------------|
| GPT-3 | 175B | 300B | 否（训练不足 3-4 倍） |
| Chinchilla | 70B | 1.4T | 是（设计如此） |
| Llama 2 | 70B | 2T | 过度训练（有意为之） |
| Llama 3 | 70B | 15T | 严重过度训练 |

Llama 3 故意违反 Chinchilla 定律。Meta 发现，在更多数据上过度训练——远超计算最优比例——能产生更适合推理的更好模型。额外的训练成本只支付一次，但更小的模型永远更便宜地 serving。这有时被称为"推理最优"缩放方法，自 2024 年以来已成为行业标准。

## 构建

### 步骤 1：文本清洗

剥离 HTML、规范化空白、移除非文本内容。我们将使用公共领域文本（Project Gutenberg）作为我们的小型语料库。

```python
import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
```

质量过滤器捕获 SEO 垃圾（全大写）、机器生成噪声（特殊字符比例高）和存根页面（太短）。仅这三项检查就能从网页爬取中移除惊人数量的垃圾。

### 步骤 2：MinHash 去重

从头实现 MinHash。不需要外部库——只需 `hashlib`。

```python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
```

`num_hashes=128` 和 `bands=16` 参数控制精确率-召回率权衡。更多哈希值提供更准确的相似度估计。更多 bands 提高召回率（捕获更多重复），但以更多假阳性为代价。这些值对典型网页文本效果良好。

### 步骤 3：分词并打包序列

取清洗、去重后的文本，分词，并打包成固定长度序列用于训练。

```python
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
```

### 步骤 4：训练用 DataLoader

生成随机批次的打包序列。这是训练循环消费的内容。

```python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
```

### 步骤 5：数据集统计

计算关键数字：总 token 数、唯一 token 数、压缩比、文档长度分布。

```python
from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
```

压缩比告诉你分词器在此语料库上的效率。英文文本通常压缩到每 token 约 3-4 个字符。如果看到每 token 1.5 个字符，你的分词器分割过于激进。如果看到 8+，它学会了非常特定领域的合并。

序列利用率告诉你打包序列中有多少是真实数据 versus 填充。低于 90% 意味着打包效率低下——你在填充 token 上浪费计算。

## 使用

### 与 HuggingFace Datasets 对比

通过 HuggingFace 的 datasets 库加载相同语料库，对比流水线速度。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
```

HuggingFace 流水线底层使用 Rust 分词器，并在 4 核上并行处理。你的纯 Python 流水线会慢 10-50 倍。这就是生产团队使用编译分词器的原因。算法相同。实现语言是差异所在。

## 交付

本节课生成一个用于验证和调试 LLM 训练流水线数据质量的 prompt。参见 `outputs/prompt-data-quality-checker.md`。

## 练习

1. **简单：** 使用简单启发式方法（字符集分析）为清洗流水线添加语言检测。仅过滤到英语文档，并测量有多少文档被移除。
2. **中等：** 使用 SHA-256 哈希实现精确去重，与 MinHash 近似去重并行。在网页爬取语料库上对比每种方法捕获的重复数量。
3. **困难：** 构建基于困惑度的质量过滤器。在 Wikipedia 文本上训练一个小型二元语言模型，按困惑度对每个文档打分，并移除底部 20%。对比在过滤与未过滤数据上训练时模型输出质量。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------------------|
| Common Crawl | "互联网" | 一家每月爬取网页的非营利组织——~250TB 原始数据，大多数 LLM 训练数据的起点 |
| MinHash | "某种哈希技巧" | 使用固定大小签名估计集合间 Jaccard 相似度的技术——实现大规模近似重复检测 |
| LSH | "局部敏感哈希" | 将相似项分组到同一桶中的方法——将成对比较从 O(n^2) 降低到近线性 |
| 序列打包 | "拼接文档" | 将多个文档装入固定长度序列并设置正确注意力掩码——消除填充浪费 |
| Chinchilla 缩放 | "用更多数据训练" | 对于固定计算预算，最优性能需要大致相等地缩放模型大小和训练 token |
| Fertility | "每词 token 数" | 每词平均 token 数——GPT-4 的英文为 1.3，非拉丁文字更高 |
| 数据混合 | "选择训练数据" | 代码 vs 文本 vs 数学 vs 多语言数据的比例——没有公式，需要实验 |
| 困惑度过滤器 | "质量评分" | 使用小型语言模型对文档打分——高困惑度意味着文本不像干净的参考数据 |
| 去重 | "移除副本" | 消除完全重复和近似重复文档——通常移除 30-40% 的原始网页数据 |
| 注意力掩码 | "看哪些 token" | 防止在打包序列中跨文档边界注意的二进制掩码 |

## 延伸阅读

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) -- 改变我们对数据规模认知的论文
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) -- 如何将 Common Crawl 过滤到高质量
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) -- Llama 2 的数据流水线细节
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) -- 为什么去重比你想象的更重要
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900) -- 原始 MinHash 论文
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) -- 15.6T token、数据混合比例、过滤流水线
