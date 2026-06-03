# 分词器：BPE、WordPiece、SentencePiece

> 你的 LLM 不读英语。它读整数。分词器决定这些整数承载意义还是浪费意义。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 05（NLP 基础）
**时间：** ~90 分钟

## 学习目标

- 从零实现 BPE、WordPiece 和 Unigram 分词算法，并比较它们的合并策略
- 解释词汇表大小如何影响模型效率：太小会导致序列过长，太大则会浪费嵌入参数
- 分析分词器在不同语言和代码上的缺陷，识别特定分词器失效的场景
- 使用 tiktoken 和 sentencepiece 库对文本进行分词，并检查生成的 token ID

## 问题

你的 LLM 不读英语。它不读任何语言。它读数字。

"Hello, world!" 和 [15496, 11, 995, 0] 之间的差距就是分词器。每个词、每个空格、每个标点符号都必须转换为整数，模型才能处理。这种转换并非中立——它将假设烘焙进模型，且后续无法撤销。

搞错了，你的模型会浪费容量，用多个 token 编码常见词汇。"unfortunately" 变成四个 token 而非一个。对于多音节词汇密集的文本，你的 128K 上下文窗口直接缩水 75%。搞对了，同样的上下文窗口能容纳双倍信息。"这个模型处理代码很好"和"这个模型在 Python 上卡死"之间的区别，往往取决于分词器的训练方式。

每次调用 GPT-4 或 Claude 的 API，都是按 token 计价。模型生成的每个 token 都消耗算力。表示输出所需的 token 越少，端到端推理越快。分词不是预处理。它是架构。

## 概念

### 三种失败的方法（和一种成功的方法）

将文本转为数字有三种显而易见的做法。其中两种无法规模化。

**词级分词** 按空格和标点切分。"The cat sat" 变成 ["The", "cat", "sat"]。简单。但 "tokenization" 呢？或 "GPT-4o"？或德语复合词 "Geschwindigkeitsbegrenzung"？词级分词需要庞大的词汇表来覆盖每种语言的每个词。遗漏一词，就会得到 dreaded `[UNK]` token——模型在说"我不知道这是什么"。仅英语就有超过一百万种词形。加上代码、URL、科学记数法和其他 100 多种语言，你需要无限的词汇表。

**字符级分词** 走向另一个极端。"hello" 变成 ["h", "e", "l", "l", "o"]。词汇表极小（几百个字符）。永远不会出现未知 token。但序列变得极长。10 个词级 token 的句子变成 50 个字符级 token。模型必须学会 "t"、"h"、"e" 一起表示 "the"——浪费注意力容量在人类三岁就学会的东西上。

**子词分词** 找到甜蜜点。常见词保持完整："the" 是一个 token。罕见词分解为有意义的片段："unhappiness" 变成 ["un", "happi", "ness"]。词汇表保持可控（30K 到 128K token）。序列保持简短。未知 token 基本消失，因为任何词都可以由子词片段构建。

每个现代 LLM 都使用子词分词。GPT-2、GPT-4、BERT、Llama 3、Claude——全是。问题在于用哪种算法。

```mermaid
graph TD
    A["Text: 'unhappiness'"] --> B{"Tokenization Strategy"}
    B -->|Word-level| C["['unhappiness']\n1 token if in vocab\n[UNK] if not"]
    B -->|Character-level| D["['u','n','h','a','p','p','i','n','e','s','s']\n11 tokens"]
    B -->|Subword BPE| E["['un','happi','ness']\n3 tokens"]

    style C fill:#ff6b6b,color:#fff
    style D fill:#ffa500,color:#fff
    style E fill:#51cf66,color:#fff
```

### BPE：字节对编码

BPE 是一种被重新用于分词的贪婪压缩算法。思路简单到可以写在索引卡上。

从单个字符开始。统计训练语料中每对相邻字符。将最频繁的一对合并为新 token。重复直到达到目标词汇表大小。

以下是 BPE 在包含 "lower"、"lowest" 和 "newest" 的小型语料上的运行过程：

```
Corpus (with word frequencies):
  "lower"  x5
  "lowest" x2
  "newest" x6

Step 0 -- Start with characters:
  l o w e r       (x5)
  l o w e s t     (x2)
  n e w e s t     (x6)

Step 1 -- Count adjacent pairs:
  (e,s): 8    (s,t): 8    (l,o): 7    (o,w): 7
  (w,e): 13   (e,r): 5    (n,e): 6    ...

Step 2 -- Merge most frequent pair (w,e) -> "we":
  l o we r        (x5)
  l o we s t      (x2)
  n e we s t      (x6)

Step 3 -- Recount and merge (e,s) -> "es":
  l o we r        (x5)
  l o we s t      (x2)    <- 'es' only forms from 'e'+'s', not 'we'+'s'
  n e we s t      (x6)    <- wait, the 'e' before 'we' and 's' after 'we'

Actually tracking this precisely:
  After "we" merge, remaining pairs:
  (l,o): 7   (o,we): 7   (we,r): 5   (we,s): 8
  (s,t): 8   (n,e): 6    (e,we): 6

Step 3 -- Merge (we,s) -> "wes" or (s,t) -> "st" (tied at 8, pick first):
  Merge (we,s) -> "wes":
  l o we r        (x5)
  l o wes t       (x2)
  n e wes t       (x6)

Step 4 -- Merge (wes,t) -> "west":
  l o we r        (x5)
  l o west        (x2)
  n e west        (x6)

...continue until target vocab size reached.
```

合并表就是分词器。编码新文本时，按学习顺序应用合并。训练语料决定哪些合并存在，而这个选择永久塑造模型看到的内容。

```mermaid
graph LR
    subgraph Training["BPE Training Loop"]
        direction TB
        T1["Start: character vocabulary"] --> T2["Count all adjacent pairs"]
        T2 --> T3["Merge most frequent pair"]
        T3 --> T4["Add merged token to vocab"]
        T4 --> T5{"Reached target\nvocab size?"}
        T5 -->|No| T2
        T5 -->|Yes| T6["Done: save merge table"]
    end
```

### 字节级 BPE（GPT-2、GPT-3、GPT-4）

标准 BPE 操作 Unicode 字符。字节级 BPE 操作原始字节（0-255）。这给你恰好 256 的基础词汇表，处理任何语言或编码，且永远不会产生未知 token。

GPT-2 引入了这种方法。基础词汇表覆盖每个可能的字节。BPE 合并在此基础上构建。OpenAI 的 tiktoken 库实现字节级 BPE，词汇表大小如下：

- GPT-2：50,257 token
- GPT-3.5/GPT-4：~100,256 token（cl100k_base 编码）
- GPT-4o：200,019 token（o200k_base 编码）

### WordPiece（BERT）

WordPiece 看起来类似 BPE，但选择合并的方式不同。它不是用原始频率，而是最大化训练数据的似然：

```
BPE merge criterion:      count(A, B)
WordPiece merge criterion: count(AB) / (count(A) * count(B))
```

BPE 问："哪对出现最频繁？" WordPiece 问："哪对一起出现的频率高于偶然预期？" 这个细微差别产生不同的词汇表。WordPiece 偏好共现令人惊讶的合并，而非仅仅频繁的合并。

WordPiece 还为延续性子词使用 "##" 前缀：

```
"unhappiness" -> ["un", "##happi", "##ness"]
"embedding"   -> ["em", "##bed", "##ding"]
```

"##" 前缀告诉你这个片段延续前一个 token。BERT 使用 30,522 token 的 WordPiece。每个 BERT 变体——DistilBERT、RoBERTa 的分词器实际上是 BPE，但 BERT 本身是 WordPiece。

### SentencePiece（Llama、T5）

SentencePiece 将输入视为原始 Unicode 字符流，包括空白字符。没有预分词步骤。没有关于词边界的语言特定规则。这使其真正语言无关——适用于中文、日文、泰文等不以空格分词的语言。

SentencePiece 支持两种算法：
- **BPE 模式**：与标准 BPE 相同的合并逻辑，应用于原始字符序列
- **Unigram 模式**：从大型词汇表开始，迭代移除对整体似然影响最小的 token。BPE 的反向操作——修剪而非合并。

Llama 2 使用 32,000 token 的 SentencePiece BPE。T5 使用 32,000 token 的 SentencePiece Unigram。注意：Llama 3 切换到基于 tiktoken 的字节级 BPE 分词器，128,256 token。

### 词汇表大小的权衡

这是有实际可衡量后果的真实工程决策。

```mermaid
graph LR
    subgraph Small["Small Vocab (32K)\ne.g., BERT, T5"]
        S1["More tokens per text"]
        S2["Longer sequences"]
        S3["Smaller embedding matrix"]
        S4["Better rare-word handling"]
    end
    subgraph Large["Large Vocab (128K+)\ne.g., Llama 3, GPT-4o"]
        L1["Fewer tokens per text"]
        L2["Shorter sequences"]
        L3["Larger embedding matrix"]
        L4["Faster inference"]
    end
```

具体数字。对于 128K 词汇表和 4,096 维嵌入，仅嵌入矩阵就有 128,000 × 4,096 = 5.24 亿参数。对于 32K 词汇表，则是 1.31 亿参数。仅分词器选择就带来 4 亿参数的差异。

但更大的词汇表更激进地压缩文本。同样的英文段落，32K 词汇表需要 100 token，128K 词汇表可能只需 70 token。这意味着生成时减少 30% 的前向传播。对于服务数百万请求的模型，这是直接的算力成本降低。

趋势明显：词汇表大小在增长。GPT-2 用 50,257。GPT-4 用 ~100K。Llama 3 用 128K。GPT-4o 用 200K。

| 模型 | 词汇表大小 | 分词器类型 | 平均每英语单词 token 数 |
|-------|-----------|----------------|---------------------------|
| BERT | 30,522 | WordPiece | ~1.4 |
| GPT-2 | 50,257 | 字节级 BPE | ~1.3 |
| Llama 2 | 32,000 | SentencePiece BPE | ~1.4 |
| GPT-4 | ~100,256 | 字节级 BPE | ~1.2 |
| Llama 3 | 128,256 | 字节级 BPE (tiktoken) | ~1.1 |
| GPT-4o | 200,019 | 字节级 BPE | ~1.0 |

### 多语言税

主要用英语训练的分词器对其他语言很残酷。GPT-2 的分词器处理韩文平均每个词 2-3 token。中文可能更糟。这意味着韩国用户的有效上下文窗口只有英语用户的一半——付同样的价格，信息密度却更低。

这就是 Llama 3 将词汇表从 32K 四倍扩展到 128K 的原因。更多 token 分配给非英语文字，意味着跨语言更公平的压缩。

## 动手实现

### 步骤 1：字符级分词器

从基础开始。字符级分词器将每个字符映射为其 Unicode 码点。无需训练。没有未知 token。直接映射。

```python
class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)
```

"hello" 变成 [104, 101, 108, 108, 111]。每个字符都是自己的 token。这是我们改进的基线。

### 步骤 2：从零实现 BPE 分词器

真正的实现。我们在原始字节上训练（像 GPT-2），统计字节对，合并最频繁的，并按顺序记录每次合并。合并表就是分词器。

```python
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def _get_pairs(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, text, num_merges):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            new_token = 256 + i
            tokens = self._merge_pair(tokens, best_pair, new_token)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        return self

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for pair, new_token in self.merges.items():
            tokens = self._merge_pair(tokens, pair, new_token)
        return tokens

    def decode(self, tokens):
        byte_sequence = b"".join(self.vocab[t] for t in tokens)
        return byte_sequence.decode("utf-8", errors="replace")
```

训练循环是 BPE 的核心：统计字节对，合并赢家，重复。每次合并减少总 token 数。经过 `num_merges` 轮后，词汇表从 256（基础字节）增长到 256 + num_merges。

编码时按学习的确切顺序应用合并。这很重要。如果合并 1 创建了 "th"，合并 5 创建了 "the"，编码必须先应用合并 1，这样 "the" 才能从合并 5 中的 "th" + "e" 形成。

解码是逆操作：在词汇表中查找每个 token ID，拼接字节，解码为 UTF-8。

### 步骤 3：编码解码往返

```python
corpus = (
    "The cat sat on the mat. The cat ate the rat. "
    "The dog sat on the log. The dog ate the frog. "
    "Natural language processing is the study of how computers "
    "understand and generate human language. "
    "Tokenization is the first step in any NLP pipeline."
)

tokenizer = BPETokenizer()
tokenizer.train(corpus, num_merges=40)

test_sentences = [
    "The cat sat on the mat.",
    "Natural language processing",
    "tokenization pipeline",
    "unhappiness",
]

for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    decoded = tokenizer.decode(encoded)
    raw_bytes = len(sentence.encode("utf-8"))
    ratio = len(encoded) / raw_bytes
    print(f"'{sentence}'")
    print(f"  Tokens: {len(encoded)} (from {raw_bytes} bytes) -- ratio: {ratio:.2f}")
    print(f"  Roundtrip: {'PASS' if decoded == sentence else 'FAIL'}")
```

压缩比告诉你分词器的效率。0.50 的比率意味着分词器将文本压缩到原始字节 token 数的一半。越低越好。在训练语料上，比率会很好。在分布外文本如 "unhappiness"（未出现在语料中）上，比率会更差——分词器对未见模式回退到字符级编码。

### 步骤 4：与 tiktoken 对比

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

texts = [
    "The cat sat on the mat.",
    "unhappiness",
    "Hello, world!",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Geschwindigkeitsbegrenzung",
]

for text in texts:
    our_tokens = tokenizer.encode(text)
    tiktoken_tokens = enc.encode(text)
    tiktoken_pieces = [enc.decode([t]) for t in tiktoken_tokens]
    print(f"'{text}'")
    print(f"  Our BPE:   {len(our_tokens)} tokens")
    print(f"  tiktoken:  {len(tiktoken_tokens)} tokens -> {tiktoken_pieces}")
```

tiktoken 使用完全相同的算法，但在数百 GB 文本上训练了 100,000 次合并。算法相同。差异在于训练数据和合并次数。你的分词器在段落上训练了 40 次合并，无法与 tiktoken 在庞大语料上的 100K 次合并竞争。但机制相同。

### 步骤 5：词汇表分析

```python
def analyze_vocabulary(tokenizer, test_texts):
    total_tokens = 0
    total_chars = 0
    token_usage = Counter()

    for text in test_texts:
        encoded = tokenizer.encode(text)
        total_tokens += len(encoded)
        total_chars += len(text)
        for t in encoded:
            token_usage[t] += 1

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Total tokens across all texts: {total_tokens}")
    print(f"Total characters: {total_chars}")
    print(f"Avg tokens per character: {total_tokens / total_chars:.2f}")

    print(f"\nMost used tokens:")
    for token_id, count in token_usage.most_common(10):
        token_bytes = tokenizer.vocab[token_id]
        display = token_bytes.decode("utf-8", errors="replace")
        print(f"  Token {token_id:4d}: '{display}' (used {count} times)")

    unused = [t for t in tokenizer.vocab if t not in token_usage]
    print(f"\nUnused tokens: {len(unused)} out of {len(tokenizer.vocab)}")
```

这揭示了词汇表中的 Zipf 分布。少数 token 占主导（空格、"the"、"e"）。大多数 token 很少使用。生产分词器针对这种分布优化——常见模式获得短 token ID，罕见模式获得更长表示。

## 使用它

你的从零 BPE 能用了。现在看看生产工具长什么样。

### tiktoken（OpenAI）

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

text = "Tokenizers convert text to integers"
tokens = enc.encode(text)
print(f"Tokens: {tokens}")
print(f"Pieces: {[enc.decode([t]) for t in tokens]}")
print(f"Roundtrip: {enc.decode(tokens)}")
```

tiktoken 用 Rust 编写，带 Python 绑定。每秒编码数百万 token。同样的 BPE 算法，工业级实现。

### Hugging Face tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(vocab_size=1000, special_tokens=["<pad>", "<eos>", "<unk>"])
tokenizer.train(["corpus.txt"], trainer)

output = tokenizer.encode("The cat sat on the mat.")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")
```

Hugging Face tokenizers 库底层也是 Rust。它在数秒内对 GB 级语料训练 BPE。训练自己的模型时就用这个。

### 加载 Llama 的分词器

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

text = "Tokenizers are the unsung heroes of LLMs"
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens)}")
print(f"Vocab size: {tokenizer.vocab_size}")

multilingual = ["Hello world", "Hola mundo", "Bonjour le monde"]
for text in multilingual:
    ids = tokenizer.encode(text)
    print(f"'{text}' -> {len(ids)} tokens")
```

Llama 3 的 128K 词汇表比 GPT-2 的 50K 词汇表显著更好地压缩非英语文本。你可以自己验证——用多种语言编码同一句子，统计 token 数。

## 交付

本课程产出 `outputs/prompt-tokenizer-analyzer.md`——一个可复用的提示词，分析任何文本和模型组合的分词效率。输入文本样本，它告诉你哪个模型的分词器处理得最好。

## 练习

1. 修改 BPE 分词器，在每次合并步骤打印词汇表。观察 "t" + "h" 如何变成 "th"，然后 "th" + "e" 如何变成 "the"。追踪常见英语词如何被逐块组装。

2. 向 BPE 分词器添加特殊 token（`<pad>`、`<eos>`、`<unk>`）。将它们分配为 ID 0、1、2，并相应偏移所有其他 token。实现一个在运行 BPE 前按空白分割的预分词步骤。

3. 实现 WordPiece 合并准则（似然比替代频率）。在相同语料上用相同合并次数训练 BPE 和 WordPiece。比较生成的词汇表——哪个产生更多语言学上有意义的子词？

4. 构建多语言分词器效率基准。取英语、西班牙语、中文、韩语、阿拉伯语各 10 个句子。用 tiktoken（cl100k_base）对每个语言分词，测量平均每字符 token 数。量化每种语言的"多语言税"。

5. 在更大的语料上训练你的 BPE 分词器（下载一篇维基百科文章）。调整合并次数，使压缩比达到 tiktoken 在同文本上的 10% 以内。这迫使你理解语料大小、合并次数和压缩质量之间的关系。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Token | "一个词" | 模型词汇表中的单位——可以是字符、子词、词或多词片段 |
| BPE | "某种压缩的东西" | 字节对编码——迭代合并最频繁的相邻 token 对，直到达到目标词汇表大小 |
| WordPiece | "BERT 的分词器" | 类似 BPE，但合并最大化似然比 count(AB)/(count(A)*count(B)) 而非原始频率 |
| SentencePiece | "一个分词器库" | 语言无关的分词器，在原始 Unicode 上操作，无需预分词，支持 BPE 和 Unigram 算法 |
| 词汇表大小 | "它认识多少词" | 唯一 token 的总数：GPT-2 有 50,257，BERT 有 30,522，Llama 3 有 128,256 |
| Fertility | "不是分词器术语" | 平均每词 token 数——衡量跨语言分词器效率的指标（1.0 完美，3.0 意味着模型多干三倍工） |
| 字节级 BPE | "GPT 的分词器" | 在原始字节（0-255）而非 Unicode 字符上操作的 BPE，保证任何输入都没有未知 token |
| 合并表 | "分词器文件" | 训练期间学习的有序字节对合并列表——这就是分词器，顺序很重要 |
| 预分词 | "按空格分割" | 子词分词前应用的规则：空白分割、数字分离、标点处理 |
| 压缩比 | "分词器效率如何" | 产生的 token 数除以输入字节数——越低意味着更好的压缩和更快的推理 |

## 延伸阅读

- [Sennrich et al., 2016 -- "Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) —— 将 1994 年的压缩算法引入 NLP 的 BPE 开创论文，奠定了现代分词的基础
- [Kudo & Richardson, 2018 -- "SentencePiece: A simple and language independent subword tokenizer"](https://arxiv.org/abs/1808.06226) —— 语言无关的分词，让多语言模型成为现实
- [OpenAI tiktoken repository](https://github.com/openai/tiktoken) —— Rust 实现的生产级 BPE，带 Python 绑定，用于 GPT-3.5/4/4o
- [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers) —— 工业级分词器训练，Rust 性能
