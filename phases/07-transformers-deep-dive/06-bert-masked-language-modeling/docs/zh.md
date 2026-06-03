# BERT — 掩码语言建模

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差别——却催生了长达半个世纪的各类嵌入技术。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer），Phase 5 · 02（文本表示）
**时间：** ~45 分钟

## 问题背景

2018 年，每个 NLP 任务——情感分析、命名实体识别、问答、蕴含判断——都在各自的标注数据上从头训练专属模型。当时没有预训练的"理解英语"的 checkpoint 可供微调。ELMo（2018）证明了可以用双向 LSTM 预训练上下文嵌入；它有所帮助，但未能实现泛化。

BERT（Devlin et al. 2018）提出了一个问题：如果我们拿一个 transformer encoder，在互联网上的每个句子中训练它，并强迫它从两侧上下文预测缺失的词，会怎样？然后你在下游任务上微调一个 head。参数效率的发现令人震惊。

结果是：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的每一个 NLP 排行榜。到 2020 年，地球上的每个搜索引擎、内容审核管道和语义搜索系统内部都有一个 BERT。

2026 年，仅编码器模型仍然是分类、检索和结构化提取的正确工具——它们每 token 的运行速度比解码器快 5–10 倍，其嵌入是现代每个检索技术栈的支柱。ModernBERT（2024 年 12 月）将架构推进到了 8K 上下文，采用 Flash Attention + RoPE + GeGLU。

## 核心概念

![掩码语言建模：挑选 token，将其掩码，预测原始词](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测掩码位置上的原始 token。由于编码器是双向的，预测位置 1 上的 `[MASK]` 可以使用位置 2+ 上的 `brown fox jumps`。这正是 GPT 无法做到的。

### BERT 掩码规则

在被选中用于预测的 15% token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机 token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为 `[MASK]` 在推理时永远不会出现。如果在 100% 的掩码位置训练模型预期 `[MASK]`，会在预训练和微调之间造成分布偏移。10% 的随机 + 10% 的保持不变让模型保持诚实。

### 下一句预测（NSP）——以及为何被弃用

原始 BERT 还训练了 NSP：给定两个句子 A 和 B，预测 B 是否紧跟 A。RoBERTa（2019）通过消融实验表明 NSP 有害无益。现代编码器都跳过了它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的基础组件重建了编码器模块：

| 组件 | 原始 BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置编码 | 可学习的绝对位置编码 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化 | LayerNorm | Pre-norm RMSNorm |
| 注意力 | 全稠密注意力 | 交替局部（128）+ 全局注意力 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与 2018 年的技术栈不同，它是原生 Flash Attention 的。在 8K 序列长度下，推理速度比 DeBERTa-v3 快 2–3 倍，同时 GLUE 分数更高。

### 2026 年仍然选择编码器的用例

| 任务 | 编码器优于解码器的原因 |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 每 token 更好的嵌入质量 |
| 分类（情感、意图、毒性） | 一次前向传播；无生成开销 |
| NER / token 标注 | 逐位置输出，原生双向 |
| 零样本蕴含判断（NLI） | 在编码器之上加分类器 head |
| RAG 的重排序器 | 交叉编码器打分，比 LLM 重排序器快 10 倍 |

## 动手实现

### 步骤 1：掩码逻辑

参见 `code/main.py`。函数 `create_mlm_batch` 接收一个 token ID 列表、词表大小和掩码概率。返回应用了掩码的输入 ID，以及标签（仅在掩码位置有值，其余为 -100——PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 步骤 2：在微型语料库上运行 MLM 预测

在一个包含 20 个词、200 个句子的词表上，训练一个 2 层编码器 + MLM head。不做梯度计算——我们只做前向传播的正确性检查。完整训练需要 PyTorch。

### 步骤 3：比较掩码类型

展示三路规则如何让模型在没有 `[MASK]` 的情况下仍然可用。在未掩码句子和掩码句子上分别预测。两者都应该产生合理的 token 分布，因为模型在训练中见过两种模式。

### 步骤 4：微调 head

在一个玩具情感数据集上，将 MLM head 替换为分类 head。只训练 head；编码器冻结。这是每个 BERT 应用遵循的模式。

## 实际应用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型就是微调后的 BERT。** `sentence-transformers` 模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。编码器是相同的。改变的是损失函数。

**交叉编码器重排序器也是微调后的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上进行配对分类。查询和文档之间的双向注意力正是交叉编码器比双编码器质量更高的原因。

**2026 年不应选择 BERT 的场景。** 任何生成式任务。编码器无法以自回归方式合理地生成 token。另外：任何参数低于 1B 的场景，小型解码器可以在保持灵活性的同时达到相当质量（Phi-3-Mini、Qwen2-1.5B）。

## 交付上线

参见 `outputs/skill-bert-finetuner.md`。该技能涵盖了针对新分类或提取任务的 BERT 微调范围（骨干网络选择、head 规格、数据、评估、停止条件）。

## 练习题

1. **简单。** 运行 `code/main.py` 并打印 10,000 个 token 的掩码分布。确认约 15% 被选中，其中约 80% 变为 `[MASK]`。
2. **中等。** 实现整词掩码：如果一个词被切分为多个子词，则一起掩码所有子词或不掩码。测量这是否能在 500 句语料库上提高 MLM 准确率。
3. **困难。** 在来自公开数据集的 10,000 个句子上训练一个微型（2 层，d=64）BERT。微调 `[CLS]` token 用于 SST-2 情感分析。与同等参数量的仅解码器基线比较——哪个胜出？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | "掩码语言建模" | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始词。 |
| 双向 | "两边都看" | 编码器注意力没有因果掩码——每个位置都能看到其他所有位置。 |
| `[CLS]` | "池化 token" | 添加到每个序列开头的特殊 token；其最终嵌入用作句子级表示。 |
| `[SEP]` | "分段分隔符" | 分隔成对序列（如查询/文档、句子 A/B）。 |
| NSP | "下一句预测" | BERT 的第二个预训练任务；RoBERTa 证明其无用，2019 年后被弃用。 |
| 微调 | "适配到任务" | 保持编码器大部分冻结；在其上训练一个小型 head 用于下游任务。 |
| 交叉编码器 | "重排序器" | 将查询和文档同时作为输入，输出相关性分数的 BERT。 |
| ModernBERT | "2024 年更新" | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；淘汰了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在同等计算量下优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 经典编码器参考。
