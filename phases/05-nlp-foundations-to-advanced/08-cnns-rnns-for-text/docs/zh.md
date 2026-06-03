# 用于文本的 CNN 与 RNN

> 卷积学习 n-gram。循环网络记忆。两者都被注意力机制取代。但在资源受限的硬件上仍然重要。

**类型：** Build
**语言：** Python
**前置条件：** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**时间：** ~75 分钟

## 问题

TF-IDF 和 Word2Vec 生成的是忽略词序的扁平向量。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。而词序有时承载着关键信号。

在 Transformer 出现之前，有两类架构填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 对词嵌入序列应用一维卷积。宽度为 3 的滤波器是一个可学习的 trigram 检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）以检测多尺度模式。通过最大池化得到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 逐个处理 token，维护一个向前传递信息的隐藏状态。顺序的、带记忆的、支持可变输入长度。从 2014 年到 2017 年主导了序列建模，然后注意力机制出现了。

本节课构建两者，然后指出促使注意力机制诞生的那个失败。

## 概念

**TextCNN** (Kim, 2014)。Token 被嵌入。宽度为 `k` 的一维卷积在连续的 `k`-gram 嵌入上滑动滤波器，生成特征图。对该图进行全局最大池化，选取最强激活。拼接来自多个滤波器宽度的池化输出。送入分类头。

为什么有效。滤波器是一个可学习的 n-gram。最大池化是位置不变的，因此 "not good" 无论在评论开头还是中间都会触发相同的特征。三个滤波器宽度各 100 个滤波器，就得到了 300 个学习到的 n-gram 检测器。训练是并行的；没有顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。跨时间共享 `W`、`U`、`b`。时间 `T` 的隐藏状态是整个前缀的摘要。对于分类，在 `h_1 ... h_T` 上进行池化（最大、平均或最后）。

普通 RNN 存在梯度消失问题。**LSTM** 增加了门控来决定遗忘什么、存储什么、输出什么，从而在长序列上稳定梯度。**GRU** 将 LSTM 简化为两个门；参数更少，性能相似。

**双向 RNN** 同时运行一个前向和一个后向 RNN，拼接隐藏状态。每个 token 的表示都能看到左右两侧的上下文。对标注任务至关重要。

## 构建

### 步骤 1：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。无论输入长度如何，池化输出都是固定大小的。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列进行最大池化，而不是取最后状态。对于分类，最大池化通常优于取最后隐藏状态，因为在长序列末尾的信息往往会主导最后状态。

### 步骤 3：梯度消失演示（直观理解）

没有门控的普通 RNN 无法学习长距离依赖。考虑一个玩具任务：预测 token `A` 是否出现在序列中的任何位置。如果 `A` 位于位置 1，而序列长度为 100，那么损失产生的梯度必须回溯经过 99 次循环权重的乘法。如果权重小于 1，梯度消失。如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 通过**细胞状态**解决这个问题，它只在网络中通过加法交互传播（遗忘门对其进行乘法缩放，但梯度仍能沿"高速公路"流动）。GRU 用更少的参数实现了类似效果。两者都能在 100+ 步的序列上稳定训练。

### 步骤 4：为什么这仍然不够

即使有了 LSTM，三个问题依然存在。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 个串行的前向/反向步骤。无法跨时间并行。
2. **编码器-解码器架构中的固定大小上下文向量。** 解码器只能看到编码器的最终隐藏状态，这是对整个输入的压缩。长输入会丢失细节。第 09 课直接讨论这一点。
3. **远距离依赖的精度上限。** LSTM 优于普通 RNN，但在 200+ 步的距离上传输特定信息仍然困难。

注意力机制解决了所有三个问题。Transformer 完全抛弃了循环。第 10 课是转折点。

## 使用

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 是生产就绪的。训练代码是标准的。

Hugging Face 提供预训练嵌入，你可以将其作为输入层接入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用场景清单。

- **边缘 / 设备端推理。** 使用 GloVe 嵌入的 TextCNN 比 Transformer 小 10-100 倍。如果你的部署目标是手机，这就是首选方案。
- **流式 / 在线分类。** RNN 逐个处理 token；Transformer 需要完整序列。对于实时流入的文本，LSTM 仍然占优。
- **用于基线的微型模型。** 在新任务上快速迭代。用 CPU 在 5 分钟内训练一个 TextCNN。
- **有限数据的序列标注。** BiLSTM-CRF（第 06 课）对于 1k-10k 标注句子的 NER 仍然是生产级架构。

其他一切情况都交给 Transformer。

## 部署

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在一个 3 类玩具数据集上训练 TextCNN（自己构造数据）。验证滤波器宽度 (2, 3, 4) 在平均 F1 上是否优于单一宽度 (3)。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后状态池化。在小型数据集上比较；记录哪种池化胜出并假设原因。
3. **困难。** 构建一个 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的纯 CRF 基线以及 BERT 微调进行比较。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| TextCNN | 用于文本的 CNN | 在词嵌入上堆叠一维卷积，配合全局最大池化。Kim (2014)。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 增加输入 / 遗忘 / 输出门 + 细胞状态。在长序列上稳定训练。 |
| GRU | 简化版 LSTM | 两个门而非三个。精度相似，参数更少。 |
| 双向 | 两个方向 | 前向 + 后向 RNN 拼接。每个 token 都能看到其上下文两侧。 |
| 梯度消失 | 训练信号衰减 | 普通 RNN 中反复乘以 <1 的权重，使早期步骤的梯度实际上变为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。易读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。出乎意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 对所有人变得易懂的图解。
