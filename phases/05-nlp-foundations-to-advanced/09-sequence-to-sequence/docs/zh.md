# 序列到序列模型

> 两个 RNN 假装自己是翻译器。它们遇到的瓶颈正是注意力机制存在的原因。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 08（用于文本的 CNN + RNN），Phase 3 · 11（PyTorch 入门）
**时间：** ~75 分钟

## 问题所在

分类将变长序列映射到单个标签。翻译将变长序列映射到另一个变长序列。输入和输出使用不同的词汇表，可能是不同的语言，长度也不一定相等。

seq2seq 架构（Sutskever、Vinyals、Le，2014）用一个刻意简单的方案解决了这个问题。两个 RNN。一个读取源句子并生成固定大小的上下文向量。另一个读取该向量并逐个 token 生成目标句子。与你在第 08 课写的代码相同，只是以不同方式拼接在一起。

研究这个架构有两个原因。首先，上下文向量瓶颈是 NLP 中最具教学价值的失败案例。它解释了注意力机制和 transformers 为何如此重要。其次，训练方法（teacher forcing、scheduled sampling、推理时的 beam search）仍然适用于包括 LLM 在内的每一个现代生成系统。

## 核心概念

**编码器（Encoder）。** 读取源句子的 RNN。它的最终隐藏状态就是**上下文向量**——对整个输入的固定大小摘要。理论上不会丢失源信息。

**解码器（Decoder）。** 从上下文向量初始化的另一个 RNN。每一步它接收先前生成的 token 作为输入，并在目标词汇表上产生一个分布。采样或取 argmax 来选择下一个 token。将其反馈回去。重复直到生成 `<EOS>` token 或达到最大长度。

**训练：** 在解码器的每一步计算交叉熵损失，对序列求和。通过两个网络进行标准的反向传播。

**Teacher forcing。** 训练期间，解码器在第 `t` 步的输入是位置 `t-1` 的*真实* token，而不是解码器自己之前的预测。这稳定了训练；没有它，早期错误会级联扩散，模型永远无法学会。推理时，必须使用模型自己的预测，因此始终存在训练/推理分布差距。这个差距称为**曝光偏差（exposure bias）**。

**瓶颈。** 编码器学到的关于源的所有信息必须被压缩到那一个上下文向量中。长句子丢失细节。罕见词被模糊。语序调整（chat noir vs. black cat）必须死记硬背，而非计算得出。

注意力机制（第 10 课）通过让解码器查看*每一个*编码器隐藏状态来解决这个问题，而不仅仅是最后一个。这就是全部要点。

## 动手实现

### 步骤 1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]`——每个输入位置对应一个隐藏状态。`hidden` 的形状为 `[1, batch, hidden_dim]`——最后一步。第 08 课说"对输出进行池化以用于分类"。这里我们保留最后一个隐藏状态作为上下文向量，忽略每步的输出。

### 步骤 2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器每次调用一步。输入：一批单 token 和当前隐藏状态。输出：下一个 token 的词汇表 logits 和更新后的隐藏状态。

### 步骤 3：带 teacher forcing 的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得注意的参数。`ignore_index=0` 跳过对 padding token 的损失计算。`teacher_forcing_ratio` 是在每一步使用真实 token 还是模型预测的概率。从 1.0 开始（完全 teacher forcing），在训练过程中逐渐降低到 ~0.5，以缩小曝光偏差差距。

### 步骤 4：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每一步选择概率最高的 token。它可能会偏离：一旦确定某个 token，就无法撤回。**Beam search** 保留前 `k` 个部分序列，最后选择得分最高的完整序列。Beam width 3-5 是标准设置。

### 步骤 5：瓶颈演示

在玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损地记住 40 个 token 的输入。信息存在于编码器的每一步，但解码器只能看到最后一个状态。注意力机制直接解决了这个问题。

## 实际应用

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供了完整的编码器-解码器模型（BART、T5、mBART、NLLB），在数十亿 token 上训练。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器用 transformers 替代了 RNN。高层结构（编码器、解码器、逐 token 生成）与 2014 年的 seq2seq 论文完全相同。每个块内部的机制不同。

### 何时仍应使用基于 RNN 的 seq2seq

新项目几乎从不。具体例外：

- 流式翻译，需要逐 token 消费输入且内存受限。
- 设备端文本生成，transformer 的内存成本过高。
- 教学。理解编码器-解码器瓶颈是理解 transformers 为何胜出的最快途径。

### 曝光偏差及其缓解方法

- **Scheduled sampling。** 在训练过程中逐渐降低 teacher forcing 比例，让模型学会从自己的错误中恢复。
- **Minimum risk training。** 在句子级 BLEU 分数而非 token 级交叉熵上训练。更接近实际需求。
- **Reinforcement learning fine-tuning。** 用指标奖励序列生成器。现代 LLM RLHF 中使用。

这三种方法仍然适用于基于 transformer 的生成。

## 交付

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现玩具复制任务。在输入-输出对上训练 GRU seq2seq，其中目标等于源。测量长度 5、10、20 时的准确率。复现瓶颈现象。
2. **中等。** 添加 beam width 为 3 的 beam search 解码。在小型平行语料库上测量与贪心解码相比的 BLEU。记录 beam search 获胜的地方（通常是最后几个 token）以及没有区别的地方。
3. **困难。** 在 10k 对的复述数据集上微调 `facebook/bart-base`。比较微调模型与基础模型在 beam-4 输出上的差异，使用 held-out 输入。报告 BLEU 并挑选 10 个定性示例。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Encoder | 输入 RNN | 读取源。产生每步隐藏状态和最终上下文向量。 |
| Decoder | 输出 RNN | 从上下文向量初始化。逐个生成目标 token。 |
| Context vector | 摘要 | 编码器最终隐藏状态。固定大小。注意力机制解决的瓶颈。 |
| Teacher forcing | 使用真实 token | 训练时输入前一步的真实 token。稳定学习。 |
| Exposure bias | 训练/测试差距 | 模型在真实 token 上训练，从未练习过从自己的错误中恢复。 |
| Beam search | 更好的解码 | 每步保留前 k 个部分序列，而非贪心确定。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) —— 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) —— 引入 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 注意力论文。本课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) —— 可构建的 seq2seq + 注意力代码。
