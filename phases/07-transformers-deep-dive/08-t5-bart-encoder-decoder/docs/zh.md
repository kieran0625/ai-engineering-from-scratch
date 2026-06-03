# T5、BART — 编码器-解码器模型

> 编码器负责理解。解码器负责生成。把它们组合起来，你就得到了一个为输入→输出任务而设计的模型：翻译、摘要、改写、转录。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）、Phase 7 · 07（GPT）
**时间：** ~45 分钟

## 问题

仅解码器的 GPT 和仅编码器的 BERT 各自将 2017 年的架构简化，服务于不同的目标。但许多任务天然就是输入-输出形式的：

- 翻译：英语 → 法语。
- 摘要：5,000 token 的文章 → 200 token 的摘要。
- 语音识别：音频 token → 文本 token。
- 结构化抽取：散文 → JSON。

对于这些任务，编码器-解码器是最合适的选择。编码器生成源文本的密集表示。解码器生成输出，并在每一步交叉注意力到该表示。训练时在输出侧进行逐位移位。与 GPT 使用相同的损失函数，只是以编码器输出为条件。

两篇论文定义了现代范式：

1. **T5**（Raffel 等，2019）。"Text-to-Text Transfer Transformer"。每个 NLP 任务都被重新定义为文本输入、文本输出。单一架构、单一词表、单一损失函数。在掩码跨度预测上预训练（在输入中损坏跨度，在输出中解码它们）。
2. **BART**（Lewis 等，2019）。"Bidirectional and Auto-Regressive Transformer"。去噪自编码器：以多种方式损坏输入（打乱、掩码、删除、旋转），让解码器重建原始文本。

在 2026 年，编码器-解码器架构在输入结构重要的场景中继续存在：

- Whisper（语音 → 文本）。
- Google 的翻译技术栈。
- 一些具有明确上下文和编辑结构的代码补全/修复模型。
- Flan-T5 及其变体，用于结构化推理任务。

仅解码器模型赢得了聚光灯，但编码器-解码器从未消失。

## 概念

![带交叉注意力的编码器-解码器](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键在于，编码器对每个输入只运行一次。解码器自回归地运行，但在每一步都交叉注意力到*同一个*编码器输出。缓存编码器输出对于长输入来说是一个免费的加速。

### T5 预训练 — 跨度损坏

从输入中随机选取跨度（平均长度 3 个 token，占总量的 15%）。用唯一的哨兵替换每个跨度：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出带哨兵前缀的损坏跨度：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

这比预测整个序列的信号更廉价。在 T5 论文的消融实验中，与 MLM（BERT）和 prefix-LM（UniLM）相比具有竞争力。

### BART 预训练 — 多噪声去噪

BART 尝试了五种噪声函数：

1. Token 掩码。
2. Token 删除。
3. 文本填充（掩码一个跨度，解码器插入正确长度）。
4. 句子排列。
5. 文档旋转。

文本填充 + 句子排列的组合产生了最好的下游效果。解码器始终重建原始文本。BART 的输出是完整序列，而不仅仅是损坏的跨度 —— 因此预训练计算量高于 T5。

### 推理

与 GPT 相同的自回归生成。应用贪心/束搜索/top-p 采样。束搜索（宽度 4–5）是翻译和摘要的标准做法，因为输出分布比对话更窄。

### 2026 年如何选择每种变体

| 任务 | 编码器-解码器？ | 原因 |
|------|----------------|------|
| 翻译 | 是，通常 | 明确的源序列；固定的输出分布；束搜索有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器处理音频特征 |
| 对话/推理 | 否，仅解码器 | 没有持久的"输入" —— 对话本身就是序列 |
| 代码补全 | 通常否 | 长上下文下的仅解码器模型胜出；Qwen 2.5 Coder 等代码模型是仅解码器的 |
| 摘要 | 两者皆可 | BART、PEGASUS 击败了早期的仅解码器基线；现代仅解码器 LLM 与之持平 |
| 结构化抽取 | 两者皆可 | T5 很简洁，因为"文本 → 文本"可以吸收任何输出格式 |

自 ~2022 年以来的趋势：仅解码器模型接管了编码器-解码器曾经擅长的任务，因为（a）经过指令微调的仅解码器 LLM 可以通过提示泛化到任何任务，（b）单一架构比两种架构更容易扩展，（c）RLHF 假设使用解码器。编码器-解码器在输入模态不同（语音、图像）或束搜索质量重要的场景中保持优势。

## 动手实现

参见 `code/main.py`。我们为一个小型语料库实现 T5 风格的跨度损坏 —— 这是本课最有用的单个部分，因为它出现在此后每个编码器-解码器预训练方案中。

### 步骤 1：跨度损坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 的约定：`<sent0> span0 <sent1> span1 ...`。损坏的输入在跨度位置将未改变的 token 与哨兵 token 交错排列。

### 步骤 2：验证往返

给定损坏的输入和目标，重建原始句子。如果你的损坏是可逆的，前向传播就是良定义的。这是一个健全性检查 —— 真实训练永远不会这样做，但测试成本低廉，能捕获跨度记录中的差一错误。

### 步骤 3：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称放入输入文本中。同一个模型处理数十种任务，因为每个任务都是文本输入、文本输出。在 2026 年，这一模式已被经过指令微调的仅解码器模型泛化，但 T5 首次将其系统化。

## 实战应用

参见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为新任务在编码器-解码器和仅解码器之间做出选择。

## 练习

1. **简单。** 运行 `code/main.py`，对 30 个 token 的句子应用跨度损坏，验证将非哨兵源 token 与解码后的目标跨度拼接能否重建原始文本。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机跨度，解码器必须推断正确的跨度长度和内容。展示一个示例。
3. **困难。** 在小型英语 → 儿童黑话（pig-Latin）语料库（200 对）上微调 `flan-t5-small`。在保留的 50 对测试集上测量 BLEU。与在相同数据和相同计算量下微调 `Llama-3.2-1B` 进行比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Encoder-decoder | "Seq2seq transformer" | 两个堆栈：用于输入的双向编码器，带交叉注意力的因果解码器用于输出。 |
| Cross-attention | "源与目标对话的地方" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一通道。 |
| Span corruption | "T5 的预训练技巧" | 用哨兵 token 替换随机跨度；解码器输出这些跨度。 |
| Denoising objective | "BART 的游戏" | 对输入应用噪声函数，训练解码器重建干净序列。 |
| Sentinel token | "`<extra_id_N>` 占位符" | 在源中标记损坏跨度、在目标中重新标记它们的特殊 token。 |
| Flan | "指令微调的 T5" | 在 >1,800 个任务上微调的 T5；使编码器-解码器在指令遵循方面具备竞争力。 |
| Beam search | "解码策略" | 每步保留 top-k 个部分序列；翻译/摘要的标准做法。 |
| Teacher forcing | "训练时的输入" | 训练期间，向解码器馈入真实的上一个输出 token，而非采样得到的 token。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年典型的编码器-解码器模型。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
