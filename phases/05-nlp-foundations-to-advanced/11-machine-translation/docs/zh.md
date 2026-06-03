# 机器翻译

> 翻译是过去三十年资助 NLP 研究、并且至今仍在资助的任务。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 5 · 10（注意力机制），阶段 5 · 04（GloVe、FastText、子词）
**时间：** ~75 分钟

## 问题

模型读取一种语言的句子，生成另一种语言的句子。长度各异。语序各异。某些源词对应多个目标词，反之亦然。习语拒绝一对一映射。"I miss you" 在法语中是 "tu me manques"——字面意思是 "你对我来说是缺失的"。没有任何词级别的对齐能经受住这种转换。

机器翻译是迫使 NLP 发明编码器-解码器、注意力机制、Transformer，最终催生整个 LLM 范式的任务。每一步进展都源于翻译质量可衡量，以及人机差距的顽固存在。

本课跳过历史回顾，教授 2026 年的实际工作流：预训练多语言编码器-解码器（NLLB-200 或 mBART）、子词分词、束搜索、BLEU 和 chrF 评估，以及那些仍会未经察觉地部署到生产环境的少数失败模式。

## 概念

![MT 流水线：分词 → 编码 → 注意力解码 → 去分词](../assets/mt-pipeline.svg)

现代 MT 是基于 Transformer 的编码器-解码器，在平行语料上训练。编码器以源语言的 tokenization 读取源文本。解码器通过交叉注意力（第 10 课）使用编码器的输出，逐个子词生成目标文本。解码使用束搜索以避免贪心解码陷阱。输出经过去分词、去大小写化，并与参考译文对比评分。

三个关键操作选择决定真实场景中的 MT 质量。

- **分词器。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词汇表是 NLLB 实现零样本语言对的关键。
- **模型大小。** NLLB-200 distilled 600M 可在笔记本上运行。NLLB-200 3.3B 是发布的生产默认配置。54.5B 是研究上限。
- **解码。** 通用内容使用束宽 4-5。长度惩罚避免输出过短。需要术语一致性时使用约束解码。

## 动手构建

### 步骤 1：调用预训练 MT 模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三个要点。`src_lang` 告诉分词器应用哪种文字系统和分词方式。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用各自的约定，不可互换。

### 步骤 2：BLEU 和 chrF

BLEU 衡量输出与参考译文之间的 n-gram 重叠。四个参考 n-gram 大小（1-4）、精确率的几何平均、对过短输出的简短惩罚。分数范围为 [0, 100]。常用但难以解释：30 BLEU 为"可用"；40 为"良好"；50 为"优秀"；1 BLEU 以内的差异属于噪声。

chrF 衡量字符级 F-score。对形态丰富的语言更敏感，因为 BLEU 会低估匹配数。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

务必使用 `sacrebleu`。它对分词进行归一化，使分数在不同论文间可比。自行实现 BLEU 计算是产生误导性基准的方式。

### 三级评估体系（2026）

现代 MT 评估使用三类互补的指标族。生产部署时至少使用两种。

- **启发式**（BLEU、chrF）。快速、基于参考、可解释、对改写不敏感。用于遗留对比和回归检测。
- **学习型**（COMET、BLEURT、BERTScore）。基于人类判断训练的神经网络模型；比较译文与源文及参考译文的语义相似度。COMET 自 2023 年以来与 MT 研究的相关性最高，是 2026 年重视质量时的生产默认选择。
- **LLM-as-judge**（无参考）。提示大模型从流畅度、充分性、语气、文化适宜性等维度评分。当评分标准设计良好时，GPT-4-as-judge 与人类一致性匹配约 80%。用于无参考存在的开放式内容。

2026 年实用栈：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，以及用于最终人工信号的提示式 LLM。在生产数据上信任任何指标之前，先用 50-100 个人工标注样本进行校准。

无参考指标（COMET-QE、BLEURT-QE、LLM-as-judge）允许在没有参考译文的情况下评估翻译，这对长尾语言对至关重要，因为参考译文往往不存在。

### 步骤 3：生产环境中的失效模式

上述工作流 80% 的时间能流畅翻译，剩余 20% 则静默失败。已命名的失效模式：

- **幻觉。** 模型编造源文中不存在的内容。在不熟悉的领域词汇中常见。症状：输出流畅，但声称了源文未陈述的事实。缓解措施：对领域术语使用约束解码、对受监管内容进行人工审核、监控输出远长于输入的情况。
- **目标语言错误。** 模型翻译成了错误的语言。NLLB 在罕见语言对上 surprisingly 容易出现此问题。缓解措施：验证 `forced_bos_token_id`，并始终在输出上使用语言识别模型检查进行解码。
- **术语漂移。** "Sign up" 在文档 1 中变成 "s'inscrire"，在文档 2 中变成 "créer un compte"。对于 UI 文本和用户可见字符串，一致性比原始质量更重要。缓解措施：词汇表约束解码或后编辑词典。
- **正式程度不匹配。** 法语的 "tu" 与 "vous"、日语的敬语等级。模型选择训练中出现更频繁的形式。对于面向客户的内容，这通常是错误的。缓解措施：如果模型支持，用正式程度标记作为提示前缀；或在仅含正式语料上微调小模型。
- **短输入长度爆炸。** 极短输入句子常产生过长的翻译，因为长度惩罚在约 5 个源 token 以下急剧失效。缓解措施：设置与源长度成比例的硬最大长度上限。

### 步骤 4：领域微调

预训练模型是通才。法律、医学或游戏对话翻译从领域平行数据的微调中获益显著。方法并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量平行示例胜过几十万条噪声网络爬取数据。训练数据质量是最大的单一生产杠杆。

## 使用

2026 年 MT 生产栈：

| 用例 | 推荐起点 |
|---------|---------------------------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短运行、低成本推理、英法/德/西 | Helsinki-NLP / Marian 模型 |
| 延迟关键的浏览器端 | ONNX 量化 Marian（~50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini 配合翻译提示 |

截至 2026 年，LLM 在若干语言对上已超越专用 MT 模型，尤其在习语内容和长上下文方面。代价是每 token 成本和延迟。当上下文长度、风格一致性或通过提示进行领域适配比吞吐量更重要时，选择 LLM。

## 部署

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 使用 `nllb-200-distilled-600M` 将一段 5 句英文段落翻译成法文，再翻译回英文。衡量往返结果与原文的接近程度。你会看到语义保留但用词漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言 ID 检查。集成到 MT 调用中，使目标语言错误的生成在返回前被捕获。
3. **困难。** 在你选择的 5,000 对领域语料上微调 `nllb-200-distilled-600M`。在保留集上测量微调前后的 BLEU。报告哪些类型的句子改善了，哪些退化了。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带简短惩罚的 n-gram 精确率。[0, 100]。 |
| chrF | 字符 F-score | 字符级 F-score。对形态丰富的语言更敏感。 |
| NMT | 神经 MT | 在平行文本上训练的 Transformer 编码器-解码器。2017 年后的默认选择。 |
| NLLB | 不让任何一种语言掉队 | Meta 的 200 种语言 MT 模型族。 |
| 约束解码 | 控制输出 | 强制特定 token 或 n-gram 在输出中出现/不出现。 |
| 幻觉 | 编造内容 | 模型输出不受源文支持。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) —— NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) —— 为何 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) —— chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) —— 实用微调指南。
