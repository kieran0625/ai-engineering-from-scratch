# 从零构建 Transformer —— 终极项目

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置要求：** 第 7 阶段 · 第 01 至 13 课。不要跳过。
**时间：** ~120 分钟

## 问题

你读遍了每篇论文。你实现了注意力机制、多头拆分、位置编码、编码器与解码器模块、BERT 和 GPT 损失、MoE、KV 缓存。现在让它们在一个真实任务上协同工作。

终极项目：端到端训练一个小型仅解码器 Transformer，用于字符级语言建模任务。它阅读莎士比亚。它生成新的莎士比亚风格文本。它足够小，可以在笔记本电脑上 10 分钟内完成训练。它足够正确，以至于换用更大的数据集和更长的训练时间就能得到真正的语言模型。

这是本课程的 "nanoGPT"。它并非原创 —— Karpathy 2023 年的 nanoGPT 教程是每个学生至少写过一次的参考实现。我们借鉴其框架，并围绕我们已涵盖的内容进行重构。

## 概念

![Transformer 从零构建框图](../assets/capstone.svg)

架构说明：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们提供的组件

- `GPTConfig` —— 统一配置所有超参数的地方。
- `MultiHeadAttention` —— 因果的、批量的，可选类 Flash 路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代 FFN。
- `Block` —— 预归一化、残差包裹的注意力 + FFN。
- `GPT` —— 嵌入、堆叠的模块、语言模型头、generate()。
- 训练循环：AdamW、余弦学习率、梯度裁剪。
- 基于莎士比亚文本的字符级分词器。

### 我们不提供的组件

- RoPE —— 在第 04 课中概念性实现。这里为简化使用学习的位置嵌入。练习要求你替换为 RoPE。
- 生成时的 KV 缓存 —— 每个生成步骤都重新计算完整前缀上的注意力。更慢但更简单。练习要求你添加 KV 缓存。
- Flash Attention —— PyTorch 2.0+ 在输入匹配时自动分派；我们使用 `F.scaled_dot_product_attention`。
- MoE —— 每个模块单个 FFN。你在第 11 课中见过 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，一个 4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从 ~4.2（随机）收敛到 ~1.5，约需 6 分钟。
- 采样输出呈现莎士比亚风格：古旧词汇、换行、类似 "ROMEO:" 的专有名词出现。
- 验证损失（文本最后 10% 的留出数据）紧密跟踪训练损失；在此规模/预算下无过拟合。

## 构建它

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。参见 `code/main.py`。脚本处理：

- 若缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器。
- 90/10 的训练/验证划分。
- 在支持的硬件上使用 bf16 自动混合精度的训练循环。
- 训练完成后的采样。

### 步骤 1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。极小的词表。适合 4 字节 vocab_size。没有 BPE，没有分词器麻烦。

### 步骤 2：模型

参见 `code/main.py`。该模块是第 05 课的标准教科书结构 —— 预归一化、RMSNorm、SwiGLU、因果 MHA。4/4/128 的参数数量：~800K。

### 步骤 3：训练循环

获取长度为 256 的 token 窗口的随机批次。前向传播。移位一位的交叉熵。反向传播。AdamW 步骤。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤 4：采样

给定提示，重复前向传播，从 top-p 对数概率中采样，追加，继续。500 个 token 后停止。

### 步骤 5：阅读输出

2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但具有莎士比亚风格。对于 ~800K 参数和笔记本电脑上 6 分钟来说，这是一个明确的胜利。

## 使用它

这个终极项目是一个参考架构。三个扩展方向，将其变为真实可用的东西：

1. **替换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词表大小从 65 跃升至 ~50,000。模型容量需要相应扩展以作补偿。
2. **在更大的语料库上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单张 A100 上处理 10B token 的 125M 参数 GPT 约需 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习将带你逐一完成。

最终你会得到一个 125M 参数的 GPT，能生成流利的英文。不是前沿模型。但相同的代码路径 —— 只是更大 —— 正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用于训练研究检查点的方法。

## 交付它

参见 `outputs/skill-transformer-review.md`。该技能会审查 Transformer 从零实现的正确性，涵盖之前全部 13 课的内容。

## 练习

1. **简单。** 运行 `code/main.py`。验证你训练模型的最终步验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000 —— 验证损失会继续改善吗？
2. **中等。** 将学习的位置嵌入替换为 RoPE。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV 缓存。使用和不使用缓存生成 500 个 token。在笔记本电脑上，实际耗时应有 5–20 倍的提升。
4. **困难。** 为模型添加第二个头，预测下一个之后的 token（MTP —— DeepSeek-V3 的多 token 预测）。联合训练。有帮助吗？
5. **困难。** 将每个模块的单个 FFN 替换为 4 专家的 MoE。路由 + top-2 路由。在匹配活跃参数的情况下，观察验证损失的变化。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| nanoGPT | "Karpathy 的教程仓库" | 最简仅解码器 Transformer 训练代码，~300 行；标准参考实现。 |
| tinyshakespeare | "标准玩具语料" | ~1.1 MB 文本；2015 年以来每个字符级语言模型教程都在使用。 |
| Tied embeddings | "共享输入/输出矩阵" | 语言模型头权重 = 词嵌入矩阵的转置；节省参数，提升质量。 |
| bf16 autocast | "训练精度技巧" | 前向/反向用 bf16，优化器状态保持 fp32；2021 年以来的标准做法。 |
| Gradient clipping | "阻止尖峰" | 全局梯度范数上限为 1.0；防止训练崩溃。 |
| Cosine LR schedule | "2020 年后的默认" | 学习率线性预热上升，然后按余弦形状衰减至峰值的 10%。 |
| MFU | "模型 FLOP 利用率" | 实际 FLOPs / 理论峰值；2026 年 dense 40%、MoE 30% 即为优秀。 |
| Val loss | "留出损失" | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典的注释实现。
