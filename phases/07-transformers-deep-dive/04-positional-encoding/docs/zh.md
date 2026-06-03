# 位置编码 — 正弦、RoPE、ALiBi

> Attention 是置换不变的。"The cat sat on the mat" 和 "mat the on sat cat the" 在没有位置信号时产生相同的输出。三种算法修复了这一点 —— 每种对"位置"的含义有不同的赌注。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**时间：** ~45 分钟

## 问题所在

缩放点积注意力是顺序盲的。注意力矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算得出。打乱 `X` 的行，输出的行也以同样方式被打乱。注意力内部没有任何东西关心位置。

这在词袋模型中不是 bug。但对于语言、代码、音频、视频 —— 任何顺序承载含义的事物 —— 这是致命的。

修复方法是以某种方式将位置注入到嵌入中。三个时代的答案：

1. **绝对正弦** (Vaswani 2017)。将位置 `sin/cos` 加到嵌入上。简单、无需学习、在训练长度之外外推能力差。
2. **RoPE — 旋转位置编码** (Su 2021)。将 Q 和 K 向量按与位置成比例的角度旋转。直接在点积中编码*相对*位置。2026 年占主导地位。
3. **ALiBi — 带线性偏置的注意力** (Press 2022)。完全跳过嵌入；基于距离向注意力分数添加逐头线性惩罚。优秀的长度外推能力。

截至 2026 年，基本上所有前沿开放模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦已成为历史。

## 概念

![正弦绝对编码 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦

预计算固定矩阵 `PE`，形状为 `(max_len, d_model)`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型从相位模式中学会读取位置。在 `max_len` 之外失败：当模型只看到位置 0–2047 时，没有任何东西告诉它位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（不是嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置 `pos_k` 的键应用相同的旋转。点积 `q'_m · k'_n` 变为仅 `(m - n)` 的函数。也就是说：**注意力分数仅取决于相对距离**，即使旋转是基于绝对位置的。巧妙的技巧。

RoPE 的扩展：`base` 可以被缩放（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长的上下文。Llama 3 通过这种方式从 8K 扩展到 128K 上下文。

### ALiBi

跳过嵌入技巧。直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是逐头的斜率（例如 `1 / 2^(8·h/H)`）。更近的 token 获得提升；更远的 token 受到惩罚。无训练时成本。论文显示长度外推优于正弦编码，并在原始训练长度上匹配 RoPE。

### 2026 年如何选择

| 变体 | 外推能力 | 训练成本 | 使用者 |
|---------|---------|---------|---------|
| 绝对正弦 | 差 | 无 | 原始 transformer、早期 BERT |
| 学习式绝对编码 | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 配合缩放良好 | 无 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 优秀 | 微调阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 优秀 | 无 | BLOOM、MPT、Baichuan |

RoPE 获胜是因为它嵌入注意力而不改变架构，编码相对位置，且其 `base` 超参数为长上下文微调提供了清晰的调节旋钮。

## 动手实现

### 步骤 1：正弦编码

参见 `code/main.py`。一个 4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前将其加到嵌入矩阵上。

### 步骤 2：RoPE 应用于 Q、K

RoPE 就地操作 Q 和 K。对于每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 的 Q 和位置 `n` 的 K 应用相同函数。它们的点积在每个坐标对上获得 `cos((m-n)·θ_i)` 因子。注意力免费学会相对位置。

### 步骤 3：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

将 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后 softmax。

### 步骤 4：验证 RoPE 的相对距离性质

选取两个随机向量 `a, b`。先旋转 `(pos_a, pos_b)`，再旋转 `(pos_a + k, pos_b + k)`。两个点积必须在浮点误差范围内匹配。这个性质就是 RoPE 的全部意义 —— 它对绝对偏移不变，只有相对差距重要。

## 实际使用

PyTorch 2.5+ 在 `torch.nn.functional` 中内置了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年长上下文技巧：**

- **NTK-aware 插值。** 从 4K 扩展到 16K+ 时，将 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，在长上下文上保持注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** Microsoft 2024 年的方法，使用进化搜索选取逐维度缩放因子。Phi-3-Long 使用它。
- **位置插值 + 微调。** 只需按扩展因子缩小位置并微调 1–5B token。 surprisingly effective。

## 交付

参见 `outputs/skill-positional-encoding-picker.md`。该技能为给定目标上下文长度、外推需求和训练预算的新模型选择编码策略。

## 练习

1. **简单。** 将正弦 `PE` 矩阵绘制为 `max_len=512, d=128` 的热图。确认"随着维度索引增大，条纹变宽"的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练一个小型 LM，然后在有和没有缩放的情况下测试长度 1024。测量困惑度。
3. **困难。** 在同一注意力模块中实现 ALiBi 和 RoPE。在长度为 512 的复制任务上训练一个 4 层 transformer。在测试时外推到 2048。比较退化程度。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 位置编码 | "告诉 attention 顺序" | 任何添加到嵌入或注意力中以编码位置的信号。 |
| 正弦编码 | "最原始的那个" | 以几何频率添加到嵌入上的 `sin/cos`；无法外推。 |
| RoPE | "旋转嵌入" | 按位置相关角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 向注意力分数添加 `-m·\|i-j\|`；无需嵌入，外推能力优秀。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；增大以在推理时扩展上下文。 |
| NTK-aware | "一种 RoPE 缩放技巧" | 重新缩放 `base`，使上下文扩展时高频维度不被压缩。 |
| YaRN | "那个花哨的" | 逐维度插值+外推，保持注意力熵。 |
| 外推 | "在训练长度之外有效" | 位置方案能否在训练见过的 `max_len` 之后继续正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Microsoft 的方法，Phi-3-Long 使用，在"实际使用"部分引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每种 RoPE 缩放方案（默认、线性、动态、YaRN、LongRoPE、Llama-3）的生产级实现。
