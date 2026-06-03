# 注意力机制 — 突破性进展

> 解码器不再眯着眼看压缩后的摘要，而是开始查看整个源序列。此后的所有进展都是注意力加工程化。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 09（序列到序列模型）
**时间：** ~45 分钟

## 问题所在

第 09 课以一个克制的失败收尾。一个 GRU 编码器-解码器在玩具复制任务上，序列长度从 5 时的 89% 准确率暴跌到长度 80 时接近随机猜测。原因是结构性的，不是训练 bug：编码器提取的每一条信息都必须塞进一个固定大小的隐藏状态，而解码器永远看不到其他任何东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行代码的修复方案。不再只给解码器最终的编码器状态，而是保留每一个编码器状态。在解码器的每一步，计算编码器状态的加权平均，权重表示"解码器现在需要看多长编码器位置 `i` 一眼？"这个加权平均就是上下文，而且每一步都会变化。

这就是全部思想。Transformer 扩展了它。自注意力将其应用到单个序列上。多头注意力并行运行它。但 2014 年的版本已经打破了瓶颈，一旦掌握了这个概念，转向 Transformer 就是工程问题，而非概念飞跃。

## 核心概念

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在解码器每一步 `t`：

1. 用前一个解码器隐藏状态 `s_{t-1}` 作为 **query（查询）**。
2. 将其与每个编码器隐藏状态 `h_1, ..., h_T` 打分。每个编码器位置一个标量。
3. 对分数做 softmax 得到注意力权重 `α_{t,1}, ..., α_{t,T}`，和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。编码器状态的加权平均。
5. 解码器将 `c_t` 加上前一个输出 token，生成下一个 token。

加权平均是关键。当解码器需要将 "Je" 翻译为 "I" 时，它给 "Je" 上方的编码器状态高权重，其他低权重。当需要 "not" 时，它给 "pas" 高权重。上下文向量每一步都在重塑。

## 维度（让所有人栽跟头的地方）

这是每个注意力实现第一次都会出错的地方。慢慢读。

| 项目 | 维度 | 说明 |
|------|------|------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 做 softmax 后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态同维度 |

**Bahdanau（加性）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的维度是 `(d_s,)`，`h_i` 的维度是 `(d_h,)`。
- `W_a` 的维度是 `(d_attn, d_s)`。`U_a` 的维度是 `(d_attn, d_h)`。
- 它们在 tanh 内部的和的维度是 `(d_attn,)`。
- `v_α` 的维度是 `(d_attn,)`。与 `v_α` 的内积坍缩为标量。**这就是 `v_α` 的作用。** 它不是魔法。它是将注意力维度向量投影为标量分数的变换。

**Luong（乘性）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬性约束。如果编码器是双向的，跳过这个。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 的维度为 `(d_s, d_h)`。消除了等维约束。
- `concat`：本质上就是 Bahdanau 形式。由于前两个更便宜，很少使用。

**一个值得指出的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的解码器状态）。Luong 使用 `s_t`（生成当前词*之后*的状态）。混淆它们会产生极难调试的微妙错误梯度。选定一篇论文，坚持它的约定。

## 动手实现

### 步骤 1：加性（Bahdanau）注意力

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表格检查你的维度。`encoder_states` 的维度是 `(T_enc, d_h)`。`projected_enc` 的维度是 `(T_enc, d_attn)`。`projected_dec` 的维度是 `(d_attn,)` 并进行广播。`combined` 的维度是 `(T_enc, d_attn)`。`scores` 的维度是 `(T_enc,)`。`weights` 的维度是 `(T_enc,)`。`context` 的维度是 `(d_h,)`。完成。

### 步骤 2：Luong 点积与 general 形式

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每种三行代码。这就是 Luong 的论文能火的原因。大多数任务上准确率相同，代码量少很多。

### 步骤 3：一个完整的数值示例

给定三个编码器状态（大致对应 "cat"、"sat"、"mat"）和一个与第一个最对齐的解码器状态，注意力分布集中在位置 0。如果解码器状态转向与最后一个对齐，注意力就移到位置 2。上下文向量随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行获胜。然后把解码器状态移近第三个编码器状态，观察权重变化。就是这样。注意力就是显式的对齐。

### 步骤 4：为什么这是通往 Transformer 的桥梁

把上面的语言翻译成 Q/K/V：

- **Query** = 解码器状态 `s_{t-1}`
- **Key** = 编码器状态（我们与之打分的对象）
- **Value** = 编码器状态（我们加权求和的对象）

在经典注意力中，key 和 value 是同一个东西。自注意力将它们分开：你可以用序列自身查询自身，对 K 和 V 使用不同的可学习投影。多头注意力用不同的可学习投影并行运行。Transformer 将整个过程堆叠多次并抛弃 RNN。

数学是一样的。维度是一样的。从 Bahdanau 注意力到缩放点积注意力的教学跳跃，大部分只是符号变化。

## 使用现成的

PyTorch 和 TensorFlow 直接内置了注意力。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。Query batch 为 5 个位置，key/value batch 为 10 个位置，每个 128 维，8 个头。`output` 是新的、增强上下文的 query。`weights` 是 5×10 的对齐矩阵，可以可视化。

### 经典注意力仍然重要的场景

- 教学。单头、单层、基于 RNN 的版本让每个概念都清晰可见。
- Transformer 放不下的设备端序列任务。
- 2014-2017 年的任何论文。不懂 Bahdanau 的约定，你就会误读。
- MT 中的细粒度对齐分析。原始注意力权重即使在 Transformer 模型上也是可解释性工具，阅读它们需要知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来可解释。它们是跨位置求和为 1 的权重；你可以画出来；高值意味着"看了这里"。审稿人喜欢它们。

它们没有看起来那么可解释。Jain 和 Wallace（2019）表明，对于某些任务，注意力分布可以被置换或替换为任意替代而不改变模型预测。永远不要在没有消融或反事实检验的情况下，将注意力权重作为推理的证据。

## 提交代码

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 `softmax` 掩码，让编码器中的 padding token 获得零注意力权重。在可变长度序列的 batch 上测试。
2. **中等。** 为 Luong `general` 形式添加多头注意力。将 `d_h` 分成 `n_heads` 组，每头分别运行注意力，然后拼接。验证单头情况与之前的实现一致。
3. **困难。** 在 09 课的玩具复制任务上，训练一个带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率 vs 序列长度。与无注意力基线对比。你应该看到随着长度增长，差距扩大，确认注意力消除了瓶颈。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Attention | 看东西 | 对 value 序列的加权平均，权重由 query-key 相似度计算。 |
| Query, Key, Value | QKV | 三个投影：Q 提问，K 是被匹配的对象，V 是被返回的内容。 |
| Additive attention | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | 分数为 `q^T k` 或 `q^T W k`。更便宜，大多数任务准确率相同。 |
| Alignment matrix | 那张漂亮的图 | 注意力权重构成的 `(T_dec, T_enc)` 网格。阅读它以查看模型关注了哪里。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种分数变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性警示。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可运行的 PyTorch 逐步讲解。
