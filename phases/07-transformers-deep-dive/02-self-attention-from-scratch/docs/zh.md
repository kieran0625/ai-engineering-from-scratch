# 从零实现 Self-Attention

> Attention 是一张查询表，每个词都在问"谁对我重要？"——并自己学会答案。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3（深度学习核心）、Phase 5 第 10 课（序列到序列模型）
**时间：** ~90 分钟

## 学习目标

- 仅使用 NumPy 从零实现缩放点积 self-attention，包括 query/key/value 投影和 softmax 加权求和
- 构建多头注意力层，实现分头、并行计算 attention 并拼接结果
- 追踪 attention 矩阵如何捕捉 token 间的关系，并解释为什么除以 sqrt(d_k) 能防止 softmax 饱和
- 应用因果掩码，将双向 attention 转换为自回归（decoder 风格）attention

## 问题背景

RNN 逐个 token 处理序列。当你到达第 50 个 token 时，第 1 个 token 的信息已经被压缩了 50 轮。长距离依赖被挤压进固定大小的隐藏状态——这个瓶颈无论多少 LSTM 门控都无法完全解决。

2014 年 Bahdanau 的 attention 论文给出了解法：让 decoder 回顾每一个 encoder 位置，决定哪些对当前步重要。但它仍然依附于 RNN。2017 年的 "Attention Is All You Need" 提出了更尖锐的问题：如果 attention 是*唯一*的机制呢？没有循环。没有卷积。只有 attention。

Self-attention 让序列中的每个位置在单步并行中 attend 到其他所有位置。这正是 transformer 快速、可扩展且占据主导地位的原因。

## 核心概念

### 数据库查询类比

将 attention 想象成软性的数据库查询：

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

每个 token 生成三个向量：
- **Query (Q)**："我在找什么？"
- **Key (K)**："我包含什么？"
- **Value (V)**："如果被选中，我提供什么信息？"

Query 与所有 key 的点积产生 attention score。高分意味着"这个 key 匹配我的 query"。这些分数为 value 加权。输出是 value 的加权和。

### Q, K, V 的计算

每个 token 的 embedding 通过三个可学习的权重矩阵投影：

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

直观地看，单个 token 的计算过程：

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### Attention 矩阵

一旦获得所有 token 的 Q、K、V，attention score 就形成一个矩阵：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### 为什么要缩放？

点积随维度 d_k 增长。如果 d_k = 64，点积可能达到数十的量级，将 softmax 推入梯度消失的区域。解决方法：除以 sqrt(d_k)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这让数值保持在 softmax 能产生有效梯度的范围内。

### Softmax 将分数转为权重

Softmax 将原始分数转换为每行上的概率分布：

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

现在每个 token 有了一组权重，表示对其他每个 token 的 attend 程度。

### Value 的加权和

每个 token 的最终输出是所有 value 向量的加权和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

一行公式：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 动手实现

### 步骤 1：从零实现 Softmax

Softmax 将原始 logit 转换为概率。减去最大值以保证数值稳定性。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 步骤 2：缩放点积 attention

核心函数。接收 Q、K、V 矩阵，返回 attention 输出和权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 步骤 3：带可学习投影的 Self-Attention 类

完整的 self-attention 模块，Wq、Wk、Wv 权重矩阵使用类 Xavier 缩放初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 步骤 4：在句子上运行

为句子创建模拟 embedding，观察 attention 权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 步骤 5：用 ASCII 热力图可视化

将 attention 权重映射为字符，快速可视化。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 实际使用

PyTorch 的 `nn.MultiheadAttention` 正是我们构建的功能，外加多头拆分和输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别：多头注意力并行运行多个 attention 函数，每个使用自己的 Q、K、V 投影，维度为 d_k = d_model / n_heads，然后拼接结果。这让模型能同时关注不同类型的关系。

## 产出物

本节课产出：
- `outputs/prompt-attention-explainer.md` —— 一个通过数据库查询类比解释 attention 的 prompt

## 练习

1. 修改 `scaled_dot_product_attention`，使其接受可选的掩码矩阵，在 softmax 前将特定位置设为负无穷（这就是因果/decoder 掩码的工作原理）
2. 从头实现多头注意力：将 Q、K、V 拆分为 `n_heads` 个块，对每个块运行 attention，拼接结果，并通过最终权重矩阵 Wo 投影
3. 取两个等长的不同句子，用同一个 SelfAttention 实例处理，比较它们的 attention 模式。什么变了？什么没变？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Query (Q) | "问题向量" | 输入的可学习投影，表示该 token 正在寻找什么信息 |
| Key (K) | "标签向量" | 可学习投影，表示该 token 包含什么信息，与 query 匹配 |
| Value (V) | "内容向量" | 可学习投影，携带实际信息，根据 attention score 聚合 |
| 缩放点积 attention | "attention 公式" | softmax(QK^T / sqrt(d_k)) @ V —— 缩放防止高维 softmax 饱和 |
| Self-attention | "token 看自己和其他 token" | Q、K、V 都来自同一序列的 attention，让每个位置能 attend 到其他所有位置 |
| Attention 权重 | "关注程度" | 对位置的概率分布，由缩放点积的 softmax 产生 |
| 多头注意力 | "并行 attention" | 用不同投影运行多个 attention 函数，拼接结果以获得更丰富的表示 |

## 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) —— 原始 transformer 论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) —— 最直观的完整架构图解
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 逐行 PyTorch 实现与讲解
