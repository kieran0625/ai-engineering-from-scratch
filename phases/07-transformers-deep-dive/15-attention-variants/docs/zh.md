# 注意力变体 — 滑动窗口、稀疏、差分

> 全注意力是一个圆。每个 token 都能看到每个 token，而内存为此付出代价。四种变体弯曲了圆的形状，挽回了一半成本。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**时间：** ~60 分钟

## 问题

全注意力在序列长度上的内存成本为 `O(N²)`，计算成本为 `O(N²)`。对于 128K 上下文的 Llama 3 70B，每层有 160 亿个注意力条目，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但并未改变算术成本 — 每个 token 仍然要 attend 到其他每个 token。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **滑动窗口注意力（SWA)。** 每个 token 只 attend 固定窗口内的邻居，而非完整前缀。内存和计算降至 `O(N · W)`，其中 `W` 为窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **稀疏 / 分块注意力。** 只有选定的配对 `(i, j)` 会被计算分数；其余强制设为零权重。Longformer、BigBird、OpenAI sparse transformer。
3. **差分注意力。** 用独立的 Q/K 投影计算两个注意力图，相互相减。消除将权重泄露给前几个 token 的"注意力汇"。Microsoft 的 DIFF Transformer (2024)。

这些可以共存。2026 年的前沿模型通常混合使用它们：大多数层是 SWA-1024，每五层是全局全注意力，少数是差分注意力头来清理检索。Gemma 3 的 5:1 SWA 与全局注意力比例是当前教科书级默认配置。

## 概念

### 滑动窗口注意力（SWA）

位置 `i` 处的每个 query 只 attend `[i - W, i]` 中的位置（因果 SWA）或 `[i - W/2, i + W/2]`（双向）。窗口外的 token 在分数矩阵中获得 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵期望中有 1024 × 8192 个非零行 — 8 倍缩减。

**KV cache 随 SWA 缩小。** 每层只需保留 K 和 V 的最后 `W` 个 token。对于类 Gemma-3 配置（1024 窗口，128K 上下文），KV cache 降低 128 倍。

**质量代价。** 纯 SWA transformer 难以进行长距离检索。解决方案：将 SWA 层与全注意力层交错。Gemma 3 使用 5:1 的 SWA:全局比例。Mistral 7B 使用因果 SWA 堆栈，信息通过重叠窗口"向前流动" — 每层将有效感受野扩展 `W`，经过 `L` 层后模型可以 attend 回 `L × W` 个 token。

### 稀疏 / 分块注意力

预先选择 `N × N` 稀疏模式。三种经典形状：

- **局部 + 步进（OpenAI sparse transformer）。** Attend 最近的 `W` 个 token，以及此前每第 `stride` 个 token。以 `O(N · sqrt(N))` 的计算量同时捕获局部和远程信息。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（例如 `[CLS]`），它们 attend 所有人且被所有人 attend + 随机稀疏连接。在匹配质量下实现 2 倍上下文。
- **原生稀疏注意力（DeepSeek, 2025）。** 学习 `(Q, K)` 的哪些块重要；在 kernel 层面跳过零块。兼容 FlashAttention。

稀疏注意力是 kernel 工程的故事。数学很简单（mask 分数矩阵）；收益来自从不将零项加载到 SRAM 中。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer, 2024）

常规注意力存在"注意力汇"问题：softmax 强制每行和为 1，因此不想特别 attend 任何内容的 token 会将权重倾倒给第一个 token（或前几个）。这窃取了本应分配给真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来解决：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 为可学习标量（通常 0.5–0.8）。A1 捕获真实内容权重；A2 捕获注意力汇。减法抵消注意力汇，将权重重新分配给相关 token。

报告结果（Microsoft 2024）：困惑度降低 5–10%，相同训练长度下有效上下文延长 1.5–2 倍，大海捞针检索更精确。

### 变体对比

| 变体 | 计算量 | KV cache | 与全注意力相比质量 | 生产使用 |
|---------|---------|----------|-----------------|----------------|
| 全注意力 | O(N²) | 每层 O(N) | 基线 | 每个模型的默认层 |
| SWA（窗口 1024） | O(N·W) | 每层 O(W) | -0.1 ppl，配合全局层效果良好 | Gemma 2/3, Phi-3-Long |
| 局部 + 步进稀疏 | O(N·√N) | 混合 | 与 SWA 相近 | OpenAI sparse transformer, Longformer |
| BigBird（局部 + 全局 + 随机） | 约 O(N) | 混合 | 2 倍上下文下匹配全注意力 | 早期长上下文 BERT |
| 原生稀疏（DeepSeek-V3.2） | O(N · 活跃比例) | O(N) | 差距在 0.05 ppl 内 | DeepSeek-V3.2, 2025 |
| 差分注意力 | O(2·N²) | O(2N) | -5% 至 -10% ppl | DIFF Transformer, 2026 年初模型 |

## 动手实现

参见 `code/main.py`。我们实现了一个因果 mask 比较器，在 toy 序列上并排展示全注意力、SWA、局部+步进和差分注意力。

### 步骤 1：全因果 mask（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

来自第 07 课的基线。下三角；对角线以上为零权重。

### 步骤 2：滑动窗口因果 mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数 — `window`。当 `window >= n` 时，恢复全因果注意力。当 `window = 1` 时，每个 token 只 attend 自身。

### 步骤 3：局部 + 步进稀疏 mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

稠密局部窗口加上每第 `stride` 个 token 回溯到序列开头。感受野随额外层数以对数步增长。

### 步骤 4：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力计算，用可学习混合系数相减。代码中我们比较单注意力与差分注意力的注意力汇热图，观察汇的消失。

### 步骤 5：KV cache 大小

在 `N = 131072` 下打印每层的 cache 大小。SWA 和稀疏变体降低 10–100 倍。差分注意力翻倍。有意识地支付你的内存账单。

## 使用

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受 mask 函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这编译为自定义 Triton kernel。对于常见模式达到 FlashAttention-3 速度的 10% 以内，且 mask 函数是 Python 可调用对象。

**何时选择每种方案：**

- **纯全注意力** — 每层最多 ~16K 上下文，或检索质量至关重要时。
- **SWA + 全局混合** — 长上下文（>32K），训练和推理受内存限制。2026 年 32K 以上的默认方案。
- **稀疏分块注意力** — 自定义 kernel，自定义模式。保留给专业工作负载（检索、音频）。
- **差分注意力** — 任何注意力汇污染有害的工作负载（长上下文 RAG、大海捞针）。

## 交付

参见 `outputs/skill-attention-variant-picker.md`。该技能根据目标上下文长度、检索需求以及训练/推理计算 profile，为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 将每行最后 4 个 token 之外的所有项置零。验证 `window=n` 逐位复现全因果注意力。
2. **中等。** 在第 07 课 capstone 之上用 `window=1024` 实现因果 SWA。在 tinyshakespeare 上训练 1,000 步。与全注意力相比验证损失下降多少？峰值内存降低多少？
3. **困难。** 在 capstone 模型中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层全局）。在匹配参数下与纯 SWA 和纯全局基线比较损失、内存和生成质量。
4. **困难。** 实现每层 head 有可学习 `λ` 的差分注意力。在合成检索任务上训练（一个 needle，2,000 个 distractor）。在匹配参数下与单注意力基线比较检索准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 滑动窗口注意力（SWA） | "局部注意力" | 每个 query attend 最近的 `W` 个 token；KV cache 缩小到 `O(W)`。 |
| 有效感受野 | "模型能看多远" | 在 `L` 层 SWA 堆栈中，窗口为 `W` 时，最多可达 `L × W` 个 token。 |
| Longformer / BigBird | "局部 + 全局 + 随机" | 稀疏模式，少量始终 attend 的全局 token；早期长上下文方案。 |
| 原生稀疏注意力 | "DeepSeek 的 kernel 技巧" | 学习块级稀疏性；在 kernel 层面跳过零块同时保持质量。 |
| 差分注意力 | "两个图，一个相减" | DIFF Transformer：从第一个注意力图中减去可学习 `λ` 倍的第二个注意力图，以消除注意力汇。 |
| 注意力汇 | "权重泄露给 token 0" | Softmax 归一化强制行和为 1；无信息的 query 将权重倾倒在位置 0。 |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API，将任意 mask 函数编译为 FlashAttention 形态的 kernel。 |
| 层类型混合 | "5:1 SWA 到全局" | 在堆栈中交错稀疏和全注意力层，以更低内存保持质量。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 滑动窗口 + 全局 token 的经典论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的局部+步进模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:全局混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 5:1 混合，窗口=1024，现为教科书级默认配置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的学习稀疏注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — "Use It" 中 mask-as-callable 模式的 API 参考。
