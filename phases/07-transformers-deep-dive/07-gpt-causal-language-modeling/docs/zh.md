# GPT — 因果语言建模

> BERT 能看到两边。GPT 只能看到过去。三角掩码是现代 AI 中最具决定性的一行代码。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 02（自注意力）、Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）
**时间：** ~75 分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个 token，token `t` 的概率分布是什么？在这个信号——下一 token 预测——上进行训练，你就能得到一个可以逐 token 生成任意文本的模型。

要在整个序列上端到端并行训练，你需要让每个位置的预测仅依赖于更早的位置。否则模型会通过偷看答案来轻易作弊。

因果掩码实现了这一点。它是一个上三角矩阵，由 `-inf` 值组成，在 softmax 之前加到注意力分数上。经过 softmax 后，这些位置变为 0。每个位置只能关注到自身和更早的位置。而且由于你一次性将其应用于整个序列，一次前向传播就能得到 N 个并行的下一 token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是仅解码器的因果 Transformer，核心循环相同。只是更大、数据更好、RLHF 更好。

## 概念

![因果掩码创建三角注意力矩阵](../assets/causal-attention.svg)

### 掩码

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，因此被掩码的位置贡献的权重为零。注意力矩阵的每一行都是仅关于前面位置的概率分布。

实现成本：一次 `torch.tril()` 调用。计算时间：纳秒级。对领域的影响：一切。

### 并行训练，串行推理

训练：一次性对整个 `(N, d_model)` 序列进行前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列方向并行。这就是 GPT 训练能够扩展的原因——一次 GPU 传递就能在一个 batch 中处理 1M 个 token。

推理：逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV 缓存（第 12 课）保存 `t1…tn` 的隐藏状态，避免每一步重新计算。但推理时的串行深度 = 输出长度。这就是自回归代价，也是解码成为每个 LLM 延迟瓶颈的原因。

### 损失——移位一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵。

你听说过的每个 Transformer 语言模型都用这个损失训练。预训练、微调、SFT——相同的损失，不同的数据。

### 解码策略

训练完成后，采样选择比人们想象的更重要。

| 方法 | 作用 | 适用场景 |
|------|------|---------|
| 贪心（Greedy） | 每步取 argmax | 确定性任务、代码补全 |
| Temperature | 将 logits 除以 T 后采样 | 创意任务，T 越高 = 多样性越高 |
| Top-k | 仅从 top-k 个 token 中采样 | 消除低概率尾部 |
| Top-p（nucleus） | 从累积概率 ≥ p 的最小集合中采样 | 2020 年后的默认；适应分布形状 |
| Min-p | 保留满足 `p > min_p * max_p` 的 token | 2024 年后；比 top-p 更擅长拒绝长尾 |
| 推测解码（Speculative decoding） | 草稿模型提议 N 个 token，大模型验证 | 相同质量下延迟降低 2–3 倍 |

2026 年，min-p + temperature 0.7 是开放权重模型的合理默认。推测解码是任何生产推理栈的标配。

### "GPT 配方"成功的关键

1. **仅解码器。** 没有编码器开销。每层一次注意力 + FFN 传递。
2. **规模扩展。** 124M → 1.5B → 175B → 万亿级。Chinchilla 缩放定律（第 13 课）告诉你如何分配计算资源。
3. **上下文学习。** 在 6B–13B 参数左右涌现。模型无需微调即可遵循少样本示例。
4. **RLHF。** 在人类偏好上进行后训练，将原始预训练文本转化为对话助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模稳定训练。

核心架构自 GPT-2 以来变化不大。所有有趣的进展都发生在数据、规模和后训练上。

## 构建

### 步骤 1：因果掩码

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前将其加到注意力分数上。这就是全部机制。

### 步骤 2：一个 2 层类 GPT 模型

堆叠两个解码器块（掩码自注意力 + FFN，无交叉注意力）。添加 token 嵌入、位置编码和反嵌入（与 token 嵌入矩阵绑定——自 GPT-2 以来的标准技巧）。

### 步骤 3：端到端的下一 token 预测

在 20 个 token 的玩具词表上，在每个位置生成 logits。与移位一位的目标计算交叉熵损失。无梯度——这是前向传播的正确性检查。

### 步骤 4：采样

实现贪心、temperature、top-k、top-p、min-p。在固定提示上运行每种方法并比较输出。采样函数只需 10 行。

## 使用

PyTorch，2026 年风格：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 执行前向传播，提取最终位置的 logits，采样下一 token，追加，重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都用大量优化实现相同的循环——批量预填充、连续批处理、KV 缓存分页、推测解码。

**GPT vs BERT，各一行：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定了模型是否能生成。

## 交付

见 `outputs/skill-sampling-tuner.md`。该技能为新的生成任务选择采样参数，并在需要确定性解码时标记。

## 练习

1. **简单。** 运行 `code/main.py` 并验证因果注意力矩阵在 softmax 后是下三角的。抽查：第 3 行应该只在第 0–3 列有权重。
2. **中等。** 实现宽度为 4 的束搜索（beam search）。在 10 个短提示上比较 beam-4 与贪心的困惑度。beam 总是赢吗？（提示：通常翻译任务会，开放式对话不会。）
3. **困难。** 实现推测解码：用一个 2 层小模型作为草稿，6 层模型作为验证器。在 100 个长度为 64 的补全上测量 wall-clock 加速。确认输出与验证器的贪心结果一致。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 因果掩码（Causal mask） | "那个三角形" | 上三角 `-inf` 矩阵，加到注意力分数上，使得位置 `i` 只能看到位置 `≤ i`。 |
| 下一 token 预测（Next-token prediction） | "那个损失" | 模型分布与真实下一 token 在每个位置上的交叉熵。 |
| 自回归（Autoregressive） | "一个一个生成" | 输出反馈为输入；仅在训练时并行，生成时不并行。 |
| Logits | "softmax 前的分数" | LM 头在 softmax 前的原始输出；采样在这些值上进行。 |
| Temperature | "创意旋钮" | 将 logits 除以 T；T→0 = 贪心，T→∞ = 均匀分布。 |
| Top-p | "Nucleus 采样" | 将分布截断到累积和 ≥ p 的最小集合；从剩余部分采样。 |
| Min-p | "比 top-p 更好" | 保留满足 `p ≥ min_p × max_p` 的 token；根据分布尖锐程度自适应调整截断点。 |
| 推测解码（Speculative decoding） | "草稿 + 验证" | 便宜模型提议 N 个 token；大模型并行验证。 |
| 强制教学（Teacher forcing） | "训练技巧" | 训练时输入真实的上一 token，而非模型的预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 与上下文学习。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 因果语言模型的标准参考代码。
