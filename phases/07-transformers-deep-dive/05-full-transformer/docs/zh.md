# 完整 Transformer — 编码器 + 解码器

> Attention 是主角。其余部分 —— 残差连接、归一化、前馈网络、交叉注意力 —— 都是支撑你把它堆深的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**时间：** ~75 分钟

## 问题所在

单层注意力是一个特征提取器，不是模型。每层只做一次矩阵乘法，对语言来说容量不够。你需要深度 —— 但没有正确的管道，深度会崩溃。

2017 年 Vaswani 的论文打包了六个设计决策，将单层注意力变成了可堆叠的模块。此后的每个 Transformer —— 仅编码器 (BERT)、仅解码器 (GPT)、编码器-解码器 (T5) —— 都继承了同一个骨架。到 2026 年，模块已被精炼 (RMSNorm、SwiGLU、pre-norm、RoPE)，但骨架完全相同。

这节课讲骨架。后续课程专门化 —— 06 讲编码器，07 讲解码器，08 讲编码器-解码器。

## 核心概念

![编码器和解码器模块内部结构，连线示意](../assets/full-transformer.svg)

### 六个组成部分

1. **Embedding + 位置信号。** 词元 → 向量。通过 RoPE (现代) 或正弦函数 (经典) 注入位置信息。
2. **自注意力。** 每个位置关注所有其他位置。解码器中用掩码。
3. **前馈网络 (FFN)。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比率为 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在约 6 层后就会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm` (现代)。稳定残差流。
6. **交叉注意力 (仅解码器)。** Query 来自解码器，Key 和 Value 来自编码器输出。

### 编码器模块 (BERT、T5 编码器使用)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有掩码。所有位置都能看到所有位置。

### 解码器模块 (GPT、T5 解码器使用)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个模块有三个子层。中间那个 —— 交叉注意力 —— 是信息从编码器流向解码器的唯一通道。在纯解码器架构 (GPT) 中，交叉注意力被省略，只有掩码自注意力 + FFN。

### Pre-norm vs post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠 —— 没有小心的 warmup，深层训练很困难。Pre-norm (`LN` 在子层*之前*) 是 2026 年的默认选择：Llama、Qwen、GPT-3+、Mistral 都在用。

### 2026 年的现代化模块

Vaswani 2017 年发布的是 LayerNorm + ReLU。现代堆栈把两者都换了。生产环境的模块实际长这样：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展比率 | 4× | 2.6× (SwiGLU 用三个矩阵，总参数量匹配) |
| 位置编码 | 正弦绝对位置 | RoPE |
| 注意力 | 完整 MHA | GQA (或 MLA) |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化 (少一次减法)，节省计算且经验上至少同样稳定。SwiGLU (`Swish(W1 x) ⊙ W3 x`) 在 Llama、PaLM 和 Qwen 的论文中，持续比 ReLU/GELU FFN 低约 0.5 的 ppl。

### 参数量

对于一个模块，`d_model = d`，FFN 扩展 `r`：

- MHA：`4 · d²` (Q、K、V、O 投影)
- FFN (SwiGLU)：`3 · d · (r · d)` ≈ `3rd²`
- 归一化：可忽略

在 `d = 4096, r = 2.6, layers = 32` (大致 Llama 3 8B)，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (加上 embedding 和输出头)。与公开数字一致。

## 动手实现

### 步骤 1：基础组件

使用第 03 课中的小型 `Matrix` 类 (复制到本文件以保证独立)：

- `layer_norm(x, eps=1e-5)` —— 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` —— 除以 RMS。不减均值。
- `gelu(x)` 和 `silu(x) * W3 x` (SwiGLU)。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整连线见 `code/main.py`。

### 步骤 2：搭建 2 层编码器和 2 层解码器

堆叠它们。将编码器输出传入每个解码器交叉注意力。在输出投影前加最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤 3：在 toy 示例上运行前向

输入 6 个词元的源序列和 5 个词元的目标序列。验证输出形状为 `(5, vocab)`。不训练 —— 这节课讲架构，不是损失函数。

### 步骤 4：换入 RMSNorm + SwiGLU

把 LayerNorm 和 ReLU-FFN 替换为 RMSNorm 和 SwiGLU。确认形状仍然匹配。这就是 2026 年的现代化，只需替换一个函数。

## 使用它

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年的生产代码自己实现模块，因为：

- Flash Attention 在注意力内部调用，不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考实现中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认选项。

HF `transformers` 有干净的参考模块值得阅读：`modeling_llama.py` 是标准的 2026 年仅解码器模块。约 500 行，值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器 —— 如何选择：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、embedding、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、对话、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出 (翻译、摘要) | 编码器-解码器 | T5、BART、Whisper |

仅解码器在语言任务中胜出，因为它扩展最干净，同时处理理解和生成。编码器-解码器在输入有明确"源序列"身份时仍然最佳 (翻译、语音识别、结构化任务)。

## 交付

见 `outputs/skill-transformer-block-reviewer.md`。该技能要求对照 2026 年默认标准审查新的 Transformer 模块实现，并标记缺失部分 (pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比率)。

## 练习

1. **简单。** 统计你的 encoder_block 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 的参数量。通过实现该模块并使用 `sum(p.numel() for p in block.parameters())` 来验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化两者，在随机输入上测量 12 层堆叠后的激活范数。Post-norm 的激活应该爆炸；pre-norm 的应该保持有界。
3. **困难。** 在 toy copy 任务上实现 4 层编码器-解码器 (复制 `x` 的反转)。训练 100 步。报告损失。换入 RMSNorm + SwiGLU + RoPE —— 损失会下降吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Block | "一层 Transformer" | norm + attention + norm + FFN 的堆叠，包裹在残差连接中。 |
| Residual | "跳跃连接" | `x + f(x)` 输出；实现深层堆栈中的梯度流动。 |
| Pre-norm | "先归一化，不是后归一化" | 现代：`x + sublayer(LN(x))`。无需 warmup 技巧即可训练更深。 |
| RMSNorm | "去掉均值的 LayerNorm" | 除以 RMS；少一个操作，经验稳定性相同。 |
| SwiGLU | "大家都换上的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上击败 ReLU/GELU。 |
| Cross-attention | "解码器如何看到编码器" | Q 来自解码器，K/V 来自编码器输出的 MHA。 |
| FFN expansion | "中间 MLP 有多宽" | hidden-size 与 d_model 的比率，通常为 4 (LayerNorm) 或 2.6 (SwiGLU)。 |
| Bias-free | "去掉 +b 项" | 现代堆栈在线性层中省略偏置；ppl 略有提升，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 原始模块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) —— 为什么 pre-norm 在深层击败 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) —— RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) —— SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 标准的 2026 年仅解码器模块。
