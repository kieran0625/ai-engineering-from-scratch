# 视觉自回归建模（VAR）：下一尺度预测

> 扩散模型在时间维度上迭代采样（去噪步骤）。VAR 在尺度维度上迭代采样——它先预测 1×1 的 token，然后是 2×2，接着是 4×4，直至最终分辨率，每一尺度都以之前所有尺度为条件。2024 年的论文表明，VAR 在图像生成上遵循类 GPT 的缩放定律，并在相同计算预算下超越 DiT。本课构建其核心机制。

**类型：** 构建
**语言：** Python（使用 PyTorch）
**前置知识：** Phase 7 Lesson 03（多头注意力），Phase 8 Lesson 06（DDPM）
**时间：** ~90 分钟

## 问题背景

自回归生成在语言建模中占据主导地位，因为它具有可预测的扩展性：更多计算、更多参数、更低困惑度、更好输出。2024 年之前，图像生成有两次主要的自回归尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 编码上逐 token）。

两者都受制于生成顺序问题。像素和 token 按二维网格排列，但自回归模型必须按一维光栅顺序访问它们。早期角落的像素对图像最终形态一无所知。生成质量的扩展性不如类 GPT 文本模型，且在匹配计算量下始终未能达到扩散模型的质量。

VAR 通过改变生成对象来解决生成顺序问题。不再是在空间中逐个预测图像 token，VAR 预测的是分辨率递增的完整图像。步骤 1：预测 1×1 token（整体图像"摘要"）。步骤 2：预测 2×2 token 网格（较粗特征）。步骤 3：预测 4×4 网格。步骤 K：预测最终的 (H/8)×(W/8) 网格。

每一尺度都关注所有先前尺度（在"尺度顺序"上因果），而在自身尺度内并行计算。顺序问题消失了：尺度 k 的完整图像在一次 transformer 前向传播中生成。

## 核心概念

### VQ-VAE 多尺度分词器

VAR 需要一个**多尺度离散分词器**。对于图像 x，它生成一系列分辨率逐渐升高的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的码本（典型大小 4096-16384）。各尺度的分词并非独立——它被训练为通过累加各尺度的残差来重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一种**残差 VQ** 变体。尺度 k 捕捉了尺度 1..k-1 遗漏的信息。解码器取所有尺度嵌入之和并生成图像。

多尺度 VQ 分词器训练一次（类似 VQGAN）后冻结。所有生成工作都由其上方的自回归模型完成。

### 下一尺度预测

生成模型是一个 transformer，它接收所有先前尺度的 token 并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入同时编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k、位置 (i, j) 的 token 可以关注尺度 1..k 的所有 token，以及尺度 k 中按某种尺度内顺序更早出现的 token（VAR 使用固定位置注意力，无尺度内因果性——同一尺度内的所有位置并行预测）。

训练损失：在每个尺度 k，给定所有先前尺度的 token 预测 z_k。对离散 VQ 编码使用交叉熵损失。结构与 GPT 相同，只是"序列"现在是尺度结构化的。

### 生成过程

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K=10 个尺度，生成需要 10 次 transformer 前向传播。每次传播并行生成整个尺度——尺度内无需逐 token 自回归。对于 256×256 图像，这大约是 10 次传播，而 DiT 需要 28-50 次。

### 为何下一尺度优于下一 Token

三项结构性优势：
1. **由粗到细符合自然图像统计。** 人类视觉感知和图像数据集都表现出尺度相关的规律性：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 与类 GPT token 自回归不同，VAR 一步生成尺度内的所有 token。有效生成长度是对数级而非线性级。
3. **无生成顺序偏差。** 尺度 k 的 token 能看到完整的尺度 k-1；不存在"左侧"或"上方"偏差，迫使早期 token 在晚期上下文可用前就做出承诺。

### 缩放定律

Tian 等人证明，VAR 在 ImageNet 上的 FID 遵循幂律缩放曲线——正如 GPT 对困惑度所做的那样。参数或计算量翻倍可靠地使误差减半。这是首个展现出如此清晰的语言模型式缩放行为的图像生成模型。结果是 VAR 尺度的预测可以从计算量推导，而非针对每个架构的经验猜测。

### 与扩散模型的关系

VAR 和扩散共享相同的数据压缩故事：都将生成问题分解为一系列更简单的子问题。

- 扩散：逐渐添加噪声，学习撤销一步。
- VAR：逐渐增加分辨率，学习预测下一尺度。

它们是问题的不同轴向。两者都产生可处理的条件分布。经验上 VAR 推理更快（更少传播次数，尺度内全并行），且在类别条件 ImageNet 上匹配或超越 DiT。文本条件 VAR（VARclip、HART）是活跃的研究方向。

## 动手构建

在 `code/main.py` 中，你将：
1. 在合成"图像"数据（二维高斯环）上构建一个微型**多尺度 VQ 分词器**。
2. 训练一个**类 VAR transformer** 进行下一尺度预测。
3. 通过调用 transformer 4 次（4 个尺度）进行采样并解码。
4. 验证尺度有序训练确实实现了尺度内并行生成。

这是一个玩具实现。重点是让尺度结构化注意力掩码和尺度内并行生成真正运行起来。

## 交付成果

本课产出 `outputs/skill-var-tokenizer-designer.md`——一项设计多尺度分词器的技能：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习题

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量与自回归传播次数的关系。更多尺度 = 更精细残差 = 更好质量但更多传播。

2. **码本大小。** 用码本大小 512、4096、16384 训练分词器。更大码本带来更好重建但更难预测。寻找拐点。

3. **尺度内并行性检验。** 对训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置而非尺度内位置？验证掩码实现。

4. **VAR 与 DiT 缩放对比。** 在相同 ImageNet 类别条件任务上，以匹配参数预算（如 33M、130M、458M）训练 VAR 和 DiT。绘制 FID 与计算量的关系。VAR 应在每个尺寸上超越 DiT——在小规模上复现论文结果。

5. **文本条件。** 扩展 VAR，通过 adaLN 将文本嵌入（CLIP 池化）作为额外条件输入。这是 HART 的做法。在文本对齐采样上 FID 提升多少？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| VAR | "Visual AutoRegressive" | 通过在 VQ token 网格金字塔上进行下一尺度预测来生成图像 |
| 下一尺度预测 | "先粗后细" | 模型预测分辨率递增的尺度 token，以所有先前尺度为条件 |
| 多尺度 VQ 分词器 | "残差 VQ" | 生成 K 个分辨率递增 token 网格的 VQ-VAE，解码器对所有尺度求和 |
| 尺度 k | "金字塔层级 k" | K 个分辨率层级之一，从 k=1 的 1×1 到 k=K 的 (H/p)×(W/p) |
| 尺度内并行 | "每尺度一次前向" | 尺度 k 的所有 token 在一次 transformer 传播中预测，非自回归 |
| 跨尺度因果 | "尺度有序注意力" | 尺度 k 的 token 可关注尺度 1..k 的全部，不可关注尺度 k+1..K |
| 残差 VQ | "加性分词" | 每个尺度的 token 编码低尺度遗留的残差；解码器累加所有尺度嵌入 |
| VAR 缩放定律 | "图像 GPT 缩放" | FID 在计算量上遵循可预测的幂律，如语言模型的困惑度 |
| HART | "混合 VAR + 文本" | 文本条件 VAR 变体，结合 MaskGIT 式迭代解码与 VAR 的尺度结构 |
| 尺度位置嵌入 | "(尺度, 行, 列) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，扩散对比基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 多尺度分词器所扩展的分词器家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像分词的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — 文本条件 VAR
