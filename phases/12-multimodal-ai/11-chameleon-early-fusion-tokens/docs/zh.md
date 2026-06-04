# Chameleon 与仅基于 Token 的早期融合多模态模型

> 到目前为止我们见过的所有 VLM 都将图像和文本分开处理。视觉 Token 来自视觉编码器，流经投影层（projector），然后在 LLM 内部与文本汇合。视觉和文本的词表从未重叠。Chameleon（Meta，2024年5月）提出了一个问题：如果它们重叠呢？训练一个 VQ-VAE，将图像转换为共享词表中的离散 Token 序列。现在，每个多模态文档都变成了一个序列——文本 Token 和图像 Token 交错排列，使用单一的自回归损失。副作用是：该模型可以生成混合模态的输出——在单次推理调用中交替输出文本和图像 Token。本课将研读这篇早期融合的论文，并从头构建一个玩具版本。

**类型：** 构建
**语言：** Python（标准库、VQ-VAE 分词器 + 交错解码器）
**前置要求：** 第 12 阶段 · 05，第 8 阶段（生成式 AI）
**耗时：** 约 180 分钟

## 学习目标

- 解释为什么共享词表 + 单一损失会改变模型的能力边界。
- 描述 VQ-VAE 如何将图像分词为与 Transformer 下一个 Token 预测目标兼容的离散序列。
- 列举 Chameleon 的训练稳定性技巧：QK-Norm、Dropout 放置位置、LayerNorm 顺序。
- 对比 Chameleon 与 BLIP-2 的 Q-Former 方法，并说明各自适用的场景。

## 问题所在

基于适配器的 VLM（如 LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的事物。文本 Token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型拥有两条输入路径，它们在中间某处汇合。

带来三个后果：

1. LLM 只能消费（接收）图像，无法生成图像。输出仅限于文本。
2. 混合模态文档（如文章中段落与图像交替出现）的处理很别扭——你要么在模型外部解析多模态输入，要么进行链式生成。
3. 分布不匹配。视觉 Token 和文本 Token 位于隐藏空间的不同区域，导致细微的对齐问题。

Chameleon 摒弃了这一前提：图像仅仅是共享词表中离散 Token 的序列。在交错文档上训练模型，使用单一损失和单一自回归解码器，即可免费解锁混合模态生成能力。

## 核心概念

### 作为图像分词器的 VQ-VAE

该分词器是一个向量量化变分自编码器（VQ-VAE）。架构如下：

- 编码器：CNN + ViT，将图像映射为空间特征图，例如 32x32、维度为 256 的特征。
- 码本（Codebook）：学习得到的 K 个向量词表（Chameleon 使用 8192），维度同样为 256。
- 量化：对每个空间特征，通过 L2 距离查找最近的码本条目。用整数索引替换连续特征值。
- 解码器：CNN，将量化后的特征还原为像素。

训练过程：VAE 重建损失 + 承诺损失（commitment loss）+ 码本损失。码本索引构成了图像的离散字母表。

对于 Chameleon：一张图像被转换为 32*32 = 1024 个 Token，这些 Token 取自大小为 8192 的词表。将其与文本 Token（来自 LLM 的 BPE 词表，假设为 32000）拼接。最终词表大小：40192。Transformer 看到的只是一个序列、一种损失。

### 共享词表

Chameleon 的词表结合了文本 Token、图像 Token 和模态分隔符。每个 Token 只有一个 ID。输入嵌入层将每个 ID 映射为 D 维隐藏向量。输出投影层将隐藏向量映射回词表 logits。Softmax 选择下一个 Token，无论其属于哪种模态。

分隔符至关重要：`<image>` 和 `</image>` 标签用于包裹图像 Token 序列。在生成时，如果模型输出了 `<image>`，下游软件就知道接下来的 1024 个 Token 是 VQ 索引，需要发送给解码器进行像素渲染。

### 混合模态生成

推理过程是在共享词表中进行下一个 Token 预测。示例提示词：“画一只猫并描述它。” Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主决定顺序——它可以先输出图像再输出文本，先文本后图像，或者交错输出。使用相同的解码器和相同的损失函数。

与仅能生成文本的适配器 VLM 相比，Chameleon 重新开启了关于模型输出模态的讨论。

### 训练稳定性 —— QK-Norm、Dropout 与 LayerNorm 顺序

早期融合训练在大规模下不稳定。Chameleon 的论文记录了三种技巧：

- QK-Norm。在注意力机制内部、点积计算之前，对查询（query）和键（key）的投影应用 LayerNorm。防止深层网络中 logit 幅值爆炸。已被多个 2024 年后的大模型采用。
- Dropout 放置位置。在每个残差相加（residual-add）之后应用 Dropout，而不仅限于注意力层和 MLP 之后。当图像 Token 的梯度可能占主导时，需要更强的正则化。
- LayerNorm 顺序。残差分支采用 Pre-LN（标准做法），并在最后一个 Block 的跳跃连接（skip connection）上额外添加一个 LN。稳定最后一层的梯度流动。

没有这些技巧，340 亿参数的 Chameleon 训练会在多个检查点发散。有了它们，模型才能收敛。训练配方（recipe）本身与架构设计同等重要。

### 分词器的重建上限

VQ-VAE 是有损的。在码本条目为 8192、每张 512x512 图像对应 1024 个 Token 的情况下，重建 PSNR 上限约为 26-28 dB。这足以实现可识别的图像生成，但明显不如连续空间的扩散模型（Stable Diffusion 3 可达 32+ dB）。

分词器是瓶颈。更好的分词器（如 MAGVIT-v2、IBQ、SBER-MoVQGAN）能够提升这一上限。Emu3（见第 12.12 课）仅凭更优秀的分词器就实现了达到 SDXL 质量的生成效果。

### Chameleon 与 BLIP-2 / LLaVA 对比

Chameleon（早期融合，共享词表）：
- 单一损失，单一解码器。
- 生成混合模态输出。
- 分词器决定了质量上限。
- 成本高：推理路径上每生成一张图像都需要运行一次 VQ-VAE 解码器。

BLIP-2 / LLaVA（晚期融合，独立塔结构）：
- 仅支持视觉输入、文本输出。
- 复用预训练的 LLM。
- 理解任务不受分词器瓶颈限制。
- 成本低：仅需单次前向传播。

根据任务选择。如果需要图像生成，选 Chameleon 系列。如果只需要理解能力，适配器 VLM 更简单且能复用更多预训练算力。

### Fuyu 与 AnyGPT

Fuyu（Adept，2023）是一种相关思路：完全跳过独立的视觉编码器，将原始图像块直接通过 LLM 的输入投影层，就像它们是 Token 一样，无需分词器。比 Chameleon 更简单，但失去了共享词表的输出生成能力。

AnyGPT（Zhan 等人，2024）将 Chameleon 扩展至四种模态：文本、图像、语音、音乐。每种模态使用相同的 VQ-VAE 技巧，共享同一个 Transformer。实现任意到任意的生成。更多内容见第 12.16 课。

## 动手实践

`code/main.py` 构建了一个端到端的玩具级早期融合模型：

- 一个微型 VQ-VAE 风格量化器，将 8x8 图像块映射为码本索引（K=16）。
- 一个共享词表，包含（文本 id 0..31）+（图像 id 32..47）+（分隔符 48, 49）。
- 一个玩具级自回归解码器（二元语法表），在合成描述文本 + 图像 Token 序列上训练。
- 采样循环，根据提示词交替输出文本和图像 Token。

代码故意将 Transformer 保持极小（仅使用二元语法），以便你能从头到尾追踪信号流。

## 交付成果

本课将产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品需求规格（仅理解 vs 理解+生成、要求的图像质量、成本预算），它将在 Chameleon 系列（早期融合）和 LLaVA 系列（晚期融合）之间做出选择，并用定量经验法则加以论证。

## 练习

1. Chameleon 使用 K=8192 的码本条目，每张 512x512 图像对应 1024 个 Token。估算其与 24 位 RGB 图像相比的压缩率。它是有损的吗？损耗程度如何？

2. 相同 VQ-VAE 密度下，一张 4K 图像（3840x2160）会产生多少个图像 Token？Chameleon 风格的模型能否在一次推理调用中生成 4K 图像？最先崩溃的是什么——上下文窗口、分词器质量还是 KV Cache？

3. 用纯 Python 实现 QK-Norm。给定 64 维的 query 和 key，展示 LayerNorm 前后的点积结果。为什么在深层网络中控制幅值很重要？

4. 阅读 Chameleon 论文第 2.3 节关于训练稳定性的内容。描述论文在 34B 参数规模下未使用 QK-Norm 时观察到的具体失效模式。“范数爆炸”的表现特征是什么？

5. 扩展现有的玩具解码器，使其在仅给定文本提示词时能输出混合模态响应。在训练数据分布为 60% 文本优先 / 40% 图像优先的条件下，统计模型选择图像优先与文本优先的频率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Early fusion（早期融合） | “统一 Token” | 从第一步起就将图像转换为与 Transformer 词表共享的离散 Token |
| VQ-VAE | “图像分词器” | 由 CNN + ViT + 码本组成的模块，将图像映射为 Transformer 可预测的整数索引 |
| Shared vocabulary | “单一字典” | 覆盖文本 + 图像 + 模态分隔符的统一 Token ID 空间 |
| QK-Norm | “注意力稳定器” | 在 query 和 key 点积前对其应用 LayerNorm，防止范数爆炸 |
| Mixed-modality generation | “文本+图像输出” | 推理过程中自主地在单次传递中生成交错的文本和图像 Token |
| Codebook size | “K 个条目” | VQ-VAE 可量化的离散向量数量；在压缩率与保真度之间权衡 |
| Tokenizer ceiling | “重建上限” | 解码 VQ Token 所能达到的最佳 PSNR；限制了模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
