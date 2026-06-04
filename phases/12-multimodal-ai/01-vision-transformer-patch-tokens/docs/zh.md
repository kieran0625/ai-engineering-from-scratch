# 视觉Transformer与Patch-Token基础单元

> 在涉及多模态之前，图像必须先转化为Transformer能够处理的token序列。2020年的ViT论文通过16x16像素的patch、线性投影和位置嵌入回答了这一问题。五年后的今天，所有2026年的前沿模型（原生支持2576px的Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）依然以此为基础——编码器从ViT演进到DINOv2再到SigLIP 2，加入了register token，位置编码方案变为2D-RoPE，但这一基础单元始终未变。本课将端到端地阅读patch-token流水线，并用标准库Python实现它，以便为Phase 12的其余内容建立关于“视觉tokens”的具体心智模型。

**类型：** 学习
**语言：** Python（标准库，含patch tokenizer与几何计算器）
**前置知识：** Phase 7（Transformer）、Phase 4（计算机视觉）
**耗时：** 约120分钟

## 学习目标

- 将HxWx3维度的图像转换为带有正确位置编码的patch token序列。
- 计算给定（patch大小、分辨率、隐藏层维度、深度）的ViT的序列长度、参数量和FLOPs。
- 列举将ViT从2020年研究推向2026年生产环境的三大升级：自监督预训练（DINO / MAE）、register tokens以及原生分辨率打包。
- 根据下游任务在CLS池化、均值池化和register tokens之间做出选择。

## 问题所在

Transformer处理的是向量序列。文本本身已经是序列（字节或token）。图像是一个具有三个颜色通道的二维像素网格——并非序列。如果展平每个像素，一张224x224的RGB图像会变成150,528个token，而在此长度下进行自注意力计算是不可行的（复杂度随序列长度呈二次方增长）。

2020年之前的方法是在前端拼接一个CNN特征提取器：ResNet生成一个7x7的特征图，包含2048维向量的49个token，然后将这些token输入Transformer。这种方法有效，但继承了CNN的偏差（平移等变性、局部感受野），并失去了Transformer对规模扩展的渴望。

Dosovitskiy等人（2020）提出了一个直白的问题：如果我们跳过CNN会怎样？将图像分割成固定大小的patch（例如16x16像素），将每个patch线性投影为向量，添加位置嵌入，然后将其作为序列输入标准的Transformer。在当时这被视为异端——没有卷积的视觉。但在足够多的数据（JFT-300M，随后是LAION）支持下，它在ImageNet上击败了ResNet并持续改进。

到了2026年，ViT的基础单元已成为无可争议的核心。每个开源权重VLM的视觉塔都是其衍生版本（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是“我们是否应该使用patch？”，而是“采用多大的patch尺寸、什么样的分辨率调度、什么预训练目标、什么位置编码”。

## 核心概念

### 将Patch作为Token

给定一张形状为`(H, W, 3)`的图像`x`和一个patch大小`P`，你将图像切割成`(H/P) x (W/P)`个不重叠的patch网格。每个patch是一个`P x P x 3`的像素立方体。将每个立方体展平为一个`3 P^2`的向量。应用一个形状为`(3 P^2, D)`的共享线性投影`W_E`，将每个patch映射到模型的隐藏维度`D`。

对于ViT-B/16的标准配置：
- 分辨率224，patch大小16 → 网格14x14 → 196个patch token。
- 每个patch包含`16 x 16 x 3 = 768`个像素值，被投影到`D = 768`。
- 添加一个可学习的`[CLS]` token → 序列长度为197。

该patch投影在数学上等同于一个核大小为`P`、步幅为`P`、输出通道数为`D`的2D卷积。这正是生产代码实际实现它的方式——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。“线性投影”是一种概念性表述；而“卷积核”表述则更高效。

### 位置嵌入

Patch本身没有内在顺序——Transformer将它们视为一个无序集合。早期的ViT添加了可学习的1D位置嵌入（每个位置一个768维向量，共197个）。这可行，但将模型绑定到了训练分辨率：在推理时，如果你改变网格，就必须插值位置表。

现代视觉骨干网络使用2D-RoPE（Qwen2-VL的M-RoPE、SigLIP 2的默认选项）或分解的2D位置。2D-RoPE根据patch的（行，列）索引旋转查询和键向量，因此模型可以从旋转角度推断相对2D位置。无需位置表。模型在推理时可以处理任意网格尺寸。

### CLS Token、池化输出与Register Tokens

什么是图像级别的表示？三种选择并存：

1. `[CLS]` token。在patch序列前添加一个可学习向量。经过所有Transformer块后，CLS token的隐藏状态即为图像表示。继承自BERT。用于原始ViT和CLIP。
2. 均值池化。对patch token的输出隐藏状态求平均。用于SigLIP、DINOv2及大多数现代VLM。
3. Register tokens。Darcet等人（2023）观察到，在没有显式sink token的情况下训练的ViT会产生高范数的“伪影”patch，从而劫持自注意力机制。添加4到16个可学习的register token可以吸收这种负载，并提升密集预测的质量（分割、深度估计）。DINOv2和SigLIP 2均内置了registers。

这一选择对下游任务至关重要。CLS适用于分类任务。对于将patch token输入LLM的VLM，你完全跳过池化——每个patch都成为LLM的输入token。Registers在交接前会被丢弃（它们是脚手架，而非内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020年的ViT使用JFT-300M上的监督分类进行预训练。很快被以下方法取代：

- CLIP (2021)：基于4亿对图像的对比学习。见课程12.02。
- MAE (2021, He等人)：掩蔽75%的patch，重建像素。自监督，适用于纯图像。
- DINO (2021) / DINOv2 (2023)：基于师生架构的自蒸馏，无标签，无描述。2023年的DINOv2 ViT-g/14是最强的纯视觉骨干网络，也是“密集特征”用例的默认选择。
- SigLIP / SigLIP 2 (2023, 2025)：带有sigmoid损失和NaFlex以支持原生纵横比的CLIP。是2026年开源VLM（Qwen、Idefics2、LLaVA-OneVision）中的主导视觉塔。

你选择的预训练方式决定了骨干网络的适用场景：CLIP/SigLIP擅长与文本进行语义匹配，DINOv2擅长提供密集视觉特征，MAE适合作为下游微调的起点。

### 缩放定律

ViT缩放定律（Zhai等人，2022）确立了ViT的质量遵循模型大小、数据大小和计算量之间的可预测规律。在固定计算量下：
- 更大的模型 + 更多的数据 → 更好的质量。
- Patch大小是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m的典型配置）比patch 16产生更多的图像token；更适合OCR和密集任务，但速度较慢。
- 分辨率是另一个关键杠杆。从224提升到384再到512几乎总是有益的，但FLOPs成本呈二次方增长。

ViT-g/14（10亿参数，patch 14，分辨率224 → 256个token）和SigLIP SO400m/14（4亿参数，patch 14）是2026年开源VLM的两大主力编码器。

### ViT的参数量计算

完整计算位于`code/main.py`中。对于224分辨率下的ViT-B/16：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载checkpoint之前，用这种方式估算每个ViT的参数量。骨干网络的大小决定了你在任何下游VLM中的VRAM底线。

### 2026年生产环境配置

2026年大多数开源VLM搭载的编码器是原生分辨率（NaFlex）下的SigLIP 2 SO400m/14。它具有：
- 4亿参数。
- Patch大小14，默认分辨率384 → 每张图像729个patch token。
- 图像级任务使用均值池化；所有729个patch直接流入LLM用于VQA。
- 4个register token，在交接给LLM前被丢弃。
- 带有图像级缩放的2D-RoPE以支持原生纵横比。

该配置中的每一个决策都能追溯到你可以阅读的论文。

## 实践应用

`code/main.py`是一个patch tokenizer和几何计算器。它接收（图像高H、宽W、patch大小P、隐藏维度D、深度L）并报告：

- 分块后的网格形状和序列长度。
- 合成8x8像素玩具图像的token序列（逐步演示展平+投影流程）。
- 按patch嵌入、位置嵌入、Transformer块和头拆分的参数量。
- 目标分辨率下单次前向传播的FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384之间的对比表格。

运行它。将参数量与公开数据核对。尝试调整patch大小和分辨率，直观感受token数量的成本变化。

## 交付成果

本课将产出`outputs/skill-patch-geometry-reader.md`。给定一个ViT配置（patch大小、分辨率、隐藏维度、深度），它将生成带有理由说明的token数量、参数量和VRAM估算。每当为VLM选择视觉骨干网络时，都应使用这项技能——它能避免“token爆炸导致LLM上下文溢出”的意外情况。

## 练习

1. 计算Qwen2.5-VL在原生1280x720输入且patch大小为14时的patch-token序列长度。这与仅使用CLS的表示相比如何？

2. 一个1080p帧（1920x1080）在patch 14下会产生多少token？在30 FPS下播放5分钟的视频，总共需要多少视觉token？哪种策略最节省开销：池化、帧采样还是token合并？

3. 使用纯Python实现对patch token的均值池化。验证对DINOv2输出的196个token进行均值池化的结果，是否与调用模型`forward`请求池化嵌入时返回的结果一致。

4. 阅读《Vision Transformers Need Registers》（arXiv:2309.16588）的第3节。用两句话描述registers吸收了何种伪影，以及这对下游密集预测为何重要。

5. 修改`code/main.py`以支持patch-n'-pack：给定一组不同分辨率的图像列表，生成单个打包序列和块对角注意力掩码。当你学到Lesson 12.06时，请对其进行验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Patch | “16x16像素方块” | 输入图像中固定大小且不重叠的区域；对应一个token |
| Patch Embedding | “线性投影” | 一个共享的可学习矩阵（或步长为P的Conv2d），将展平的patch像素映射为D维向量 |
| CLS Token | “类别token” | 前置的可学习向量，其最终隐藏状态代表整张图像；在2026年已非必需 |
| Register Token | “Sink token” | 额外的可学习token，用于吸收ViT在预训练期间产生的高范数注意力伪影 |
| Position Embedding | “位置信息” | 使序列感知顺序的位置向量或旋转；2D-RoPE是现代默认选项 |
| Grid | “Patch网格” | 给定分辨率和patch大小下的(H/P) x (W/P)二维patch数组 |
| NaFlex | “原生灵活分辨率” | SigLIP 2的特性：单一模型无需重新训练即可服务多种纵横比和分辨率 |
| Backbone | “视觉塔” | 预训练的图像编码器，其patch-token输出作为VLM中LLM的输入 |
| Pooling | “图像级摘要” | 将patch token转换为单个向量的策略：CLS、均值、注意力池化或基于register |
| Patch 14 vs 16 | “细粒度 vs 粗粒度网格” | Patch 14每张图像产生更多token，OCR保真度更好但速度较慢；Patch 16是经典默认值 |

## 延伸阅读

- [Dosovitskiy 等. —— An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) —— 原始ViT论文。
- [He 等. —— Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) —— MAE，自监督预训练。
- [Oquab 等. —— DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) —— 大规模自蒸馏，无标签。
- [Darcet 等. —— Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) —— register tokens与伪影分析。
- [Tschannen 等. —— SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) —— 2026年默认的视觉塔。
- [Zhai 等. —— Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) —— 经验缩放定律。
