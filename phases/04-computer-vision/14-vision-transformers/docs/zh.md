# Vision Transformers (ViT)

> 把图像切成小块，把每块当作一个词，跑一个标准 transformer。别回头。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 Lesson 02 (Self-Attention), Phase 4 Lesson 04 (Image Classification)
**时间：** ~45 分钟

## 学习目标

- 从零实现 patch embedding、可学习位置编码、class token 和 transformer encoder block，构建一个最小化的 ViT
- 解释为什么 ViT 曾被认为需要海量预训练数据，直到 DeiT 和 MAE 证明并非如此
- 比较 ViT、Swin 和 ConvNeXt 的架构先验（无先验、局部窗口注意力、卷积骨干）
- 使用 `timm` 和标准 linear-probe / fine-tune 流程，在小数据集上微调预训练 ViT

## 问题背景

十年来，卷积几乎就是计算机视觉的代名词。CNN 拥有强大的归纳偏置——局部性、平移等变性——没人认为这能被替代。然后 Dosovitskiy 等人（2020）证明，一个直接应用于展平图像块的普通 transformer，没有任何卷积机制，在足够规模下可以匹敌甚至超越最好的 CNN。

关键是"足够规模"。ViT 在 ImageNet-1k 上输给了 ResNet。但在 ImageNet-21k 或 JFT-300M 上预训练、再在 ImageNet-1k 上微调后，ViT 反超了。结论是 transformer 缺乏有用的先验，但可以从足够多的数据中学会。后续工作（DeiT、MAE、DINO）表明，只要有正确的训练配方——强数据增强、自监督预训练、知识蒸馏——ViT 在小数据上也能很好地训练。

到 2026 年，纯 CNN 在边缘设备上仍有竞争力（ConvNeXt 是最强的），但 transformer 主导了其他所有领域：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。ViT 的块结构是你需要掌握的核心。

## 核心概念

### 流程

```mermaid
flowchart LR
    IMG["Image<br/>(3, 224, 224)"] --> PATCH["Patch embedding<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["Flatten to<br/>(196, 768) tokens"]
    FLAT --> CAT["Prepend<br/>[CLS] token"]
    CAT --> POS["Add learned<br/>positional embed"]
    POS --> ENC["N transformer<br/>encoder blocks"]
    ENC --> CLS["Take [CLS]<br/>token output"]
    CLS --> HEAD["MLP classifier"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七个步骤。Patches -> tokens -> attention -> classifier。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）只改动其中一两个步骤，其余保持不变。

### Patch embedding

第一个卷积是关键。Kernel size 16，stride 16，所以 224x224 的图像变成 14x14 的网格，每个格子是 16x16 的 patch，每个 patch 被投影到 768 维的 embedding。这一个卷积同时完成了 patch 化和线性投影。

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196 个 patches = 196 个 tokens。每个 token 的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### Class token

一个可学习的向量被 prepend 到序列开头：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 transformer block 后，`[CLS]` 的输出就是全局图像表示。分类头只读取这一个向量。

### 位置编码

Transformer 没有内置的空间位置概念。给每个 token 加上一个可学习的向量：

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

这个 embedding 是模型的参数；基于梯度的训练会把它适配到二维图像结构。正弦 2D 替代方案存在，但实践中很少使用。

### Transformer encoder block

标准的。多头自注意力、MLP、残差连接、pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的 block，每个有 12 个注意力头，总共 86M 参数。

### 为什么用 pre-LN

早期 transformer 使用 post-LN（`x = LN(x + sublayer(x))`），在没有 warmup 的情况下训练超过 6-8 层会很困难。Pre-LN（`x = x + sublayer(LN(x))`）可以稳定地训练更深的网络，无需 warmup。每个 ViT 和每个现代 LLM 都使用 pre-LN。

### Patch size 的权衡

- 16x16 patches -> 196 tokens，标准配置。
- 32x32 patches -> 49 tokens，更快但分辨率更低。
- 8x8 patches -> 784 tokens，更精细但 O(n^2) 的注意力成本增长很快。

更大的 patch = 更少的 token = 更快但空间细节更少。SwinV2 在分层窗口中使用 4x4 patches。

### DeiT 在 ImageNet-1k 上训练 ViT 的配方

原始 ViT 需要 JFT-300M 才能打败 CNN。DeiT（Touvron et al., 2020）仅通过四个改动，就在 ImageNet-1k 上将 ViT-B 训练到了 81.8% top-1：

1. 重度数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（训练时随机丢弃整个 block）。
3. 重复增强（每个 batch 中同一张图像采样 3 次）。
4. 从 CNN 教师模型蒸馏（可选，进一步提升精度）。

每个现代 ViT 训练配方都源自 DeiT。

### Swin vs ConvNeXt

- **Swin**（Liu et al., 2021）——基于窗口的注意力。每个 block 在局部窗口内做注意力；交替 block 会移动窗口以在窗口间混合信息。在保留注意力操作的同时，重新引入了类 CNN 的局部性先验。
- **ConvNeXt**（Liu et al., 2022）——重新设计的 CNN，匹配了 Swin 的架构选择（depthwise convs、LayerNorm、GELU、inverted bottleneck）。证明差距不在于"注意力 vs 卷积"，而在于"现代训练配方 + 架构"。

2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级方案；选择取决于你的推理栈（ConvNeXt 在边缘设备上编译更好）和预训练语料。

### MAE 预训练

Masked Autoencoder（He et al., 2022）：随机 mask 75% 的 patches，训练 encoder 只处理可见的 25%，训练一个小的 decoder 从 encoder 输出重建被 mask 的 patches。预训练后丢弃 decoder，微调 encoder。

MAE 让 ViT 仅在 ImageNet-1k 上就能训练，达到 SOTA，是目前默认的自监督预训练配方。

## 动手实现

### Step 1: Patch embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积、一个展平、一个转置。这就是图像到 tokens 的全部步骤。

### Step 2: Transformer block

Pre-LN、多头自注意力、GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 处理拆分成 heads、缩放点积和输出投影。`batch_first=True` 所以形状是 `(N, seq, dim)`。

### Step 3: The ViT

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 2.8M 参数——一个可在 CPU 上运行的小 ViT。真实 ViT-B 是 86M；同样的类定义，`dim=768, depth=12, num_heads=12`。

### Step 4: Sanity check — 单图推理

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应该无错误运行。概率之和为 1。

## 实际使用

`timm` 为每个 ViT 变体提供了 ImageNet 预训练权重。一行代码：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm` 是 2026 年 vision transformer 的生产默认选择。支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 等数十种模型，统一 API。

对于多模态工作（图像 + 文本），`transformers` 提供 CLIP、SigLIP、BLIP-2、LLaVA。这些模型的图像编码器都是 ViT 变体。

## 交付产出

本节课产生：

- `outputs/prompt-vit-vs-cnn-picker.md` —— 一个根据数据集大小、计算资源和推理栈在 ViT、ConvNeXt 或 Swin 之间做选择的 prompt。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` —— 一个验证 ViT 的 patch embedding 和位置编码形状是否与模型期望的序列长度匹配的技能，捕获最常见的移植 bug。

## 练习题

1. **(Easy)** 打印上述小 ViT 前向传播中每个中间 tensor 的形状。确认：输入 `(N, 3, 64, 64)` -> patches `(N, 16, 192)` -> 加入 CLS 后 `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **(Medium)** 在 Lesson 4 的 synthetic-CIFAR 数据集上微调预训练的 `timm` ViT-S/16。与在同一数据上微调的 ResNet-18 对比。报告训练时间和最终精度。
3. **(Hard)** 为小 ViT 实现 MAE 预训练：mask 75% 的 patches，训练 encoder + 小 decoder 重建被 mask 的 patches。评估预训练前后在 synthetic 数据上的 linear-probe 精度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch embedding | "第一个卷积" | 卷积核大小 = 步长 = patch 大小的卷积；将图像变成 token embedding 的网格 |
| Class token | "[CLS]" | 一个可学习的向量，prepend 到 token 序列；其最终输出是全局图像表示 |
| Positional embedding | "Learned pos" | 一个可学习的向量，加到每个 token 上，让 transformer 知道每个 patch 来自哪里 |
| Pre-LN | "LayerNorm 在子层之前" | 稳定的 transformer 变体：`x + sublayer(LN(x))` 而非 `LN(x + sublayer(x))` |
| Multi-head attention | "并行注意力" | 标准 transformer 注意力拆分成 num_heads 个独立子空间，之后拼接 |
| ViT-B/16 | "Base, patch 16" | 经典尺寸：dim=768, depth=12, heads=12, patch_size=16, image=224；~86M 参数 |
| DeiT | "Data-efficient ViT" | 仅用强数据增强在 ImageNet-1k 上训练的 ViT；证明大规模预训练数据集并非严格必需 |
| MAE | "Masked autoencoder" | 自监督预训练：mask 75% patches，重建；主导的 ViT 预训练配方 |

## 延伸阅读

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) —— ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) —— 如何仅在 ImageNet-1k 上训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) —— MAE 预训练
- [timm documentation](https://huggingface.co/docs/timm) —— 生产环境中你会用到的每个 vision transformer 的参考文档
