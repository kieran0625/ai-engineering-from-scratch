# Vision Transformers (ViT)

> 图像是由图像块组成的网格。句子是由词元组成的网格。同一个 transformer 可以处理两者。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 4 · 03（CNN）、Phase 4 · 14（Vision Transformers 简介）
**时间：** ~45 分钟

## 问题

2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个 SOTA 都使用 CNN 骨干网络。Transformer 用于语言。

Dosovitskiy 等人（2020）——"An Image is Worth 16x16 Words"——表明可以完全抛弃卷积。将图像切分成固定大小的图像块，将每个图像块线性投影为嵌入，然后将序列输入到普通的 transformer 编码器中。在足够大的规模下（ImageNet-21k 预训练或更大），ViT 能够匹配或超越基于 ResNet 的模型。

ViT 开启了 2026 年更广泛的模式：一种架构，多种模态。Whisper 将音频词元化。ViT 将图像词元化。机器人学中的动作词元。视频中的像素词元。Transformer 并不关心——给它一个序列，他就会学习。

到 2026 年，ViT 及其衍生模型（DeiT、Swin、DINOv2、ViT-22B、SAM 3）主导了大部分视觉领域。CNN 在边缘设备和延迟敏感任务上仍然获胜。其他所有领域都在某个地方使用了 ViT。

## 概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 步骤 1 — 图像块化（patchify）

将 `H × W × C` 图像分割为 `N × (P·P·C)` 的扁平图像块序列。典型设置：`224 × 224` 图像，`16 × 16` 图像块 → 196 个图像块，每个 768 个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

图像块大小是关键杠杆。更小的图像块 = 更多词元、更好的分辨率、二次方的注意力成本。更大的图像块 = 更粗糙、更便宜。

### 步骤 2 — 线性嵌入

一个单一的学习矩阵将每个扁平图像块投影到 `d_model`。等价于核大小为 `P`、步幅为 `P` 的卷积。在 PyTorch 中，这实际上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`——一个两行的实现。

### 步骤 3 — 前置 `[CLS]` 词元，添加位置嵌入

- 前置一个可学习的 `[CLS]` 词元。其最终的隐藏状态是用于分类的图像表示。
- 添加可学习的位置嵌入（ViT 原版）或正弦 2D（后续变体）。
- 在 2024 年之后，RoPE 扩展到 2D 用于位置，有时不使用显式嵌入。

### 步骤 4 — 标准 transformer 编码器

堆叠 L 层 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`。与 BERT 相同。没有视觉特定的层。这是论文的教学要点。

### 步骤 5 — 头部

对于分类：取 `[CLS]` 的隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用图像块嵌入。

### 重要的变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原版。固定图像块大小，完全全局注意力。 |
| DeiT | 2021 | 蒸馏；仅在 ImageNet-1k 上可训练。 |
| Swin | 2021 | 分层，使用移位窗口。固定次二次方成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；适用缩放定律。 |
| SigLIP | 2023 | ViT + 语言对，sigmoid 对比损失。 |
| SAM 3 | 2025 | Segment anything；ViT-Large + 可提示掩码解码器。 |

### 为什么花了一段时间

ViT 需要*大量*数据才能匹配 CNN，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有 >1 亿张标注图像或强自监督预训练的情况下，CNN 在相同计算量下仍然获胜。DeiT 在 2021 年通过蒸馏技巧解决了这个问题；DINOv2 在 2023 年通过自监督永久解决了这个问题。

## 构建

参见 `code/main.py`。纯标准库的图像块化 + 线性嵌入 + 合理性检查。不训练——任何现实规模的 ViT 都需要 PyTorch 和数小时的 GPU 时间。

### 步骤 1：假图像

一个 24 × 24 的 RGB 图像，表示为 `(R, G, B)` 元组的行列表。我们使用 6×6 图像块 → 16 个图像块，每个 108 维嵌入向量。

### 步骤 2：图像块化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：网格的行优先。每个 ViT 都使用这种排序。

### 步骤 3：线性嵌入

将每个扁平图像块乘以一个随机的 `(patch_flat_size, d_model)` 矩阵。验证前置 `[CLS]` 后的输出形状为 `(N_patches + 1, d_model)`。

### 步骤 4：计算现实 ViT 的参数数量

打印 ViT-Base 的参数数量：12 层，12 个头，d=768，patch=16。与 ResNet-50（~25M）比较。ViT-Base 约为 ~86M。ViT-Large ~307M。ViT-Huge ~632M。

## 使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 嵌入是 2026 年图像特征的默认选择。** 冻结骨干网络，训练一个微小的头部。适用于分类、检索、检测、描述。Meta 的 DINOv2 检查点在每项非文本视觉任务上都优于 CLIP。

**图像块大小选择。** 小模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 交付

参见 `outputs/skill-vit-configurator.md`。该技能根据数据集大小、分辨率和计算预算，为新视觉任务选择 ViT 变体和图像块大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证图像块数量等于 `(H/P) * (W/P)`，且扁平图像块维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置嵌入——为每个图像块的 `row` 和 `col` 使用两个独立的正弦编码，拼接起来。将其输入到小型 PyTorch ViT 中，并与 CIFAR-10 上的可学习位置嵌入比较准确率。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上用 4×4 图像块训练。测量测试准确率。现在添加 DINOv2 预训练（简化版：仅训练编码器从掩码图像块预测图像块嵌入）。准确率是否提高？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | "视觉 transformer 的词元" | 图像中 `P × P × C` 区域的像素值的扁平向量。 |
| Patchify | "切碎 + 扁平化" | 将图像切分为不重叠的图像块，每个扁平化为向量。 |
| `[CLS]` 词元 | "图像摘要" | 前置的可学习词元；其最终嵌入是图像表示。 |
| Inductive bias | "模型假设什么" | ViT 比 CNN 有更少的先验；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师无标签训练。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的继任者" | ViT + 文本编码器，使用 sigmoid 对比损失训练；在相同计算量下优于 CLIP。 |
| Swin | "窗口化 ViT" | 分层 ViT，局部注意力 + 移位窗口；次二次方。 |
| Register tokens | "2023 年的技巧" | 几个额外的可学习词元，吸收注意力汇；改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的寄存器词元修复。
