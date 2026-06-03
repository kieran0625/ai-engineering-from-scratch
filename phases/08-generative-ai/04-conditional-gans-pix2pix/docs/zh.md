# 条件 GAN 与 Pix2Pix

> 2014-2017 年的第一个重大突破是控制 GAN 生成什么内容。附加一个标签、一张图像或一句话。Pix2Pix 实现了图像版本，在狭义的图像到图像任务上，它至今仍优于所有通用文本到图像模型。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**时间：** ~75 分钟

## 问题

无条件 GAN 采样任意的人脸。用于演示还行，生产环境毫无用处。你想要的是：*将草图映射为照片*、*将地图映射为航拍照片*、*将白天场景映射为夜晚*、*为灰度图像上色*。在所有这些任务中，给定输入图像 `x`，必须输出与之有语义对应的 `y`。每个 `x` 都有多个合理的 `y`。均方误差会将它们压成模糊的一团。对抗损失不会，因为"看起来真实"是锐利的。

条件 GAN（Mirza & Osindero, 2014）将条件 `c` 作为输入同时加入 `G` 和 `D`。Pix2Pix（Isola 等人, 2017）对此进行了专门化：条件是一幅完整输入图像，生成器是 U-Net，判别器是基于*块*的分类器（PatchGAN），损失函数为对抗损失 + L1。这个配方即使在 2026 年仍能在狭义图像到图像领域击败从零训练的文本到图像模型，因为它是在*成对数据*上训练的——你拥有恰好需要的信号。

## 概念

![Pix2Pix: U-Net 生成器, PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被忽略）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*成对*的（条件, 输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 看起来是否真实。

**U-Net 生成器。** 带有跨瓶颈跳跃连接的编码器-解码器结构。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN 判别器。** 不输出单一的 real/fake 分数，而是输出一个 `N×N` 网格，每个单元判断约 70×70 像素的感受野。取平均。这是马尔可夫随机场假设：真实性是局部的。训练更快，参数更少，输出更锐利。

**损失函数。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并推动 G 向已知目标靠近。L1 比 L2 产生更锐利的边缘（中位数，而非均值）。`λ = 100` 是 Pix2Pix 的默认设置。

## CycleGAN —— 当你没有成对数据时

Pix2Pix 需要成对的 `(x, y)` 数据。CycleGAN（Zhu 等人, 2017）放弃了这一要求，代价是额外增加一个损失：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 和 `G(F(y)) ≈ y`。这让你无需成对示例就能将马翻译为斑马、夏天翻译为冬天。

在 2026 年，非成对图像到图像翻译主要通过扩散模型（ControlNet、IP-Adapter）完成，而非 CycleGAN，但循环一致性的思想几乎在每篇非成对域适应论文中都有延续。

## 动手实现

`code/main.py` 在一维数据上实现了一个小型条件 GAN。条件 `c` 是一个类别标签（0 或 1）。任务：为给定类别生成来自条件分布的样本。

### 步骤 1：将条件附加到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码是最简单的方式。更大的模型使用学习嵌入、FiLM 调制或交叉注意力。

### 步骤 2：训练条件 GAN

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配给定条件下的真实分布，而非边缘分布。

### 步骤 3：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 陷阱

- **条件被忽略。** G 学习边缘化，D 从不惩罚，因为条件信号太弱。修复：更积极地将条件加入 D（早期层，而非仅后期），使用投影判别器（Miyato & Koyama 2018）。
- **L1 权重过低。** G 漂向任意看起来真实的输出，而非忠实于输入。对于 Pix2Pix 风格任务，初始 λ≈100。
- **L1 权重过高。** G 产生模糊输出，因为 L1 仍然是 L_p 范数。训练稳定后逐渐降低。
- **D 中的真值泄漏。** 将 `(x, y)` 作为 D 的输入拼接，而非仅 `y`。没有这一点，D 无法检查一致性。
- **每类模式坍塌。** 每个类别可能独立坍塌。运行条件多样性检查。

## 应用

2026 年图像到图像任务现状：

| 任务 | 最佳方法 |
|------|----------|
| 草图 → 照片，同域，成对数据 | Pix2Pix / Pix2PixHD（仍然快速、仍然锐利） |
| 草图 → 照片，非成对 | 带 Scribble 条件模型的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散模型；GAN 方法已成遗产 |
| 深度 → 照片 | Stable Diffusion 上的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN (GAN)、ESRGAN-Plus 或 SD-Upscale (扩散) |
| 上色 | ColTran、基于扩散的上色器或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方法 |

当你有（a）数千成对示例、（b）任务狭窄且可重复、（c）需要快速推理时，Pix2Pix 仍是合适工具。在通用开放域任务上，扩散模型胜出。

## 交付

保存 `outputs/skill-img2img-chooser.md`。该技能接收任务描述、数据可用性（成对 vs 非成对，N 个样本）以及延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本、评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py` 添加第三个类别。确认 G 仍能将每个类别的噪声映射到正确的模式。
2. **中等。** 在一维设置中用感知风格损失替换 L1（例如，一个小的冻结 D 作为特征提取器）。它会改变条件分布的锐利度吗？
3. **困难。** 在一维设置中勾勒 CycleGAN：两个分布、两个生成器、循环损失。展示它如何在没有成对数据的情况下学习在它们之间映射。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|----------|
| Conditional GAN | "带标签的 GAN" | G(z, c), D(x, c)。两个网络都看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 带 U-Net G 和 PatchGAN D + L1 损失的成对 cGAN。 |
| U-Net | "带跳跃连接的编码器-解码器" | 对称卷积网络；跳跃连接保留高频信息。 |
| PatchGAN | "局部真实性分类器" | D 输出每块分数而非全局分数。 |
| CycleGAN | "非成对图像翻译" | 两个 G + 循环一致性损失；无需成对数据。 |
| SPADE | "GauGAN" | 用语义图归一化中间激活；分割到图像。 |
| FiLM | "特征级线性调制" | 来自条件的逐特征仿射变换；廉价条件化。 |

## 生产备注：Pix2Pix 作为延迟受限基线

当你有成对数据和狭窄任务（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的单步推理在延迟上比扩散模型快一个数量级。生产中的对比通常是：

| 路径 | 步数 | 512² 单 L4 典型延迟 |
|------|------|----------------------|
| Pix2Pix (U-Net 前向) | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在静态批次中吞吐量胜出（每个请求 FLOPs 相同）。扩散模型在质量和泛化性上胜出。现代做法通常是：为狭窄任务部署 Pix2Pix 风格的蒸馏模型，对尾部输入使用扩散模型回退。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — 投影 D。
