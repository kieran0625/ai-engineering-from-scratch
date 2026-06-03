# 图像修复、外扩与图像编辑

> 文生图创造新事物。图像修复修复旧事物。在生产环境中，70% 的可计费图像工作是编辑——更换背景、移除 Logo、扩展画布、重新生成手部。图像修复是扩散模型真正发挥价值的地方。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**时间：** ~75 分钟

## 问题

客户发来一张完美的产品照片，但背景中有一个分散注意力的标志。你想擦除这个标志，同时让其他所有像素保持完全一致。你不能从头运行文生图——结果会有不同的颜色、不同的光照、不同的产品角度。你只想重新生成*被遮罩的区域*，并且希望重新生成的部分与周围上下文协调一致。

这就是图像修复。其变体包括：

- **图像修复 (Inpainting)。** 在遮罩内部重新生成，保留外部像素。
- **图像外扩 (Outpainting)。** 在遮罩外部（或画布之外）重新生成，保留内部内容。
- **图像编辑 (Image editing)。** 重新生成整张图像，但保持对原始图像的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年的每个扩散流水线都内置了图像修复模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于相同的原理工作。

## 概念

![图像修复：基于遮罩的去噪与上下文保留重注入](../assets/inpainting.svg)

### 朴素方法（以及为什么它是错的）

用标准文生图配合遮罩运行。在每个采样步骤中，用干净图像的前向扩散版本替换噪声潜在表示中未遮罩的区域。它能工作……但效果很差。边界伪影会渗透出来，因为模型对被遮罩区域中的内容一无所知。

### 正确的图像修复模型

训练一个修改后的 U-Net，接收 9 个输入通道而非 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码后的源图像副本，加上一个单通道遮罩。在训练时，随机遮罩图像的区域，并训练模型仅对被遮罩区域进行去噪，同时未遮罩区域作为干净的 conditioning 信号提供。在推理时，模型可以"看到"被遮罩区域周围的内容，并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022) — 自由编辑

对源图像添加噪声直到某个中间 `t`，然后从 `t` 到 0 运行反向链，使用新的提示词。无需重新训练。起始 `t` 的选择在保真度和创作自由度之间权衡：

- `t/T = 0.3` → 与源图几乎一致，仅有小幅风格变化
- `t/T = 0.6` → 适度编辑，保留粗略结构
- `t/T = 0.9` → 从接近噪声的状态生成，极少保留源图信息

### InstructPix2Pix (Brooks et al., 2023)

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，以输入图像和文本指令为条件（"改成日落"、"加一条龙"）。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint (Lugmayr et al., 2022)

保持标准的无条件扩散模型。在每个反向步骤中，重新采样——偶尔跳回到更噪声的状态并重新生成。避免边界伪影。在没有训练好的图像修复模型时使用。

## 动手构建

`code/main.py` 在 5 维数据上实现了一个简单的 1-D 图像修复方案。我们在来自两个聚类的 5 维混合数据上训练 DDPM，每个样本是 5 个浮点数。推理时，我们"遮罩" 5 个维度中的 2 个，在每个步骤注入未遮罩 3 个维度的噪声前向版本，并仅重新生成被遮罩的维度。

### 步骤 1：5-D DDPM 数据

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步骤 2：在所有 5 个维度上训练去噪器

标准 DDPM。网络对 5-D 噪声输入输出 5-D 噪声预测。

### 步骤 3：推理时，遮罩感知的反向过程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，在简单的 1-D 数据上能工作。真实图像修复使用 9 通道输入，因为纹理连贯性更重要。

### 步骤 4：图像外扩

图像外扩是遮罩反转的图像修复：遮罩新的（之前不存在的）画布区域，用原始图像填充其余部分。相同的训练目标。

## 陷阱

- **接缝。** 朴素方法会留下可见边界，因为梯度信息不会跨越遮罩流动。修复：将遮罩膨胀 8-16 像素，或使用正确的图像修复模型。
- **遮罩泄漏。** 如果条件图像的未遮罩区域质量低或有噪声，它会污染遮罩内部的生成。轻微去噪或模糊处理。
- **CFG 与遮罩大小相互作用。** 小遮罩配高 CFG = 饱和斑块。小编辑时降低 CFG。
- **SDEdit 保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能丢失主体身份。需要扫描和检查点。
- **提示词不匹配。** 提示词应描述*整张*图像，而非仅新内容。"一只猫坐在椅子上"而非"一只猫"。

## 应用

| 任务 | 流水线 |
|------|--------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准提示词 |
| 替换天空 | SD-Inpaint + "日落时的蓝天" |
| 扩展画布 | SDXL 外扩模式（8px 羽化）或 Flux-Fill 配外扩遮罩 |
| 重新生成手部/面部 | SD-Inpaint 配重新描述主体的提示词 + ControlNet-Openpose |
| 改变某区域风格 | 在遮罩区域上用 `t/T=0.5` 运行 SDEdit |
| "改成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM 遮罩 → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image（托管）用于最难的案例 |

SAM（Meta 的 Segment Anything，2023）+ 扩散图像修复是 2026 年的背景移除流水线。SAM 2（2024）支持视频。

## 交付

保存 `outputs/skill-editing-pipeline.md`。该技能接收原始图像 + 编辑描述 + 可选遮罩（或 SAM 提示词），输出：遮罩生成方法、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或图像修复模式，以及 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，将被遮罩维度的比例从 0.2 变化到 0.8。在哪个比例下，图像修复质量（遮罩维度的残差）等于无条件生成？
2. **中等。** 实现 RePaint：每 10 个反向步骤，跳回 5 步（添加噪声）并重新去噪。测量它是否减少了遮罩边缘的边界残差。
3. **困难。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose 与 Flux.1-Fill 在 20 个面部重新生成任务上的表现。分别对姿势遵循度和身份保留度打分。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Inpainting | "填补空洞" | 在遮罩内部重新生成；保留外部像素。 |
| Outpainting | "扩展画布" | 在画布外部重新生成；保留内部内容。 |
| 9-channel U-Net | "正确的图像修复模型" | 以 `noisy \| encoded-source \| mask` 为输入的 U-Net。 |
| SDEdit | "带噪声水平的 Img2img" | 噪声到时间 `t`，用新提示词去噪。 |
| InstructPix2Pix | "纯文本编辑" | 在（图像、指令、输出）三元组上微调的扩散模型。 |
| RePaint | "无需重新训练" | 反向过程中定期重新加噪声以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框选生成遮罩；与图像修复配合使用。 |
| Flux-Kontext | "上下文编辑" | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## 生产备注：编辑流水线对延迟敏感

用户编辑图像时期望低于 5 秒的往返时间。30 步 SDXL-Inpaint 在 1024² 上于 L4 需要 3-4 秒，加上 SAM 遮罩生成（~200 ms）和 VAE 编码/解码（合计 ~500 ms）。在生产环境中，这是 TTFT 受限而非吞吐量受限——batch 1、低并发、最小化每个阶段：

- **SAM-H 是慢的。** SAM-H 在 1024² 上约 ~200 ms；SAM-ViT-B 约 ~40 ms，质量损失轻微。SAM 2（视频）增加时间开销；不要用于单张图像编辑。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 编码为潜在表示。如果已有前次生成的潜在表示（迭代编辑 UI 中常见），通过 `latents=...` 直接传递以跳过一次 VAE 编码。
- **遮罩膨胀也影响吞吐量。** 小遮罩意味着大部分 U-Net 前向传播被浪费（未遮罩像素无论如何会被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 始终运行完整 U-Net；只有 9 通道的 proper-inpaint 变体利用遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 单次前向传播覆盖 `(source_image, instruction)`——无需单独遮罩，无需 SDEdit 噪声扫描。在 H100 上约 1.5 秒完成编辑。架构启示：合并阶段。

## 延伸阅读

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的图像修复。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩来源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — 注意力层编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年工具。
