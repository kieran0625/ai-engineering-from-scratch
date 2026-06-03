# 扩散 Transformer 与整流流

> U-Net 并非扩散的秘密。将其替换为 Transformer，把噪声调度换成直线流，你就得到了 SD3、FLUX 以及所有 2026 年的文生图模型。

**类型：** 学习 + 构建
**语言：** Python
**前置知识：** 阶段 4 第 10 课（扩散 DDPM）、阶段 4 第 14 课（ViT）、阶段 7 第 02 课（自注意力）
**时间：** ~75 分钟

## 学习目标

- 追溯从 U-Net DDPM（第 10 课）到扩散 Transformer（DiT）、MMDiT（SD3）以及单流+双流 DiT（FLUX）的演进
- 解释整流流：为什么噪声与数据之间的直线路径能让模型在 20 步而非 1000 步内完成采样
- 实现一个微型 DiT 模块和整流流训练循环，均在 100 行以内
- 通过架构、参数量和许可证区分各模型变体（SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、Qwen-Image）

## 问题

第 10 课用 U-Net 去噪器构建了 DDPM。这一方案主导了 2020-2023 年：U-Net + beta 调度 + 噪声预测损失。它产出了 Stable Diffusion 1.5 和 2.1 以及 DALL-E 2。

每个 2026 年的最先进文生图模型都已超越它。Stable Diffusion 3、FLUX、SD4、Z-Image、Qwen-Image、Hunyuan-Image——没有一家使用 U-Net。它们使用扩散 Transformer（DiT）。SD3 和 FLUX 还将 DDPM 噪声调度替换为整流流，它将噪声到数据的路径拉直，并通过一致性蒸馏变体实现 1-4 步推理。

这一转变至关重要，因为它是基于扩散的图像生成变得可控、提示精准（SD3/SD4 解决了文本渲染）且达到生产速度的原因。理解 DiT + 整流流就是理解 2026 年的生成式图像技术栈。

## 概念

### 从 U-Net 到 Transformer

```mermaid
flowchart LR
    subgraph UNET["DDPM U-Net (2020)"]
        U1["Conv encoder"] --> U2["Conv bottleneck"] --> U3["Conv decoder"]
    end
    subgraph DIT["DiT (2023)"]
        D1["Patch embed"] --> D2["Transformer blocks"] --> D3["Unpatchify"]
    end
    subgraph MMDIT["MMDiT (SD3, 2024)"]
        M1["Text stream"] --> M3["Joint attention<br/>(separate weights per modality)"]
        M2["Image stream"] --> M3
    end
    subgraph FLUX["FLUX (2024)"]
        F1["Double-stream blocks<br/>(text + image separate)"] --> F2["Single-stream blocks<br/>(concat + shared weights)"]
    end

    style UNET fill:#e5e7eb,stroke:#6b7280
    style DIT fill:#dbeafe,stroke:#2563eb
    style MMDIT fill:#fef3c7,stroke:#d97706
    style FLUX fill:#dcfce7,stroke:#16a34a
```

- **DiT**（Peebles & Xie, 2023）—— 将 U-Net 替换为类 ViT 的 Transformer，在潜在块上操作。通过自适应层归一化（AdaLN）进行条件化。
- **MMDiT**（SD3, Esser et al., 2024）—— 两条流，文本和图像令牌使用独立权重，共享联合注意力。
- **FLUX**（Black Forest Labs, 2024）—— 前 N 个块为类 SD3 的双流，后续块拼接并共享权重（单流），以在更大深度下提升效率。
- **Z-Image**（2025）—— 高效的 6B 参数单流 DiT，挑战"不惜一切代价地扩展"。

### 整流流一段话

DDPM 将前向过程定义为噪声 SDE，其中 `x_t` 被逐渐破坏。学习的反向过程是第二个 SDE，通过 1000 个小步求解。

整流流定义了干净数据与纯噪声之间的**直线**插值：

```
x_t = (1 - t) * x_0 + t * epsilon,     t in [0, 1]
```

训练网络预测速度 `v_theta(x_t, t) = epsilon - x_0`——沿从干净数据到噪声的直线路径的前向方向（`dx_t/dt`）。采样时，反向积分这一速度，从噪声走向数据。得到的 ODE 更接近直线，因此采样所需的积分步数大大减少。

SD3 称之为**整流流匹配**。FLUX、Z-Image 和大多数 2026 年模型使用相同的目标函数。典型推理：20-30 步 Euler（确定性），对比旧 DDPM 方案中的 50+ 步 DDIM。蒸馏 / turbo / schnell / LCM 变体进一步降至 1-4 步。

### AdaLN 条件化

DiT 通过**自适应层归一化**对时间步和类别/文本进行条件化：从条件向量预测 `scale` 和 `shift`，并在 LayerNorm 后应用。比 U-Net 中的 FiLM 式调制更简洁，也是每个现代 DiT 的默认选择。

```
cond -> MLP -> (scale, shift, gate)
norm(x) * (1 + scale) + shift, then residual add * gate
```

### SD3 和 FLUX 中的文本编码器

- **SD3** 使用三个文本编码器：两个 CLIP 模型 + T5-XXL。嵌入拼接后作为文本条件输入图像流。
- **FLUX** 使用一个 CLIP-L + T5-XXL。
- **Qwen-Image / Z-Image** 变体使用与其基础 LLM 对齐的自有文本编码器。

文本编码器是 SD3/FLUX 比 SD1.5 更好地理解提示的重要原因。仅 T5-XXL 就有 4.7B 参数。

### 无分类器引导依然适用

整流流改变的是采样器，而非条件化方式。无分类器引导（训练时以 10% 概率丢弃文本，推理时混合条件和非条件预测）在整流流中完全相同。大多数 2026 年模型使用 3.5-5 的引导尺度——低于 SD1.5 的 7.5，因为整流流模型默认更紧密地遵循提示。

### Consistency、Turbo、Schnell、LCM

四个名称，同一理念：将慢速多步模型蒸馏为快速少步模型。

- **LCM（潜在一致性模型）** —— 训练学生网络从任意中间 `x_t` 一步预测最终 `x_0`。
- **SDXL Turbo / FLUX schnell** —— 通过对抗扩散蒸馏训练的 1-4 步模型。
- **SD Turbo** —— 适配到潜在扩散的 OpenAI 风格一致性模型。

任何新模型的生产部署都会同时发布"完整质量"检查点和"turbo / schnell"变体。Schnell（德语"快速"，Black Forest Labs 的惯例）在 1-4 步内运行，适合实时流水线。

### 2026 年模型格局

| 模型 | 规模 | 架构 | 许可证 |
|-------|------|--------------|---------|
| Stable Diffusion 3 Medium | 2B | MMDiT | SAI Community |
| Stable Diffusion 3.5 Large | 8B | MMDiT | SAI Community |
| FLUX.1-dev | 12B | 双流 + 单流 DiT | non-commercial |
| FLUX.1-schnell | 12B | 同上，蒸馏 | Apache 2.0 |
| FLUX.2 | — | FLUX.1 迭代 | mixed |
| Z-Image | 6B | S3-DiT（可扩展单流） | permissive |
| Qwen-Image | ~20B | DiT + Qwen 文本塔 | Apache 2.0 |
| Hunyuan-Image-3.0 | ~80B | DiT | research |
| SD4 Turbo | 3B | DiT + 蒸馏 | SAI Commercial |

FLUX.1-schnell 是 2026 年开源默认选择。Z-Image 是效率领导者。FLUX.2 和 SD4 是当前质量巅峰。

### 为什么这一范式转变重要

DDPM + U-Net 有效。DiT + 整流流**更好、更快、扩展更干净**。这一转变类似于 NLP 中从 RNN 到 Transformer 的过渡：两种架构解决了相同问题，但 Transformer 扩展性更好并占据主导。2026 年关于图像、视频或 3D 生成的每篇论文都使用 DiT 形状的去噪器，且通常采用整流流目标函数。U-Net DDPM 现在主要用于教学（第 10 课）。

## 构建

### 步骤 1：带 AdaLN 的 DiT 模块

```python
import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """
    Adaptive LayerNorm with a gate. Predicts (scale, shift, gate) from the conditioning.
    Init such that the whole block starts as identity ("zero init").
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Linear(cond_dim, dim * 3)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.mlp(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h, gate.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4, cond_dim=192):
        super().__init__()
        self.adaln1 = AdaLNZero(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.adaln2 = AdaLNZero(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h, gate1 = self.adaln1(x, cond)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1 * a
        h, gate2 = self.adaln2(x, cond)
        x = x + gate2 * self.mlp(h)
        return x
```

`AdaLNZero` 初始为恒等映射，因为其 MLP 权重初始化为零。训练将模块从恒等状态推开；这极大地稳定了深度 Transformer 扩散模型。

### 步骤 2：微型 DiT

```python
def timestep_embedding(t, dim):
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TinyDiT(nn.Module):
    def __init__(self, image_size=16, patch_size=2, in_channels=3, dim=96, depth=4, heads=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, cond_dim=dim) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        n = x.size(0)
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2) + self.pos
        t_emb = self.time_mlp(timestep_embedding(t, self.pos.size(-1)))
        for blk in self.blocks:
            x = blk(x, t_emb)
        x = self.norm_out(x)
        x = self.head(x)
        return self._unpatchify(x, n)

    def _unpatchify(self, x, n):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        x = x.view(n, h, w, p, p, -1).permute(0, 5, 1, 3, 2, 4).reshape(n, -1, h * p, w * p)
        return x
```

### 步骤 3：整流流训练

```python
import torch.nn.functional as F

def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon

    target_velocity = epsilon - x0
    pred_velocity = model(x_t, t)

    loss = F.mse_loss(pred_velocity, target_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

与 DDPM 的噪声预测损失（第 10 课）对比：结构相同，目标不同。不是预测噪声 `epsilon`，而是预测**速度** `epsilon - x_0`，它沿直线插值从数据指向噪声。

### 步骤 4：Euler 采样器

整流流是一个 ODE。Euler 方法是最简单的，对于训练良好的整流流模型，在 20+ 步时几乎与高阶求解器同样精确。

```python
@torch.no_grad()
def rectified_flow_sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

20 步。在训练好的模型上，这能产生与 1000 步 DDPM 相当的样本。

### 步骤 5：端到端冒烟测试

```python
import numpy as np

def synthetic_blobs(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((num, 3, size, size), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num):
        cx, cy = rng.uniform(4, size - 4, size=2)
        r = rng.uniform(2, 4)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        colour = rng.uniform(-1, 1, size=3)
        for c in range(3):
            out[i, c][mask] = colour[c]
    return torch.from_numpy(out)
```

用整流流在此上训练 `TinyDiT`。500 步后，采样输出应呈现淡淡的颜色斑块。

## 使用

对于使用 FLUX / SD3 / Z-Image 的真实图像生成，`diffusers` 为每个模型提供统一 API：

```python
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    prompt="a golden retriever surfing a tsunami, hyperrealistic, studio lighting",
    guidance_scale=0.0,           # schnell was trained without CFG
    num_inference_steps=4,
    max_sequence_length=256,
).images[0]
out.save("surf.png")
```

三行代码。`FLUX.1-schnell` 四步完成。将模型 ID 换为 `black-forest-labs/FLUX.1-dev`，配合 CFG 在 20-30 步内获得更高质量。

对于 SD3：

```python
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
).to("cuda")
out = pipe(prompt, guidance_scale=3.5, num_inference_steps=28).images[0]
```

## 交付

本课产出：

- `outputs/prompt-dit-model-picker.md` —— 根据质量、延迟和许可证约束在 SD3、FLUX.1-dev、FLUX.1-schnell、Z-Image、SD4 Turbo 之间做出选择。
- `outputs/skill-rectified-flow-trainer.md` —— 编写完整的整流流训练循环，包含 AdaLN DiT 和 Euler 采样。

## 练习

1. **（简单）** 在上述合成 blob 数据集上训练 TinyDiT 500 步。比较 10、20 和 50 步 Euler 采样产生的样本。
2. **（中等）** 添加文本条件化，将学习到的类别嵌入拼接到时间嵌入（10 个按颜色区分的 blob"类别"）。用类别 0、5 和 9 采样，验证颜色匹配。
3. **（困难）** 计算相同规模网络在相同数据上训练相同步数后，整流流版本与 DDPM 版本生成样本之间的 Fréchet 距离（FID 代理）。报告哪个收敛更快。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| DiT | "扩散 Transformer" | 替代 U-Net 作为扩散去噪器的 Transformer；在块化潜在上操作 |
| AdaLN | "自适应层归一化" | 通过学习的尺度、偏移、门控在时间步/文本条件化后应用；每个现代 DiT 的标准配置 |
| MMDiT | "多模态 DiT（SD3）" | 文本和图像令牌使用独立权重流，共享联合自注意力 |
| 单流 / 双流 | "FLUX 技巧" | 前 N 块双流（每模态独立权重），后续块单流（拼接 + 共享权重）以提升效率 |
| 整流流 | "直线噪声到数据" | 数据与噪声之间的线性插值；网络预测速度；推理需要更少 ODE 步数 |
| 速度目标 | "epsilon - x_0" | 整流流中的回归目标；从干净数据指向噪声 |
| CFG 引导 | "无分类器引导" | 混合条件和非条件预测；在整流流模型中仍被使用 |
| Schnell / turbo / LCM | "1-4 步蒸馏" | 从完整质量模型蒸馏出的小步变体；生产实时应用 |

## 延伸阅读

- [Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) —— DiT 论文
- [Scaling Rectified Flow Transformers (Esser et al., SD3 paper)](https://arxiv.org/abs/2403.03206) —— 大规模 MMDiT 和整流流
- [FLUX.1 model card and technical report (Black Forest Labs)](https://huggingface.co/black-forest-labs/FLUX.1-dev) —— 双流 + 单流细节
- [Z-Image: Efficient Image Generation Foundation Model (2025)](https://arxiv.org/html/2511.22699v1) —— 6B 单流 DiT
- [Elucidating the Design Space of Diffusion (Karras et al., 2022)](https://arxiv.org/abs/2206.00364) —— 每个扩散设计权衡的参考
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) —— LCM-LoRA 如何实现 4 步推理
