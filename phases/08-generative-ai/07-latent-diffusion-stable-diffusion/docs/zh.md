# 隐式扩散与 Stable Diffusion

> 在 512×512 图像上进行像素空间扩散是计算上的暴殄天物。Rombach 等人（2022）注意到，生成图像并不需要全部 78.6 万维——只需足够捕捉语义结构的维度，其余部分交给单独的解码器即可。在 VAE 的隐空间中运行扩散。这一思想就是 Stable Diffusion。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**时间：** ~75 分钟

## 问题所在

512² 像素空间扩散意味着 U-Net 在 `[B, 3, 512, 512]` 形状的张量上运行。每个采样步骤对于 5 亿参数的 U-Net 约为 ~100 GFLOPS。五十步就是每张图像 5 TFLOPS。训练十亿张图像，计算账单将荒谬至极。

这些 FLOPS 大多用于将感知上不重要的细节推过网络——有损 VAE 可以压缩掉的高频纹理。Rombach 的想法：训练一次 VAE（*第一阶段*），冻结它，然后在 4 通道 64×64 的隐空间中完全运行扩散（*第二阶段*）。同一个 U-Net。1/16 的像素。质量相当，FLOPS 减少约 64 倍。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 使用在 `64×64×4` 隐式上运行的 8.6 亿参数 U-Net，SDXL 使用在 `128×128×4` 上的 26 亿参数 U-Net，SD3 将 U-Net 替换为使用流匹配的 Diffusion Transformer (DiT)。Flux.1-dev（Black Forest Labs, 2024）搭载 120 亿参数的 DiT-MMDiT。它们都运行在相同的两阶段基底上。

## 核心概念

![隐式扩散：VAE 压缩 + 隐空间扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **第一阶段 — VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴 8× 下采样 + 调整通道数，使总隐式大小约为像素数的 1/16。损失 = 重建（L1 + LPIPS 感知损失）+ KL（权重较小，因此 `z` 不会被强制过于高斯，因为我们不需要从 `z` 精确采样）。通常配合对抗损失训练，使解码图像更锐利。

2. **第二阶段 — 在 `z` 上的扩散。** 将 `z = E(x_real)` 视为数据。训练 U-Net（或 DiT）去噪 `z_t`。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件。** 两个额外组件。冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。交叉注意力注入：每个 U-Net 块接收 `[Q = image features, K = V = text tokens]` 并将其融合。这些 token 是文本影响图像的唯一途径。

**损失函数与第 06 课完全相同。** 同样的 DDPM / 流匹配 MSE 噪声损失。只是换了个数据域。

## 架构变体

| 模型 | 年份 | 骨干网络 | 隐式形状 | 文本编码器 | 参数量 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT 替换 U-Net（在隐式 patch 上的 transformer），扩大文本编码器规模（T5 在提示遵循度上优于 CLIP），增加隐式通道数（4 → 16 提供更多细节余量）。

## 动手实现

`code/main.py` 在第 06 课的 DDPM 之上堆叠了一个玩具 1-D "VAE"（恒等编码器+解码器，用于演示；真实的 VAE 会是卷积网络），并添加了带 classifier-free guidance 的类别条件。它展示了相同的扩散损失无论运行在原始 1-D 值还是编码值上都有效——这是关键洞见。

### 步骤 1：编码器/解码器

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真实的 VAE 有训练好的权重。为教学目的，这个线性映射足以展示扩散在 `z` 上操作，而不关心原始数据空间。

### 步骤 2：在 `z` 空间中的扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` 解码。

### 步骤 3：classifier-free guidance

训练时，10% 的时间丢弃类别标签（替换为 null token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无 guidance（完全多样性），`w = 3` = 默认值，`w = 7+` = 饱和/过度锐利。

### 步骤 4：文本条件（概念，非代码）

将类别标签替换为冻结文本编码器的输出。通过交叉注意力将文本嵌入馈送给 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类别条件扩散模型与 Stable Diffusion 之间唯一实质性的区别。

## 常见陷阱

- **VAE 尺度不匹配。** SD 1.x 的 VAE 在编码后应用了一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这个会使 U-Net 在方差严重错误的隐式上训练。每个 checkpoint 都包含这个值。
- **文本编码器静默出错。** SD3 需要 T5-XXL 且 >=128 tokens，回退到仅 CLIP 会有损。务必检查 `use_t5=True`，否则提示保真度会暴跌。
- **混用隐空间。** SDXL、SD3、Flux 使用不同的 VAE。在 SDXL 隐式上训练的 LoRA 无法在 SD3 上工作。Hugging Face diffusers 0.30+ 会拒绝加载不匹配的 checkpoint。
- **CFG 过高。** `w > 10` 产生饱和、油腻的图像，以牺牲多样性为代价过度拟合提示。甜蜜点是 `w = 3-7`。
- **负提示泄漏。** 空负提示成为 null token；填充的负提示成为 `ε_uncond`。两者不同；某些 pipeline 静默默认为 null。

## 实际应用

2026 年的生产栈：

| 目标 | 推荐骨干网络 |
|--------|----------------------|
| 窄领域，成对数据，从头训练模型 | SDXL fine-tune (LoRA / full) — 最快上线 |
| 开放域文生图，开放权重 | Flux.1-dev (12B, Apache / 非商业) 或 SD3.5-Large |
| 最快推理，开放权重 | Flux.1-schnell (1-4 step, Apache) 或 SDXL-Lightning |
| 最佳提示遵循度，托管服务 | GPT-Image / DALL-E 3 (仍是), Midjourney v7, Imagen 4 |
| 编辑工作流 | Flux.1-Kontext (2024 年 12 月) — 原生接受图像 + 文本 |
| 研究，基线 | SD 1.5 — 古老但研究充分 |

## 交付

保存 `outputs/skill-sd-prompter.md`。该技能接收文本提示 + 目标风格，输出：模型 + checkpoint、CFG scale、采样器、负提示、分辨率、可选的 ControlNet/IP-Adapter 组合，以及每步 QA 检查清单。

## 练习

1. **简单。** 运行 `code/main.py`，guidance 为 `w ∈ {0, 1, 3, 7, 15}`。按类别记录平均样本。在 `w` 为多少时，类别均值会偏离真实数据均值？
2. **中等。** 将玩具线性编码器替换为带重建损失的 tanh-MLP 编码器/解码器对。在新的隐式上重新训练扩散。样本质量会变化吗？
3. **困难。** 用 diffusers 搭建真实的 Stable Diffusion 推理：加载 `sdxl-base`，运行 30 步 Euler，CFG=7，计时。然后切换到 `sdxl-turbo`，4 步，CFG=0。相同主题，不同质量——描述变化及原因。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| First stage | "The VAE" | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| Second stage | "The U-Net" | 在隐空间上的扩散模型。 |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`；调节条件强度。 |
| Null token | "Empty prompt embed" | 用于 `ε_uncond` 的无条件嵌入。 |
| Cross-attention | "How text gets in" | 每个 U-Net 块将文本 token 作为 K 和 V 进行注意力计算。 |
| DiT | "Diffusion Transformer" | 用 transformer 替换 U-Net，在隐式 patch 上操作；扩展性更好。 |
| MMDiT | "Multi-modal DiT" | SD3 的架构：文本和图像流联合注意力。 |
| VAE scaling factor | "Magic number" | 将隐式除以 ~5.4，使扩散在单位方差空间中操作。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

参考的 Flux 集成是典型的"我有消费级 GPU，能上线吗？"方案。诀窍是将生产推理文献中列出的相同三旋钮配方应用于扩散 DiT：

1. **交错加载。** Flux 有三个网络，它们从不需要同时存在于 VRAM 中：T5-XXL 文本编码器（fp32 约 10 GB）、CLIP-L（小）、120 亿参数的 MMDiT，以及 VAE。先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。8GB 消费级 GPU 一次只能容纳一个阶段。
2. **通过 bitsandbytes 进行 4-bit 量化。** 对 T5 编码器和 DiT 都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。内存减少 8 倍，根据 Aritra 的基准测试（链接在 notebook 中），文生图质量下降不可感知。
3. **CPU offload。** `pipe.enable_model_cpu_offload()` 在每个前向传播推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 延迟，但使 pipeline 能够运行。

内存计算为：`10 GB T5 / 8 = 1.25 GB` 量化，`12 B params × 0.5 bytes = ~6 GB` 量化 DiT，加上激活值。用 stas00 的话说，这是 TP=1 推理的极端情况——无模型并行，最大量化。生产环境你会在 H100 上运行 TP=2 或 TP=4；对于单个开发笔记本，这就是方案。

## 延伸阅读

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 家族。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述所有 checkpoint 的参考实现。
