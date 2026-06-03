# ControlNet、LoRA 与条件控制

> 纯文本是一种笨拙的控制信号。ControlNet 让你克隆一个预训练的扩散模型，并用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA 让你仅训练 1000 万个参数就能微调一个 20 亿参数的模型。它们共同将 Stable Diffusion 从一个玩具变成了 2026 年每家机构都在部署的图像生产管线。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 07（Latent Diffusion），Phase 10（LLMs from Scratch — LoRA 基础）
**时间：** ~75 分钟

## 问题所在

像"a woman in a red dress walking a dog on a busy street"这样的提示，无法告诉模型*狗在哪里*、*女人的姿态如何*、*街道的视角是什么*。文本只能确定图像所需信息的约 10%。其余部分是视觉信息，无法用文字高效描述。

为每种信号（姿态、深度、Canny 边缘、分割）从头训练一个新的条件模型是不可行的。你希望保持 26 亿参数的 SDXL 主干网络冻结，附加一个小的旁路网络来读取条件信息，并让它微调主干网络的中间特征。这就是 ControlNet。

你还希望在不需要重新训练完整模型的情况下，教会模型新概念（你的脸、你的产品、你的风格）。你想要一个缩小 100 倍的增量。这就是 LoRA——低秩适配器，插入到现有的注意力权重中。

ControlNet + LoRA + 文本 = 2026 年从业者的工具箱。大多数生产图像管线会在 SDXL / SD3 / Flux 基础模型上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 核心概念

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet（Zhang et al., 2023）

取一个预训练的 SD。*克隆* U-Net 的编码器一半。冻结原始网络。训练克隆网络接受额外的条件输入（边缘、深度、姿态）。通过*零卷积*跳跃连接（1×1 卷积，初始化为零——开始时无操作，学习增量）将克隆网络连接回原始网络的解码器一半。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 开始时是恒等映射——训练前也不会造成破坏。使用标准扩散损失，在 100 万个（提示、条件、图像）三元组上训练。

每种模态的 ControlNet 作为小型旁路模型发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA（Hu et al., 2021）

对于模型中的任意线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力层通常使用秩 4-16，重度微调使用秩 64-128。新增参数数量为 `2 · d · r`，而非 `d²`。对于 SDXL 注意力层，`d=640`，`r=16`：每个适配器仅需 20k 参数，而非 410k——缩减了 20 倍。整个模型而言：LoRA 通常为 20-200MB，而基础模型为 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是常规设置。多个 LoRA 可以叠加（需注意它们会以非线性方式相互作用）。

### IP-Adapter（Ye et al., 2023）

一个微型适配器，接受*图像*作为条件（与文本并列）。使用 CLIP 图像编码器生成图像 token，与文本 token 一起注入交叉注意力。每个基础模型约 20MB。让你无需 LoRA 即可实现"生成与这张参考图风格一致的图像"。

## 可组合性矩阵

| 工具 | 控制内容 | 大小 | 使用场景 |
|------|---------|------|---------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格迁移 |
| IP-Adapter | 参考图像的风格或主体 | 20MB | 文字无法描述的外观 |
| Textual Inversion | 单个概念作为新 token | 10KB | 遗留方案，大多被 LoRA 取代 |
| DreamBooth | 针对主体的完整微调 | 2-5GB | 强身份保持，高计算成本 |
| T2I-Adapter | 更轻量的 ControlNet 替代方案 | 70MB | 边缘设备、推理预算受限 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者结合使用。

## 动手实现

`code/main.py` 在一维上模拟这两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练一个低秩 `B @ A`，使得 `W + BA` 匹配目标线性层。证明 `r = 1` 足以完美学习一个秩-1 修正。

2. **ControlNet-lite。** 一个"冻结基础"预测器和一个读取额外信号的"旁路网络"。旁路网络的输出由一个初始化为零的可学习标量门控（我们的零卷积版本）。训练并观察门控逐渐增大。

### 步骤 1：LoRA 数学

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步骤 2：零初始化旁路网络

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在步骤 0 时，输出与基础网络相同。早期训练缓慢更新 `gate`——不会出现灾难性漂移。

## 常见陷阱

- **LoRA 过度缩放。** `α = 2` 或 `α = 3` 是常见的"让它更强"的 hack，会导致过度风格化/破碎的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 姿态 ControlNet 权重 1.0 和深度 ControlNet 权重 1.0 同时使用通常会过冲。权重之和 ≈ 1.0 是安全的默认值。
- **LoRA 基础模型不匹配。** SDXL LoRA 在 SD 1.5 上会静默无操作，因为注意力维度不匹配。Diffusers 0.30+ 版本会发出警告。
- **Textual Inversion 漂移。** 在一个 checkpoint 上训练的 token 在另一个 checkpoint 上会严重漂移。LoRA 更具可移植性。
- **LoRA 权重合并与存储。** 你可以将 LoRA 烘焙到基础模型权重中以实现更快推理（无需运行时相加），但会失去运行时缩放 `α` 的能力。建议保留两个版本。

## 实际应用

| 目标 | 2026 年管线方案 |
|------|--------------|
| 复现品牌艺术风格 | LoRA，在约 30 张精选图像上训练，秩 32 |
| 将自己的脸放入生成图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + 提示 | ControlNet-Openpose + SDXL + 文本 |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考图 + 提示 | IP-Adapter + 文本 |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + Inpainting（第 09 课） |
| 快速 1 步风格化 | SDXL-Turbo 上的 LCM-LoRA |

## 交付要求

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接收一个任务（输入资产：提示、可选参考图像、可选姿态、可选深度、可选涂鸦），输出工具栈、权重和可复现的种子协议。

## 练习题

1. **简单。** 在 `code/main.py` 中，将 LoRA 秩 `r` 从 1 变到 4。在什么秩时，LoRA 能精确匹配一个秩-2 的目标增量？
2. **中等。** 在两个目标变换上分别训练两个 LoRA。将它们一起加载，展示它们的叠加交互效果。交互在什么情况下会打破线性？
3. **困难。** 使用 diffusers 堆叠：SDXL-base + Canny-ControlNet（权重 0.8）+ 风格 LoRA（α 0.8）+ IP-Adapter（权重 0.6）。测量当堆叠权重变化时，FID 与提示遵循度之间的权衡。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------|---------|
| ControlNet | "空间控制" | 克隆编码器 + 零卷积跳跃连接；读取条件图像。 |
| Zero convolution | "从恒等开始" | 1×1 卷积初始化为零；ControlNet 开始时无操作。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；参数比完整微调少 100 倍。 |
| rank r | "那个旋钮" | LoRA 压缩程度；4-16 为典型值，64+ 用于重度个性化。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放系数。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 实现的小型图像条件适配器。 |
| DreamBooth | "完整主体微调" | 在约 30 张主体图像上训练完整模型。 |
| Textual Inversion | "新 token" | 仅学习新的词嵌入；遗留方案，大多已被取代。 |

## 生产注意事项：LoRA 热切换、ControlNet 通道、多租户服务

真正的文生图 SaaS 会在同一基础 checkpoint 上服务数百个 LoRA 和十几个 ControlNet。服务问题与 LLM 多租户非常相似（生产文献在 continuous batching 和 LoRAX / S-LoRA 下讨论了 LLM 场景）：

- **热切换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础模型中，每步推理仅快 3-5%，但会冻结 `α` 和基础模型。将 LoRA 作为秩-r 增量热保留在 VRAM 中；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于按请求激活。切换成本是 `2 · d · r · num_layers` 权重——MB 级别，亚秒级。
- **ControlNet 作为第二注意力通道。** 克隆编码器与基础网络并行运行。两个 ControlNet 各权重 1.0 = 每步两次额外前向传播，而非一次合并传播。Batch-size 余量呈二次下降。每个活跃 ControlNet 预算约 1.5 倍步长成本。
- **LoRA 也量化。** 如果基础模型已量化（参见第 07 课，8GB 上的 Flux），LoRA 增量也可以干净地量化为 8-bit 或 4-bit。QLoRA 式加载让你可以在 4-bit Flux 基础模型上叠加 5-10 个 LoRA，而不会爆内存。

Flux 特注：Niels 的 Flux-on-8GB notebook 将基础模型量化为 4-bit；在该量化基础上叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`），`weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是 2026 年大多数 SaaS 机构采用的方案。

## 延伸阅读

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；后迁移至扩散模型）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的轻量替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考管线。
