# 视频生成

> 图像是二维张量。视频是三维的。理论相同；计算量要难上 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明了这是可能的。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 已能从文本生成 1080p 的生产级视频——而开放权重栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）仅落后 12 个月。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 8 · 07（潜在扩散）、Phase 7 · 09（ViT）、Phase 8 · 06（DDPM）
**时间：** ~45 分钟

## 问题

一段 10 秒 1080p、24fps 的视频是 240 帧 1920×1080×3 像素。每段原始数据约 1.5 GB。像素空间扩散不可行。你需要：

1. **时空压缩。** 一个 VAE，将视频（而非单帧）编码为时空 patch 序列。
2. **时间一致性。** 帧与帧之间需要在数秒内保持内容、光照和物体身份一致。网络必须建模运动。
3. **计算预算。** 相同模型尺寸下，视频训练比图像训练贵 10-100 倍。
4. **条件控制。** 文本、图像（首帧）、音频或另一段视频。大多数生产模型接受全部四种。

解决这一问题的架构是将**扩散 Transformer（DiT）**应用于时空 patch，在庞大的（提示词、字幕、视频）数据集上训练。与第 06 课使用相同的扩散损失。

## 概念

![视频扩散：patchify、DiT、解码](../assets/video-generation.svg)

### Patchify

用 3D VAE（学习得到的时空压缩）编码视频。潜在表示的形状为 `[T_latent, H_latent, W_latent, C_latent]`。分割为大小为 `[t_p, h_p, w_p]` 的 patch。对于 Sora 风格的模型，`t_p = 1`（每帧 patch）或 `t_p = 2`（每两帧）。一段 10 秒 1080p 视频压缩后约为 20,000-100,000 个 patch。

### 时空 DiT

Transformer 处理扁平化的 patch 序列。每个 patch 具有 3D 位置嵌入（时间 + y + x）。注意力通常被分解：

- **空间注意力：** 在每帧的 patch 内部。
- **时间注意力：** 跨帧，在相同空间位置。
- **完整 3D 注意力：** 贵 16-100 倍；仅在低分辨率或研究中使用。

### 文本条件

与大型文本编码器进行交叉注意力（Sora 使用 T5-XXL，CogVideoX-5B 使用 T5-XXL）。长提示词很重要——Sora 的训练集使用 GPT 生成的密集重标注字幕，平均每段剪辑 200 个 token。

### 训练

在时空潜在表示上进行标准扩散损失（ε 或 v 预测）。数据：网络视频 + ~1 亿段精选剪辑 + 合成文本字幕。计算量：即使小型研究运行也需要 10,000+ GPU 小时；Sora 规模需要 100,000+。

## 2026 年生产格局

| 模型 | 日期 | 最大时长 | 最大分辨率 | 开放权重？ | 特点 |
|------|------|----------|-----------|-----------|------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 否 | 首个大规模展示世界模拟器特性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 生产级 Sora，推理速度快 5 倍 |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 否 | 2025 年最高质量 + 物理效果 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频和更强的镜头控制 |
| Kling 1.5 / 2.1 (快手) | 2024-2025 | 10s | 1080p | 否 | 2025 年 Q1 最佳人体运动 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 专业视频工具集成 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 最强角色一致性 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 是 (2B, 5B) | 首个开放的 5B 规模视频模型 |
| HunyuanVideo (腾讯) | 2024-12 | 5s | 720p | 是 (13B) | 2024 年末开放 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 是 (10B) | 授权最宽松 |
| WAN 2.2 (阿里巴巴) | 2025-07 | 5s | 720p | 是 | 2025 年中旬最强开放模型 |

开放权重的追赶速度比图像领域更快：到 2026 年中，HunyuanVideo + WAN 2.2 LoRA 已支撑大多数开源工作流。

## 动手构建

`code/main.py` 模拟核心时空 DiT 思想：将小型合成视频 patchify，为每个 patch 添加位置嵌入，并用类 Transformer 的注意力对整个序列进行去噪。不用 numpy；纯 Python。我们展示即使在 1D 情况下，当相邻帧的 patch 共享去噪器和位置嵌入时，时间一致性也会涌现。

### 步骤 1：将合成 1-D "视频" patchify

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 步骤 2：每帧的位置嵌入

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 步骤 3：去噪器看到整个序列

不是独立地对每帧去噪，我们的小网络将所有帧的值 + 它们的位置嵌入拼接起来，联合预测所有帧的噪声。

### 步骤 4：时间一致性测试

训练后，采样一段视频。测量帧间差值。如果模型学到了时间结构，差值会比独立采样每帧时更小。

## 陷阱

- **独立逐帧采样 = 闪烁。** 如果对每帧单独运行图像扩散，输出会闪烁，因为每帧的噪声是独立的。视频扩散通过注意力或共享噪声将帧耦合起来解决这个问题。
- **朴素 3D 注意力 = OOM。** 在 10 秒 1080p 潜在表示上做完整 3D 注意力是数千亿次操作。分解为空间 + 时间。
- **数据字幕比规模更重要。** Sora 相比先前工作的主要升级是训练了约 10 倍更详细的字幕（GPT-4 重标注剪辑）。OpenAI 的技术报告对此有明确说明。
- **首帧条件。** 大多数生产模型也接受图像作为首帧。这是"图生视频"模式；训练包含这一变体。
- **物理漂移。** 长片段（>10s）会累积细微不一致。滑动窗口生成 + 关键帧锚定有帮助。

## 应用

| 用例 | 2026 年选择 |
|------|-----------|
| 最高质量文生视频，托管服务 | Veo 3 或 Sora |
| 镜头控制电影级效果 | Runway Gen-3 配合 motion brush |
| 跨片段角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开放权重，快速微调 | WAN 2.2 + LoRA |
| 图生视频 | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音视频口型同步 | Veo 3（原生音频）或专用口型同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

2024 至 2026 年间，同等质量下每秒视频的成本下降了 20 倍。

## 交付

保存 `outputs/skill-video-brief.md`。该技能接收视频简报（时长、宽高比、风格、镜头计划、主体一致性、音频）并输出：模型 + 托管方案、提示词脚手架（镜头语言、主体描述、运动描述符）、种子 + 可复现协议，以及帧级 QA 检查清单。

## 练习

1. **简单。** 在 `code/main.py` 中，比较 (a) 独立逐帧采样 和 (b) 联合序列采样 的帧间差值。报告差值的均值和方差。
2. **中等。** 添加首帧条件：将第 0 帧固定为给定值，采样其余部分。测量固定值如何传播。
3. **困难。** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对 6 秒 720p 片段计时 20 步推理。分析时空注意力以识别瓶颈。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Video VAE | "3-D VAE" | 将 `(T, H, W, C)` 压缩为时空潜在表示的编码器。 |
| Patches | "The tokens" | 潜在表示的固定大小 3-D 块；DiT 的输入。 |
| Factorized attention | "Spatial + temporal" | 先在空间上做注意力，再在时间上做；跳过完整 3-D 注意力。 |
| Image-to-video (I2V) | "Animate this photo" | 模型接收图像 + 文本，输出从该图像开始的视频。 |
| Keyframe conditioning | "Anchor frames" | 固定特定帧以控制视频走向。 |
| Motion brush | "Directional hint" | 用户在图像上绘制运动向量的 UI 输入。 |
| Re-captioning | "Dense captions" | 使用 LLM 为训练剪辑重新标注详细提示词。 |
| Flicker | "Temporal artifact" | 帧间不一致；通过耦合去噪解决。 |

## 生产注记：视频潜在表示是内存带宽问题

10 秒 1080p、24fps 的片段是 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 原始像素。经过 4× 视频 VAE 压缩（`2 × spatial × 2 × temporal`）后，潜在表示约为每次请求 100 MB。在 batch 1 下用时空 DiT 跑 30 步，每步要在 HBM 中传输约 3 GB——瓶颈是内存带宽，不是 FLOPs。

三个生产调参旋钮，全部来自生产推理文献的推理章节：

- **DiT 上的 TP。** 文生视频模型通常 ≥10B 参数。4 张 H100 上 TP=4 是标准配置；405B 级别模型用 PP=2 × TP=2。每步延迟随 TP 近似线性下降，直到 all-reduce 墙。
- **帧 batching = 连续批处理。** 生成时，视频概念上是由注意力链接的帧 batch。连续批处理（飞行中调度）适用：如果模型架构允许滑动窗口生成，在返回帧 `t-1` 的同时开始渲染帧 `t+1`。
- **片段级预填充缓存。** 对于图生视频，首帧条件类似于 LLM 的提示词预填充：计算一次，在时序解码遍次中复用。这实际上是视频的 KV 缓存。

## 延伸阅读

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中开放 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 开创性视频扩散论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身。
