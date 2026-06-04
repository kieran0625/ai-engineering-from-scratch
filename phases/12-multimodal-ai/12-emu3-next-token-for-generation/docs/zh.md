# Emu3：基于下一个Token预测的图像与视频生成

> 北京智源人工智能研究院（BAAI）的 Emu3（Wang 等人，2024年9月）是终结扩散模型与自回归模型之争的2024年度标志性成果。该模型仅使用一个 Llama 风格的解码器-only Transformer，在统一的词汇表（包含文本、VQ 图像 Token 和 3D VQ 视频 Token）上仅通过下一个 Token 预测目标进行训练，在图像生成上超越了 SDXL，在感知任务上超越了 LLaVA-1.6。无需 CLIP 损失，无需扩散调度。推理时采用无分类器引导（Classifier-free guidance）以提升质量，但核心训练目标仍是带有教师强制（teacher forcing）的下一个 Token 预测。论文发表于《Nature》。本课将解读 Emu3 的论文——探讨为何更好的分词器加上规模扩展就足够了——并将其与扩散方法进行比较。

**类型：** 学习
**语言：** Python（标准库、3D 视频分词器数学推导 + 自回归采样器骨架）
**前置要求：** 第12阶段 · 第11课（Chameleon）
**耗时：** 约 120 分钟

## 学习目标

- 解释为何 Emu3 的单损失下一个 Token 预测目标能够奏效，尽管长期以来普遍认为图像生成需要扩散模型。
- 描述 3D 视频分词器：时空 VQ 码本的结构是怎样的，以及为什么 Patch 会跨越时间维度。
- 对比 Emu3 与 Stable Diffusion XL 在（训练算力、推理成本、质量上限）方面的差异。
- 指出同一个 Emu3 模型所承担的三种角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题背景

截至 2024 年的普遍共识：图像生成必须依赖扩散模型。其论据在于：离散的图像 Token 丢失了过多信息，难以重建细节；而自回归采样会在数千个 Token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 均采用了某种形式的扩散模型。Chameleon（第12.11课）在小规模上部分推翻了这一观点，但在质量上仍未能匹敌 SDXL。

Emu3 正面直击了这一论点。其主张为：更好的视觉分词器 + 足够的规模 + 下一个 Token 预测损失 = 在同一模型中实现超越扩散模型的图像生成能力，同时兼顾感知任务。

该观点在发表之初颇具争议。两年后的今天，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）已成为研究领域的默认路径；生产级前沿模型似乎也采用了其中的某种变体。

## 核心概念

### Emu3 分词器

关键组件在于视觉分词器。Emu3 训练了一个自定义的 IBQ 类分词器（Inverse Bottleneck Quantizer，属于 SBER-MoVQGAN 家族），每个 Token 对应 8x8 的分辨率缩减。一张 512x512 的图像在码本大小为 32768 时会被转换为 64x64 = 4096 个 Token。

这比 Chameleon 在 K=8192 时对 512x512 图像生成的 1024 个 Token 更多，但单个 Token 的成本更低（码本查找更小，编解码器更简单）。关键指标：重建 PSNR 达到 30.5 dB，与 Stable Diffusion 连续潜在空间的 32 dB 相当。

对于视频：3D VQ 分词器将时空 Patch（4x4x4 像素）编码为一个整数。一段 4 秒、8 FPS 的视频包含 32 帧；在 256x256 分辨率下，若空间缩减 4 倍、时间缩减 4 倍，则 Token 数量为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个 Token。

分词器的质量决定了性能上限。Emu3 的贡献之一正是“我们训练出了一个非常优秀的分词器”。

### 单损失训练

Emu3 仅使用一个目标函数：在文本 Token、2D 图像 Token 和 3D 视频 Token 共享的词汇表上进行下一个 Token 预测。训练过程中，权重会乘以模态特定的因子以平衡贡献，但损失函数完全相同。

混合训练数据包括：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：同理。
- 纯文本：标准 NTP。

模型从数据分布中学习何时输出图像 Token 与文本 Token。生成过程表现为模型在接收到 `<image>` 标签后开始预测图像 Token。

### 无分类器引导与温度参数

自回归图像生成在推理时结合无分类器引导（CFG）效果会显著提升。Emu3 采用了该方法：生成两次，一次使用完整提示词，一次使用空提示词，然后使用引导权重（通常为 3.0-7.0）混合两者的 logits。这与扩散模型使用的 CFG 技巧相同，只是被借用到自回归场景中。

温度参数至关重要：过高会导致伪影，过低会导致模式崩溃。Emu3 推荐的温度参数为：感知任务 1.0，图像生成 0.8。

### 三种角色，单一模型

Emu3 提供三个功能不同的 API，但底层共享同一套权重：

- Emu3-Gen：图像生成。输入文本，输出图像 Token。
- Emu3-Chat：视觉问答（VQA）与图像描述。输入图像（Token），输出文本。
- Emu3-Stage2：视频生成与视频 VQA。输入文本或视频，输出文本或视频。

无需任务特定的输出头。只需不同的提示词模板。共用同一检查点（checkpoint）。

### 基准测试

摘自 Emu3 论文（2024年9月）：

- 图像生成：在 MJHQ-30K FID 上超越 SDXL（5.4 vs 5.6），GenEval 总体得分持平（0.54 vs 0.55 — 统计无显著差异），Deep-Eval 综合评分相当。
- 图像感知：在 VQAv2 上超越 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致持平。
- 视频生成：4 秒片段的质量在 FVD 指标上与 Sora 时代公开基准模型具有竞争力。

数据并非全面领先——Emu3 在此处让出一分，在彼处追回一分——但“下一个 Token 预测足矣”这一主张在各模态间均站得住脚。

### 计算成本

Emu3 使用 70 亿参数的模型在约 3000 亿多模态 Token 上进行训练。GPU 小时数大致与 Llama-2-7B 预训练相当（在 A100 级别芯片上约为 2000-4000 GPU 年）。像 Stable Diffusion 3 这样的扩散模型训练预算相似，但需要独立的文本编码器且管道更复杂。

在推理阶段，Emu3 每张图像的生成速度慢于 SDXL：以 30 tok/s 的速度生成 4096 个图像 Token，生成一张 512x512 图像约需 2 分钟，而 SDXL 仅需 2-5 秒。推测解码（Speculative decoding）和 KV-cache 优化可以缩小差距，但无法完全消除。自回归图像生成计算密集；这是固有的权衡。

### 重要意义

Emu3 的深层贡献在于概念层面。如果下一个 Token 预测在图像生成上能随规模扩展至匹配扩散模型的水平，那么统一模型路径（单一损失、单一骨干网络、支持任意模态）将是可行的。未来的模型不再需要独立的文本编码器、独立的扩散调度器或独立的 VAE。只需一个 Transformer，每种模态一个分词器，加上规模扩展即可。

Show-o、Janus-Pro 和 InternVL-U 均在此基础上构建或提出挑战。截至 2025 年，中国实验室（如 BAAI、DeepSeek）在该方向上的发表力度大于美国实验室。

## 实践应用

`code/main.py` 构建了两个小型示例程序：

- 2D 与 3D VQ 分词器计数计算器：给定（分辨率、Patch大小、片段长度、FPS），计算图像与视频的 Token 数量。
- 带温度参数和无分类器引导的自回归图像 Token 采样器。

CFG 的实现遵循 Emu3 的方案——使用引导权重混合条件与非条件的 logits。

## 交付项目

本课的最终产出为 `outputs/skill-token-gen-cost-analyzer.md`。给定一个生成产品规格（图像或视频、目标分辨率、质量等级、延迟预算），它将计算 Token 数量、推理成本，并选择 Emu3 家族方案还是扩散方案。

## 练习

1. Emu3 在 8x8 缩减下，每张 512x512 图像生成 4096 个 Token。请计算 1024x1024 和 2048x2048 对应的等效 Token 数量。推理延迟会发生什么变化？

2. 阅读 Emu3 论文第 3.3 节关于视频分词器的内容。描述 3D VQ Patch 的形状，并说明为何是 4x4x4 而不是 8x8x1。

3. 无分类器引导权重设为 5.0 与 3.0 相比：会产生何种视觉效果？请在 `code/main.py` 中追踪相关数学推导。

4. 计算 Emu3-7B 在 3000 亿 Token 下的训练 FLOPs，并与 Stable Diffusion 3 进行比较。哪个模型的训练成本更高？

5. Emu3 在 FID 上优于 SDXL，但在 VQAv2 上不如专用 VLM。请解释为何统一损失方法在不同基准测试中相较于专家模型展现出不同的优势。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | 标准自回归损失：根据 token[0..i] 预测 token[i+1]；对已分词的任意模态均有效 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | 一类 VQ-VAE，拥有更大的码本（32768+）且重建质量优于 Chameleon |
| 3D VQ | "Spatiotemporal quantizer" | 按（时间、行、列）索引的码本；一个 Token 覆盖 4x4x4 像素立方体 |
| Classifier-free guidance | "CFG" | 使用权重 gamma 混合条件与非条件 logits；提升推理时的图像质量 |
| Unified vocabulary | "Shared tokens" | 文本 + 图像 + 视频均从同一整数空间中抽取；模型预测下一个出现的模态 |
| MJHQ-30K | "Image gen benchmark" | 包含 3 万个提示词的中立品质量基准；Emu3 在此报告 FID 分数 |

## 延伸阅读

- [Wang 等人 — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等人 — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等人 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等人 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等人 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
