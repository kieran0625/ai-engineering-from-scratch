# Janus-Pro：解耦编码器用于统一多模态模型

> 统一多模态模型存在不可避免的张力。理解任务需要语义特征——富含概念级信息的 SigLIP 或 DINOv2 输出向量。生成任务需要利于重建的编码——能还原为清晰像素的 VQ token。这两个目标在单个编码器中并不兼容。Janus（DeepSeek，2024年10月）和 Janus-Pro（DeepSeek，2025年1月）认为解决方案是放弃强行统一：将两个编码器解耦。在任务间共享 Transformer 主体，但让理解任务走 SigLIP 路径，生成任务走 VQ tokenizer 路径。在 7B 参数规模下，Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上与 LLaVA 持平。本课将解读为何双编码器方案能在单编码器失败的地方取得成功。

**类型：** 构建
**语言：** Python（标准库、双编码器路由 + 共享主体信号）
**前置要求：** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**时间：** 约 120 分钟

## 学习目标

- 解释为何单一共享编码器会妥协理解或生成的质量。
- 描述 Janus-Pro 的路由机制：理解任务输入端使用 SigLIP 特征，生成任务输入和输出端均使用 VQ token。
- 追踪使 Janus-Pro 在 Janus 未能成功之处取得突破的数据混合扩展策略。
- 对比解耦架构（Janus-Pro）、耦合连续架构（Transfusion）与耦合离散架构（Show-o）。

## 问题所在

统一模型在理解和生成任务间共享同一个 Transformer 主体。之前的尝试（Chameleon、Show-o、Transfusion）均对两个方向使用同一个视觉 tokenizer。该 tokenizer 是一种折衷方案：

- 针对重建优化（生成）：VQ-VAE 能捕捉细粒度的像素细节，但产生的 token 语义连贯性较弱。
- 针对语义优化（理解）：SigLIP 嵌入会将“猫”图像聚集在“cat” token 附近，但不支持高质量的重建。

Show-o 和 Transfusion 为此付出了代价，即在某一方向上出现明显的质量下降。Janus-Pro 提出疑问：既然任务需求不同，为何非要共用一个 tokenizer？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构将两个编码器分离：

- 理解路径。输入图像 → SigLIP-SO400m → 2层 MLP → Transformer 主体。
- 生成路径。输入图像（若基于现有图像进行条件生成）→ VQ tokenizer → token ID → Transformer 主体。
- 输出生成。Transformer 预测的图像 token → VQ decoder → 像素。

Transformer 主体是共享的。主体上游和下游的所有组件均针对特定任务设计。

输入通过提示词格式进行消歧：`<understand>` 标签路由至 SigLIP；`<generate>` 路由至 VQ。或者路由逻辑隐含在任务类型中。

### 为何有效

理解任务的损失函数接收 SigLIP 特征，这类特征经过类似 CLIP 的预训练已针对语义相似度进行了调优。由于输入特征更契合任务需求，模型在感知基准测试上的表现优于 Show-o / Transfusion。

生成任务的损失函数接收 VQ token，这类 token 已由 tokenizer 针对重建任务调优。由于 VQ 编码能干净地还原为像素，图像质量优于 Show-o。

共享的 Transformer 主体面对两种输入分布（SigLIP 和 VQ），并学习同时处理两者。其主张是：只要数据量足够大、参数量足够多，主体就能吸收这种切换带来的差异。

### 数据扩展 —— Janus 与 Janus-Pro 对比

Janus（初版，arXiv 2410.13848）引入了该解耦思想，但规模较小（1.3B 参数，数据有限）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 7B 参数（原为 1.3B）。
- 阶段 1（对齐）使用的图文对从 72M 提升至 90M。
- 阶段 2（统一）使用的数据从 26M 提升至 72M。
- 阶段 3 新增 20 万张图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上与 LLaVA 持平（60.3 vs ~58），并在 GenEval 上超越 DALL-E 3（0.80 vs 0.67）。单一开源模型，在统一能力谱系的两端均具备竞争力。

### JanusFlow —— 整流流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流（rectified flow）生成路径（连续型）。架构拆分为：理解用 SigLIP + 生成用整流流。质量上限进一步提升。整体架构仍保持“解耦编码器 + 共享主体”。

### 共享主体的职责

Transformer 主体处理统一的序列，但面对两种输入分布。其职责包括：

- 理解任务：接收 SigLIP 特征 + 文本 token → 自回归生成文本。
- 生成任务：接收文本 token +（可选的图像 VQ token）→ 自回归生成图像 VQ token。

该主体在每个模块中不包含特定模态的权重。它就是你预期在 Qwen 或 Llama 内部找到的文本风格 Transformer，外加两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以直接从预训练的 LLM 初始化。事实上，Janus-Pro 确实是从 DeepSeek-MoE-7B 初始化的。这一选择至关重要：LLM 提供了推理能力，这是从零开始训练的纯统一模型难以企及的。

### 与 InternVL-U 对比

InternVL-U（第 12.10 课）是 2026 年的后续演进版本。它融合了：

- 原生多模态预训练（InternVL3 骨干网络）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一的理解 + 生成 + 编辑能力。

InternVL-U 将 Janus-Pro 的架构选择融入到一个更大的框架中。“解耦编码器”理念现已成为大规模统一模型的默认设计。

### 局限性

解耦编码器增加了架构复杂度。需要训练两个 tokenizer，维护两条输入路径，应对两套故障模式。对于不需要生成能力的产品，Janus-Pro 属于过度设计——应选择 LLaVA 家族的理解模型。

对于不需要理解能力的产品，Janus-Pro 属于性能过剩——应选择 Stable Diffusion 3 / Flux 模型。

对于同时需要两者的产品，Janus-Pro 如今已成为参考性的开源架构。

## 实践应用

`code/main.py` 模拟了 Janus-Pro 的路由机制：

- 两个模拟编码器：类 SigLIP（输出 256 维语义向量）和类 VQ（输出整数编码）。
- 一个提示词路由器，根据任务标签选择对应的编码器。
- 一个共享主体（占位实现），无论 token 来自哪个编码器都能进行处理。
- 一个加权采样调度器，用于从阶段 1（对齐）切换到阶段 3（指令微调）。

打印 3 个示例的路由路径：图像问答（Image QA）、文生图（T2I）、图像编辑。

## 交付成果

本课将产出 `outputs/skill-decoupled-encoder-picker.md`。针对希望以前沿级别质量实现统一生成+理解的产品，该工具会推荐 Janus-Pro、JanusFlow 或 InternVL-U，并提供具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上超越了 DALL-E 3。请解释为何一个 7B 参数的开源模型能在生成任务上匹敌前沿闭源模型，却在理解任务上未能达到同等水平。

2. 实现一个路由器函数：给定提示词文本，将其分类为 `understand` 或 `generate`。你如何处理像“描述然后草绘”这样模糊的提示词？

3. JanusFlow 用整流流替换了 VQ 路径。此时 Transformer 主体输出什么内容？损失函数发生了哪些变化？

4. 提出一种第四种任务，Janus-Pro 架构可通过增加一个解耦编码器来处理。例如：图像分割（类 DINO 风格）、深度估计（类 MiDaS 风格）。

5. 阅读 Janus-Pro 论文第 4.2 节关于数据扩展的内容。与初版 Janus 相比，哪个数据阶段对文生图（T2I）质量的提升贡献最大？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 解耦编码 | “两个视觉编码器” | 按方向分离 tokenizer 或编码器：理解任务侧重语义，生成任务侧重重建 |
| 共享主体 | “一个 Transformer” | 单个 Transformer 处理任一编码器的输出；不含特定模态权重 |
| 用于理解的 SigLIP | “语义特征” | CLIP 系列视觉塔，提供丰富的概念特征但重建能力较差 |
| 用于生成的 VQ | “重建编码” | 矢量量化 token，可干净地解码还原为像素 |
| JanusFlow | “整流流变体” | 使用连续流匹配生成头替代 VQ 的 Janus-Pro |
| 路由标签 | “任务标签” | 选择输入编码器的提示词标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu 等人 — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen 等人 — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma 等人 — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong 等人 — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
