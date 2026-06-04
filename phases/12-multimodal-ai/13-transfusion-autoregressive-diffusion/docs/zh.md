# Transfusion：一个 Transformer 中的自回归文本与扩散图像

> Chameleon 和 Emu3 将全部赌注押在离散 token 上。它们确实有效，但量化瓶颈显而易见——图像质量在连续空间扩散模型之下趋于平缓。Transfusion（Meta，Zhou 等人，2024年8月）采取了相反的赌注：保持图像的连续性，完全放弃 VQ-VAE，并用两个损失函数训练同一个 Transformer。文本 token 采用下一个 token 预测（NTP）。图像 patch 采用流匹配/扩散损失。两个目标共同优化同一组权重。其底层架构与 Stable Diffusion 3（MMDiT）是近亲。本课将研读 Transfusion 的论文，构建一个双损失玩具训练器，并追踪让单个 Transformer 胜任两项任务的注意力掩码。

**类型：** 构建
**语言：** Python（标准库，基于 MNIST 规模的双损失玩具训练器）
**前置知识：** 第 12 阶段 · 第 11 课（Chameleon），第 8 阶段（生成式 AI）
**耗时：** 约 180 分钟

## 学习目标

- 搭建一个能在单一骨干网络上运行两种损失（文本 token 的 NTP、图像 patch 的扩散 MSE）的 Transformer。
- 解释为何跨图像 patch 的双向注意力结合文本 token 的因果注意力是正确的掩码选择。
- 从计算量、质量和代码复杂度三个维度，对比 Transfusion 风格（连续图像、扩散损失）与 Chameleon 风格（离散图像、NTP）。
- 指出 MMDiT 的贡献：每个模块使用模态特定权重，残差流中进行联合注意力计算。

## 问题背景

关于图像 token 离散与连续的争论早于大语言模型。连续表示（原始像素、VAE 潜变量）能保留细节。离散 token（VQ 索引）契合 Transformer 的原生词表，但在量化步骤中会丢失细节。

Chameleon / Emu3 选择了离散路线：单一损失、单一架构，但图像保真度受限于分词器质量。

扩散模型选择了连续路线：图像质量卓越，但与 LLM 是独立模型，需要复杂的噪声调度工程，且无法与文本生成 cleanly 集成。

Transfusion 提出疑问：我们能否兼得？保持图像连续，仍训练单一模型，将两种损失缝合进同一个梯度更新步骤中。

## 核心概念

### 双损失架构

单个仅解码器（decoder-only）Transformer 处理包含以下内容的序列：

- 文本 token（离散，来自 BPE 词表）。
- 图像 patch（连续，16x16 像素块通过线性嵌入投影到隐藏维度——与 ViT 编码器的输入方式相同）。
- `<image>` 和 `</image>` 标签，用于标记连续 patch 所在的位置。

前向传播仅执行一次。损失函数根据 token 类型选择两个头之一：

- 对于文本 token：在词表 logits 头上使用标准交叉熵。
- 对于图像 patch：在连续 patch 上使用扩散损失——预测添加到每个 patch 的噪声。

梯度流经共享的 Transformer 主体。两种损失同时优化共享权重。

### 注意力掩码：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token 关注未来的文本，否则教师强制（teacher forcing）机制会被破坏。然而，图像 patch 代表单一快照；它们应在同一图像块内相互进行双向注意力计算。

掩码设计如下：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

在训练和推理期间实现为分块三角掩码。

### Transformer 内部的扩散损失

扩散损失是标准的：向图像 patch 添加噪声，要求模型预测该噪声（或等价地预测干净 patch）。Transfusion 的版本采用流匹配（flow matching）——预测从噪声到干净数据的速度场。

训练过程：
1. 对每个图像 patch x0，采样随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（流匹配的线性插值）。
3. Transformer 预测 v_theta(xt, t)；损失 = MSE(v_theta(xt, t), ε - x0)。
4. 与该序列中文本 NTP 损失一同反向传播。

推理时，生成过程为：
- 文本 token：标准自回归采样。
- 图像 patch：扩散采样循环（通常为 10-30 步），以先前的文本 token 为条件。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser 等人，2024年3月）在与 Transfusion 大致相同的时间推出了 MMDiT（多模态扩散 Transformer）。这两种架构是同源兄弟。

MMDiT 的关键差异：

- 每个模块使用模态特定权重。每个 Transformer 模块为文本 token 和图像 patch 分别拥有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其余部分均为模态特定。
- 整流流（Rectified Flow）训练。一种特定的流匹配变体，具有已知的采样方法且数学推导比 DDPM 更简单。
- 规模。MMDiT 是 SD3 的骨干网络（提供 2B 和 8B 参数变体）。Transfusion 论文扩展至 7B。

两者都收敛于同一个核心思想：一个 Transformer 在文本上运行 NTP，在连续图像表示上运行扩散。

### 为何优于 Chameleon 风格

在图像生成任务中，连续扩散与离散 NTP 之间的质量差距是可测量的。Transfusion 论文报告称：

- 在 7B 参数下，FID 指标比同规模的 Chameleon 风格模型高出 3-5 个点。
- 无需训练分词器——图像编码器更简单（线性投影到隐藏层，与 ViT 的输入层相同）。
- 推理时可并行化图像 patch 的去噪过程，这与自回归图像 token 不同。

缺点：Transfusion 是双损失模型，使得训练动态更为复杂。损失权重需要仔细调优。NTP 与扩散之间的调度不匹配可能导致其中一个头占据主导。

### 下游演进

Janus-Pro（第 12.15 课）通过解耦理解与生成的视觉编码器来完善 Transfusion 的思想——前者使用 SigLIP，后者使用 VQ——同时共享 Transformer 主体。Show-o（第 12.14 课）用离散扩散（掩码预测）替换了连续扩散。Transfusion 之后，统一生成家族迅速分支发展。

2026 年投入生产的能生成图像的 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎肯定使用了该家族的某个衍生版本。具体细节属于商业机密。

## 实践应用

`code/main.py` 在一个极小的类 MNIST 问题上构建了一个玩具版 Transfusion：

- 文本描述是描述数字（0-9）的短整数序列。
- 图像是 4x4 的字节网格。
- 一对共享权重的线性投影充当 Transformer 的替代品；文本使用 NTP 损失，含噪 patch 使用 MSE 损失。
- 训练循环交替计算两种损失，注意力掩码显式定义。
- 生成过程在一次前向传播中输出文本描述和一张 4x4 图像。

该 Transformer 仅为玩具示例。双损失管道、注意力掩码构建以及推理循环才是真正的核心产出。

## 交付成果

本课将产出 `outputs/skill-two-loss-trainer-designer.md`。面对新的多模态训练任务（文本+图像、文本+音频、文本+视频），它能设计双损失调度方案（损失权重、掩码形状、共享模块与模态特定模块的选择），并标识实施风险。

## 练习

1. 一个 Transfusion 风格的模型训练中，70% 为文本 token，30% 为图像 patch。图像扩散损失的量级约为文本 NTP 损失的 10 倍。应设置怎样的损失权重才能使两者平衡？

2. 为一个序列实现分块三角掩码：`[T, T, <image>, P, P, P, P, </image>, T]`。请将每个条目标记为 0 或 1。

3. MMDiT 拥有模态特定的 QKV 权重。与 Transfusion 的全共享 Transformer 相比，这增加了多少参数量开销？在 7B 参数规模下，这样做值得吗？

4. 生成过程：给定一个文本提示词，模型先运行 50 个 token 的 NTP，然后触发 `<image>`，接着在 20 步去噪中对 256 个 patch 运行扩散。总共需要多少次前向传播？

5. 阅读 SD3 论文第 3 节。描述整流流（rectified flow）的原理，并说明为何它比 DDPM 在更少的推理步数内收敛。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 双损失训练 | “NTP + 扩散” | 单个 Transformer 在同一梯度步骤中同时优化文本 token 的交叉熵和连续图像 patch 的 MSE |
| 流匹配 | “整流流” | 一种扩散变体，预测从噪声到干净数据的速度场；数学推导比 DDPM 更简单 |
| MMDiT | “多模态 DiT” | Stable Diffusion 3 的架构：联合注意力、模态特定的 MLP 和归一化层 |
| 分块三角掩码 | “因果文本 + 双向图像” | 在文本范围内呈因果性，而在图像区域内呈双向性的注意力掩码 |
| 连续图像表示 | “无 VQ” | 图像 patch 作为实值向量，而非整数码本索引 |
| 速度预测 | “v 参数化” | 网络输出是噪声与数据之间的速度场，而非噪声本身 |

## 延伸阅读

- [Zhou 等人 —— Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等人 —— Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie —— DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等人 —— MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等人 —— Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
