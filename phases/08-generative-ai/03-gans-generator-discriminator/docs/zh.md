# GAN — 生成器与判别器

> Goodfellow 在 2014 年的诀窍是完全跳过密度估计。两个网络。一个造假。一个抓假。它们相互对抗，直到假样本与真样本无法区分。这本来不该奏效。它经常不奏效。但当它奏效时，在狭窄领域中生成的样本仍然是文献中最清晰的。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播），Phase 3 · 08（优化器），Phase 8 · 02（VAE）
**时间：** ~75 分钟

## 问题

VAE 生成的样本模糊，因为其 MSE 解码器损失对*均值*图像是贝叶斯最优的——而多个合理数字的均值就是一个模糊的数字。你需要一种奖励*合理性*而非像素级接近某个目标的损失。合理性没有闭式解，你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和伪造图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号就是 `D` 当前认为让某物看起来真实的特征。这个信号随着 `G` 的提升而更新，追逐一个移动的目标。如果两个网络都收敛，`G` 就学会了数据分布，而从未显式写出 `log p(x)`。

这就是对抗训练。其数学本质是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在 2026 年，GAN 已不再是 SOTA 生成器（扩散模型和流匹配夺走了这一桂冠）。但 StyleGAN 2/3 仍然是已部署的最清晰的人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，而对抗训练驱动着快速的 1 步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能够部署实时扩散。

## 概念

![GAN 训练：生成器与判别器的极小极大博弈](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射为样本 `x̂`。一个解码器形状的网络（全连接或转置卷积）。

**判别器 `D(x)`。** 将样本映射为标量概率（或分数）。真实 → 1，伪造 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对 real=1、fake=0 使用二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 置信时会饱和并杀死梯度）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，那么 `D` 无法做得比随机猜测更好，处处输出 0.5；`G` 不再有梯度。达到均衡。

**为什么失效。** 模式坍塌（`G` 找到一个 `D` 无法分类的模式并永远生成它）、梯度消失（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、批量大小，任何因素）。

## 让 GAN 生效的变体

| 年份 | 创新 | 修复 |
|------|------|------|
| 2015 | DCGAN | 卷积/反卷积、批归一化、LeakyReLU —— 第一个稳定架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替代 BCE。修复梯度消失。 |
| 2017 | 谱归一化 | 对判别器施加 Lipschitz 约束。2026 年的判别器仍在使用。 |
| 2018 | 渐进式 GAN | 先训练低分辨率，再添加层。首个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定域照片真实感的最高水平。 |
| 2021 | StyleGAN3 | 无混叠、平移等变 —— 2026 年仍是人脸金标准。 |
| 2022 | StyleGAN-XL | 条件化、类别感知、更大规模。 |
| 2024 | R3GAN | 以更强的正则化重新品牌；无需技巧即可在 1024² 上工作。 |

## 动手实现

`code/main.py` 在 1-D 数据上训练一个微型 GAN：两个高斯分布的混合。生成器和判别器都是单隐藏层 MLP。我们手动实现前向传播、反向传播和极小极大循环。目标是观察两种关键失效模式（模式坍塌 + 梯度消失）的实际发生。

### 步骤 1：非饱和损失

原始的 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 以高置信度将 G 的伪造样本判为伪造时会趋于 0。此时 G 的梯度基本为零 —— G 无法改进。非饱和形式 `-log D(G(z))` 具有相反的渐近特性：当 D 置信时它会爆炸，给 G 提供强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 步骤 2：每个生成器步骤对应一个判别器步骤

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

为 G 生成新的伪造样本，否则梯度会过时。

### 步骤 3：观察模式坍塌

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式之一停止被生成。判别器不再纠正它，因为它从未被当作伪造样本看到。

## 陷阱

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 的准确率超过 95%，G 就死了。
- **生成器 memorizes 一个模式。** 向 D 的输入添加噪声，使用 minibatch-discriminator 层，或切换到 WGAN-GP。
- **批归一化泄漏统计信息。** 真实批次和伪造批次流经同一个 BN 层会混合它们的统计信息。改用实例归一化或谱归一化。
- **Inception score 作弊。** FID 和 IS 在样本量低时很嘈杂。评估时使用 ≥10k 样本。
- **对于条件任务，一次性采样是谎言。** 你仍然需要 CFG scale、截断技巧和重采样才能获得可用输出。

## 使用

2026 年的 GAN 技术栈：

| 场景 | 选择 |
|------|------|
| 照片级真实人脸，固定姿态 | StyleGAN3（最清晰、最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN（Phase 8 · 04）或 ControlNet（Phase 8 · 08） |
| 快速 1 步文生图 | 扩散模型的对抗蒸馏（SDXL-Turbo、SD3-Turbo） |
| 扩散训练器内部的感知损失 | 图像块上的小 GAN 判别器 |
| 任何多模态、开放式任务 | 不要 —— 用扩散或流匹配 |

GAN 清晰但狭窄。一旦你的领域打开 —— 照片、任意文本提示、视频 —— 就切换到扩散。对抗技巧作为组件存活下来（感知损失、蒸馏），而非独立的生成器。

## 部署

保存 `outputs/skill-gan-debugger.md`。该技能接收一次失败的 GAN 运行（损失曲线、样本网格、数据集大小），输出按可能性排序的原因列表、一行修复方案和重跑协议。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重跑。G 的损失多快坍塌为常数？
2. **中等。** 将 Goodfellow BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`、`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练是否更稳定？比较实际收敛时间。
3. **困难。** 将 1-D 示例扩展到 2-D 数据（环上的 8 个高斯混合）。跟踪生成器在 1k、5k、10k 步时捕获了多少个模式。实现 minibatch discrimination 并重新测量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Generator | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| Discriminator | "D" | 分类器 `D: x → [0, 1]`，真实 vs 伪造。 |
| Minimax | "博弈" | 联合目标的 `min_G max_D`。 |
| Non-saturating loss | "那个修复" | 对 G 使用 `-log D(G(z))` 而非 `log(1 - D(G(z)))`。 |
| Mode collapse | "G 记住了一样东西" | 生成器产生很少的不同输出，尽管数据多样。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替代 BCE；梯度更平滑。 |
| Spectral norm | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个能用的" | 映射网络 + AdaIN；人脸的最佳选择，2026 年依然如此。 |

## 生产注记：一次性推理是 GAN 的持久优势

GAN 在开放域生成中不再以样本质量取胜，但它们在推理成本上仍然取胜。在生产推理文献术语中，GAN 具有：

- **无预填充、无解码阶段。** 单个 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **无 KV-cache 压力。** 唯一的状态是权重。批量大小受激活内存限制，而非缓存。
- **简单的连续批处理。** 由于每个请求消耗相同的固定 FLOPs，服务器目标占用率下的静态批次通常是最优的。不需要飞行中调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文生图的主导技术：它将 20-50 步的扩散管线压缩为 1-4 次 GAN 风格的前向传播，同时保持扩散基础的分布。对抗损失作为一种训练时旋钮存活下来，用于将慢速生成器转变为快速生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) —— 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) —— WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) —— SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) —— SDXL-Turbo。
