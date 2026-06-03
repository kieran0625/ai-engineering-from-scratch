# 自编码器与变分自编码器（VAE）

> 普通自编码器先压缩再重建。它只会记忆，不会生成。加入一个技巧——强制让编码服从高斯分布——你就得到了一个采样器。这一个技巧，即 `z = μ + σ·ε` 的重参数化，正是 2026 年你使用的每一个潜在扩散模型和流匹配图像模型都在输入端配备 VAE 的原因。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3 · 02（反向传播），Phase 3 · 07（CNN），Phase 8 · 01（分类学）
**时间：** 约 75 分钟

## 问题

将一个 784 像素的 MNIST 数字压缩为 16 维编码，再重建出来。普通自编码器在重建 MSE 上表现完美，但编码空间却一团糟。在编码空间中随机选一点解码，得到的只是噪声。它没有采样能力，只是一个伪装成生成模型的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑、可从中采样的分布——比如各向同性高斯 `N(0, I)`；(b) 解码任意采样点都能产生合理的数字；(c) 编码器和解码器仍然保持良好的压缩能力。三个目标，一个架构，一个损失函数。

Kingma 2013 年的 VAE 通过以下方式解决：训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)`，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z`。推理时，丢弃编码器，采样 `z ~ N(0, I)`，解码。正是 KL 惩罚迫使编码空间变得结构化。

2026 年，VAE 很少单独出现——在原始图像质量上它们已被扩散模型超越——但它们是每一个潜在扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了你使用的每一个图像 pipeline 中看不见的第一层。

## 概念

![自编码器与 VAE：重参数化技巧](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，loss = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样不可微。将样本重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一个非参数噪声——梯度通过 `μ` 和 `σ` 流动。

**损失。** 证据下界（ELBO），两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项将 `x̂` 推向 `x`。KL 项将 `q(z|x)` 推向先验。二者此消彼长。小的 β（<1）= 更锐利的样本，编码空间更少高斯性。大的 β（>1）= 更干净的编码空间，更模糊的样本。β-VAE（Higgins 2017）让这个旋钮名声大噪，并开启了解耦研究。

**采样。** 推理时：抽取 `z ~ N(0, I)`，前向通过解码器。一次前向传播——不像扩散那样迭代采样。

## 动手构建

`code/main.py` 实现了一个微型 VAE，不使用 numpy 或 torch。输入是从 8 维 2 成分高斯混合分布中抽取的 8 维合成数据。编码器和解码器都是单隐藏层 MLP。我们实现 tanh 激活、前向传播、损失和手写反向传播。不是生产代码——用于教学。

### 步骤 1：编码器前向

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而非 `σ`，因此网络输出无约束（σ 的 softplus 是陷阱——当 σ ≈ 0 时梯度消失）。

### 步骤 2：重参数化与解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤 3：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

KL 有精确闭式解，因为两个分布都是高斯分布。不要数值积分。2026 年仍有人发布使用蒙特卡洛 KL 估计的代码——毫无理由地慢 3 倍。

### 步骤 4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 陷阱

- **后验坍塌。** KL 项过于激进地将 `q(z|x) → N(0, I)` 推向先验，导致 `z` 不再携带关于 `x` 的信息。修复方法：β 退火（从 β=0 开始，逐渐升至 1）、自由比特，或在非活跃维度上跳过 KL。
- **模糊样本。** 高斯解码器似然意味着 MSE 重建，而 MSE 对于 L2（均值）是贝叶斯最优的——一组合理数字的均值是一个模糊的数字。修复方法：离散解码器（VQ-VAE、NVAE），或将 VAE 仅用作编码器并在潜变量上堆叠扩散模型（Stable Diffusion 的做法）。
- **β 过大、过早。** 见后验坍塌。从 β≈0.01 开始并逐渐增大。
- **潜变量维度太小。** 16 维适用于 MNIST，256 维适用于 ImageNet 256²，2048 维适用于 ImageNet 1024²。Stable Diffusion 的 VAE 将 512×512×3 压缩为 64×64×4（空间面积下采样 32 倍，通道数下采样 32 倍）。

## 应用

2026 年的 VAE 技术栈：

| 场景 | 选择 |
|------|------|
| 扩散模型的图像潜变量编码器 | Stable Diffusion VAE（`sd-vae-ft-ema`）或 Flux VAE |
| 音频潜变量编码器 | Encodec（Meta）、SoundStream 或 DAC（Descript） |
| 视频潜变量 | Sora 的时空 patch、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散潜变量（用于 transformer 建模） | VQ-VAE、RVQ（ResidualVQ） |
| 连续潜变量用于生成 | 普通 VAE，然后在该潜变量空间中条件化流/扩散模型 |

潜在扩散模型是一种在编码器和解码器之间住着扩散模型的 VAE。VAE 负责粗略压缩，扩散模型负责繁重工作。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）遵循相同模式。

## 交付

保存 `outputs/skill-vae-trainer.md`。

技能要求：数据集画像 + 潜变量维度目标 + 下游用途（重建、采样或潜在扩散输入），输出：架构选择（普通/β/VQ/RVQ）、β 调度、潜变量维度、解码器似然（高斯 vs 分类），以及评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## 练习

1. **简单。** 在 `code/main.py` 中将 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。哪个 β 对你的合成数据是帕累托最优的？
2. **中等。** 将高斯解码器似然替换为伯努利似然（交叉熵损失）。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为微型 VQ-VAE：将连续的 `z` 替换为在 K=32 条目的码本中的最近邻查找。比较重建 MSE，并报告有多少码本条目被使用（码本坍塌是真实存在的）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 自编码器 | 编码-解码网络 | `x → z → x̂`，学习 MSE。不是生成模型。 |
| VAE | 带采样器的 AE | 编码器输出分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时紧致。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点重写为确定性部分 + 纯噪声。实现通过采样的反向传播。 |
| 先验 | `p(z)` | 潜变量的目标分布，通常为 `N(0, I)`。 |
| 后验坍塌 | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器必须幻觉。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。更高的 β = 更解耦但更模糊。 |
| VQ-VAE | 离散潜变量 | 将连续的 `z` 替换为最近的码本向量；实现 transformer 建模。 |

## 生产提示：VAE 是扩散服务器中最热的路径

在 Stable Diffusion / Flux / SD3 pipeline 中，VAE 每个请求被调用两次——一次编码（如果做 img2img / 修复）和一次解码。在 1024² 时，解码器 pass 往往是整个 pipeline 中最大的激活内存峰值，因为它将 `128×128×16` 的潜变量上采样回 `1024×1024×3`。两个实际后果：

- **切片或分块解码。** `diffusers` 暴露 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块以微小的接缝伪影换取 `O(tile²)` 内存而非 `O(H·W)`。对于消费级 GPU 上的 1024²+ 至关重要。
- **解码器用 bf16，最终 resize 用 fp32。** SD 1.x VAE 以 fp32 发布，在 1024²+ 时转换为 fp16 会*静默产生 NaN*。SDXL 发布 `madebyollin/sdxl-vae-fp16-fix`——始终优先选择 fp16-fix 变体或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) —— VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) —— 解耦 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) —— VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) —— 最先进的图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— Encodec，音频 VAE 标准。
