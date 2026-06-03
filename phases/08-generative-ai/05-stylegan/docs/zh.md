# StyleGAN

> 大多数生成器在同一时间将 `z` 注入每一层。StyleGAN 将其拆分：首先将 `z` 映射到中间表示 `w`，然后通过 AdaIN 在每个分辨率层级*注入* `w`。这一单一改变解耦了潜在空间，使照片级真实感人脸生成连续七年成为已解决的问题。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**时间：** ~45 分钟

## 问题所在

DCGAN 通过转置卷积层堆栈将 `z` 映射为图像。问题在于：`z` 同时控制一切——姿态、光照、身份、背景——全部纠缠在一起。沿着 `z` 的一个维度移动，四个属性全部改变。你无法要求模型"同一个人，不同姿态"，因为表示方式并未按这种方式分解。

Karras 等人（2019，NVIDIA）提出：停止直接将 `z` 馈入卷积层。将一个恒定的 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP 来映射 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率注入 `w`：对每个卷积特征图进行归一化，然后通过 `w` 的仿射投影进行缩放和平移。添加逐层噪声以引入随机细节（皮肤毛孔、发丝）。

结果：`W` 具有大致正交的轴，分别对应"高层风格"（姿态、身份）与"精细风格"（光照、颜色）。可以通过对低分辨率层级使用图像 A 的 `w`、对高分辨率层级使用图像 B 的 `w` 来在两张图像之间交换风格。这解锁了编辑、跨域风格化以及整个"StyleGAN 反演"研究方向。

## 核心概念

![StyleGAN：映射网络 + AdaIN + 逐层噪声](../assets/stylegan.svg)

**映射网络。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不必强制为高斯分布——它学习适应数据的形状。

**合成网络。** 从一个可学习的恒定 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率逐层翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。对每个特征图进行归一化，然后重新风格化。这里的"风格"指的是特征图的一阶和二阶统计量。

**逐层噪声。** 单通道高斯噪声添加到每个特征图，按可学习的逐通道因子缩放。控制随机细节而不影响全局结构。

**截断技巧。** 推理时，采样 `z`，计算 `w = mapping(z)`，然后计算 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是大量样本上 `w` 的均值。`ψ < 1` 以多样性换取质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新点 |
|---------|------|------------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进式增长。 |
| StyleGAN2 | 2020 | 权重解调替代 AdaIN（修复液滴伪影）；跳跃/残差架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无混叠卷积 + 等变核；消除纹理粘附到像素网格。 |
| StyleGAN-XL | 2022 | 类别条件，1024²，ImageNet。 |
| R3GAN | 2024 | 以更强正则化重新品牌；在 FFHQ-1024 上以 20 倍更少参数缩小与扩散模型的差距。 |

2026 年 StyleGAN3 仍是以下场景的首选：（a）高 FPS 窄域照片级真实感，（b）少样本域适应（用 100 张图像训练新数据集，冻结映射网络），（c）基于反演的编辑（找到能重建真实照片的 `w`，然后编辑该 `w`）。对于开放域文本到图像生成，它不是合适的工具——扩散模型才是。

## 动手实现

`code/main.py` 实现了一个玩具版"style-GAN lite"，采用 1-D 形式：一个映射 MLP，一个以可学习常数向量为输入并通过 `w` 派生的缩放/偏置进行调制的合成函数，以及逐层噪声。它展示了通过仿射调制注入 `w` 能够匹配或超越将 `z` 拼接到生成器输入的方式。

### 步骤 1：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 步骤 2：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

逐特征图的缩放和偏置来自 `w` 的线性投影。

### 步骤 3：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

逐通道的 sigma 是可学习的。

## 常见陷阱

- **液滴伪影。** StyleGAN 1 在特征图中产生 blobby 液滴，因为 AdaIN 将均值归零。StyleGAN 2 的权重解调通过缩放卷积权重而非激活值来修复此问题。
- **纹理粘附。** StyleGAN 1 和 2 的纹理跟随像素坐标而非物体坐标（插值时可见）。StyleGAN 3 的无混叠卷积通过加窗 sinc 滤波器修复此问题。
- **模式覆盖。** 截断 `ψ < 0.7` 看起来干净，但仅从狭窄锥体中采样；如需多样性则使用 `ψ = 1.0`。
- **反演是有损的。** 将真实照片反演为 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。结果在多次迭代后会漂移。

## 实际应用

| 应用场景 | 方法 |
|----------|----------|
| 照片级真实人脸（动漫、产品、窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片进行人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 表情重演 | StyleGAN + 编码器 + 混合 |
| 头像流水线 | StyleGAN3 配合 ADA 进行低数据微调 |
| 少量图像的域适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件生成 | 不要用它——用扩散模型 |

对于答案为"人物面部照片"的产品级演示，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和同等质量下的清晰度方面优于扩散模型。

## 交付要求

保存 `outputs/skill-stylegan-inversion.md`。技能要求：输入真实照片，输出：反演方法（e4e / ReStyle / HyperStyle）、预期潜在损失、编辑预算（在 `W` 中可移动多远才会出现伪影），以及已知有效的编辑方向列表（年龄、表情、姿态）。

## 练习题

1. **简单。** 运行 `code/main.py`，使用 `adain_on=True` 和 `adain_on=False`。比较固定潜在变量与扰动潜在变量的输出分布。
2. **中等。** 实现混合正则化：对于训练批次，计算 `w_a`、`w_b`，前半段合成使用 `w_a`，后半段使用 `w_b`。解码器是否学习到了解耦的风格？
3. **困难。** 取预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标注样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告在身份漂移前可以推动多远。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 映射网络 | "那个 MLP" | `f: Z → W`，8 层，将潜在几何与数据统计解耦。 |
| W 空间 | "风格空间" | 映射网络的输出；大致解耦。 |
| AdaIN | "自适应实例归一化" | 归一化特征图，然后按 `w` 投影进行缩放 + 平移。 |
| 截断技巧 | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以多样性换取质量。 |
| 路径长度正则化 | "PL reg" | 惩罚 `w` 单位变化引起的图像大幅变化；使 `W` 更平滑。 |
| 权重解调 | "StyleGAN2 的修复" | 归一化卷积权重而非激活值；消除液滴伪影。 |
| 无混叠 | "StyleGAN3 的技巧" | 加窗 sinc 滤波器；消除纹理粘附到像素网格。 |
| 反演 | "为真实图像找到 w" | 优化或编码 `x → w` 使得 `G(w) ≈ x`。 |

## 生产备注：为什么 2026 年 StyleGAN 仍在服役

StyleGAN3 在 4090 上生成 1024² FFHQ 人脸耗时不到 10 毫秒——`num_steps = 1`，无需 VAE 解码，无需交叉注意力传递。在生产环境中，这是任何图像生成器的最低延迟基准。同等分辨率下，50 步 SDXL + VAE 解码流水线约需 3 秒。这是 **300 倍的差距**，对于窄域产品（头像服务、证件照流水线、库存人脸生成）而言，它在 TCO 上胜出。

两个运营层面的影响：

- **无需调度器，无需批处理器。** 目标占用率下的静态批次是最优的。连续批处理（对 LLM 和扩散模型至关重要）毫无收益，因为每个请求的 FLOPs 完全相同。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 从映射网络范围的狭窄锥体中采样。这是服务层对样本方差的唯一控制杠杆。峰值负载时降低 `ψ`，为高级用户提高它。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 现代极简 GAN 方案。
