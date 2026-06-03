# 视频理解 — 时序建模

> 视频是一系列图像加上连接它们的物理规律。每个视频模型要么将时间视为一个额外的维度（3D 卷积），要么将其视为需要关注的序列（transformer），要么将其视为一次性提取并池化的特征（2D+池化）。

**类型：** 学习 + 构建
**语言：** Python
**前置知识：** 阶段 4 第 03 课（CNN）、阶段 4 第 04 课（图像分类）
**时间：** ~45 分钟

## 学习目标

- 区分三种主要的视频建模方法（2D+池化、3D 卷积、时空 transformer），并预测它们的成本与精度权衡
- 在 PyTorch 中实现帧采样、时序池化和 2D+池化基线分类器
- 解释为什么 I3D 的"膨胀"3D 核能从 ImageNet 权重中良好迁移，以及分解式 (2+1)D 卷积的做法有何不同
- 了解标准动作识别数据集和指标：Kinetics-400/600、UCF101、Something-Something V2；片段级别和视频级别的 top-1 准确率

## 问题

一段 30 秒、30 fps 的视频包含 900 张图像。朴素地，视频分类就是运行 900 次图像分类，然后进行某种聚合。当动作在几乎每一帧中都可见时（体育、烹饪、健身视频），这种方法有效；但当动作本身由运动定义时，它会严重失败："把某物从左推到右"在每一帧中看起来都是两个静止的物体。

每个视频架构的核心问题是：时序结构何时被建模，以及如何建模？答案决定了一切——计算成本、预训练策略、是否能复用 ImageNet 权重、模型在哪些数据集上训练。

本课刻意比静态图像课更短。核心图像机制已经就位，视频理解主要关乎时序故事：采样、建模和聚合。

## 概念

### 三大架构家族

```mermaid
flowchart LR
    V["Video clip<br/>(T frames)"] --> A1["2D + pool<br/>run 2D CNN per frame,<br/>average over time"]
    V --> A2["3D conv<br/>convolve over<br/>T x H x W"]
    V --> A3["Spatio-temporal<br/>transformer<br/>attention over<br/>(t, h, w) tokens"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a
```

### 2D + 池化

取一个 2D CNN（ResNet、EfficientNet、ViT）。在每一帧采样帧上独立运行。对每帧嵌入进行平均（或最大池化、或注意力池化）。将池化后的向量送入分类器。

优点：
- ImageNet 预训练直接迁移。
- 实现最简单。
- 成本低：T 帧 × 单张图像推理成本。

缺点：
- 无法建模运动。动作 = 外观的聚合。
- 时序池化是顺序无关的；"开门"和"关门"看起来一样。

适用场景：外观主导的任务、小数据集上的迁移学习、初始基线。

### 3D 卷积

将 2D (H, W) 核替换为 3D (T, H, W) 核。网络在空间和时间上同时卷积。早期代表：C3D、I3D、SlowFast。

I3D 技巧：取一个预训练的 2D ImageNet 模型，沿新的时间轴复制每个 2D 核来"膨胀"它。3×3 的 2D 卷积变成 3×3×3 的 3D 卷积。这让 3D 模型获得强大的预训练权重，而非从头训练。

优点：
- 直接建模运动。
- I3D 膨胀提供免费的迁移学习。

缺点：
- 比 2D 对应物多 T/8 的 FLOPs（对于时间核为 3 堆叠 3 次的情况）。
- 时间核较小；长程运动需要金字塔或双流方法。

适用场景：运动为信号的动作识别（Something-Something V2、Kinetics 中运动重的类别）。

### 时空 Transformer

将视频 token 化为时空 patch 网格，并在所有 patch 上进行注意力计算。TimeSformer、ViViT、Video Swin、VideoMAE。

重要的注意力模式：
- **联合（Joint）** —— 对 (t, h, w) 进行单一大型注意力。与 `T*H*W` 成二次关系；昂贵。
- **分离（Divided）** —— 每个块两次注意力：一次在时间，一次在空间。接近线性扩展。
- **分解（Factorised）** —— 时间注意力与空间注意力在块之间交替。

优点：
- 在每个主要基准上达到 SOTA 精度。
- 通过 patch 膨胀从图像 transformer（ViT）迁移。
- 通过稀疏注意力支持长上下文视频。

缺点：
- 计算需求高。
- 需要仔细选择注意力模式，否则运行时会膨胀。

适用场景：大数据集、高保真视频理解、多模态视频+文本任务。

### 帧采样

一段 10 秒、30 fps 的片段有 300 帧；将所有 300 帧输入任何模型都是浪费。标准策略：

- **均匀采样** —— 在片段中均匀选取 T 帧。2D+池化的默认选择。
- **密集采样** —— 随机连续 T 帧窗口。3D 卷积常用，因为运动需要相邻帧。
- **多片段** —— 从同一视频中采样多个 T 帧窗口，分别分类，测试时平均预测。

T 通常为 8、16、32 或 64。T 越大 = 更多时序信号，更多计算。

### 评估

两个级别：
- **片段级准确率** —— 模型看到一个 T 帧片段，报告 top-k。
- **视频级准确率** —— 对每段视频的多个片段预测取平均；更高更稳定。

始终报告两者。片段 78% / 视频 82% 的模型严重依赖测试时平均；片段 80% / 视频 81% 的模型每片段更鲁棒。

### 你会遇到的数据集

- **Kinetics-400 / 600 / 700** —— 通用动作数据集。40 万片段；YouTube URL（很多已失效）。
- **Something-Something V2** —— 运动定义的动作（"把 X 从左移到右"）。2D+池化无法解决。
- **UCF-101**、**HMDB-51** —— 更老、更小，仍在报告中出现。
- **AVA** —— 空间和时间的动作*定位*；比分类更难。

## 动手实现

### 步骤 1：帧采样器

在帧列表（或视频张量）上工作的均匀和密集采样器。

```python
import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))
```

两者返回 `T` 索引，用于切片视频张量。

### 步骤 2：2D+池化基线

在每帧上运行 2D ResNet-18，平均池化特征，分类。

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # global avg pool kept
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
```

一千一百万参数，ImageNet 预训练，逐帧运行，平均，分类。这个基线在外观主导的任务上通常比真正的 3D 模型低 5-10 个百分点——有时更好，因为它复用了更强的 ImageNet 骨干。

### 步骤 3：I3D 风格的膨胀 3D 卷积

通过沿新的时间轴重复权重，将单个 2D 卷积变为 3D 卷积。

```python
def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")
```

除以 `time_kernel` 保持激活幅度大致恒定——这对首次通过时不破坏批归一化统计很重要。

### 步骤 4：分解式 (2+1)D 卷积

将 3D 卷积拆分为 2D（空间）和 1D（时间）卷积。相同的感受野，更少的参数，某些基准上更好的准确率。

```python
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")
```

完整的 R(2+1)D 网络等同于将 ResNet-18 中每个 3×3 卷积替换为 `Conv2Plus1D`。

## 应用

两个库覆盖生产视频工作：

- `torchvision.models.video` —— R(2+1)D、MViT、Swin3D 及 Kinetics 预训练权重。与图像模型相同的 API。
- `pytorchvideo`（Meta）—— 模型库、Kinetics / SSv2 / AVA 的数据加载器、标准变换。

对于视觉-语言视频模型（视频描述、视频问答），使用 `transformers`（`VideoMAE`、`VideoLLaMA`、`InternVideo`）。

## 交付

本课产出：

- `outputs/prompt-video-architecture-picker.md` —— 一个根据外观 vs 运动、数据集大小和计算预算选择 2D+池化 / I3D / (2+1)D / transformer 的提示词。
- `outputs/skill-frame-sampler-auditor.md` —— 一个检查视频管道采样器并标记常见错误的技能：索引差一、当 `num_frames < T` 时采样不均、缺少保持长宽比的裁剪等。

## 练习

1. **（简单）** 计算 FramePool（T=8）与 I3D 风格 3D ResNet（T=8）的 FLOPs（近似）。论证为什么 2D+池化便宜 3-5 倍。
2. **（中等）** 生成合成视频数据集：随机球向随机方向移动，按运动方向标注（"左到右"、"右到左"、"对角向上"）。在其上训练 FramePool。证明它接近随机准确率，说明外观 alone 对运动任务不足。
3. **（困难）** 通过将 ResNet-18 中每个 Conv2d 替换为 `Conv2Plus1D` 构建 R(2+1)D-18。从 ImageNet 预训练 ResNet-18 膨胀第一个卷积的权重。在练习 2 的运动数据集上训练并击败 FramePool。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 2D + 池化 | "逐帧分类器" | 在每张采样帧上运行 2D CNN，跨时间平均池化特征，分类 |
| 3D 卷积 | "时空核" | 在 (T, H, W) 上卷积的核；能原生建模运动 |
| 膨胀 | "将 2D 权重提升到 3D" | 通过沿新时间轴重复 2D 卷积权重来初始化 3D 卷积权重，然后除以 kernel_T 以保持激活尺度 |
| (2+1)D | "分解卷积" | 将 3D 拆分为 2D 空间 + 1D 时间；参数更少，中间额外非线性 |
| 分离注意力 | "先时间后空间" | 每层有两个注意力的 transformer 块：一个在同一帧的 token 上，一个在同一位置的 token 上 |
| 片段 | "T 帧窗口" | 采样的 T 帧子序列；视频模型消耗的单位 |
| 片段 vs 视频准确率 | "两种评估设置" | 片段 = 每视频一个样本，视频 = 多个采样片段的平均 |
| Kinetics | "视频的 ImageNet" | 400-700 个动作类别，30 万+ YouTube 片段，标准视频预训练语料库 |

## 延伸阅读

- [I3D: Quo Vadis, Action Recognition (Carreira & Zisserman, 2017)](https://arxiv.org/abs/1705.07750) —— 引入膨胀和 Kinetics 数据集
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)](https://arxiv.org/abs/1711.11248) —— 分解卷积，仍是强基线
- [TimeSformer: Is Space-Time Attention All You Need? (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) —— 第一个强大的视频 transformer
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) —— 视频的掩码自编码器预训练；当前主导的预训练方案
