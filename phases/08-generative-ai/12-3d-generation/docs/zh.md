# 3D 生成

> 3D 是 2D 到 3D  leverage 最强的模态。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成式推进是在此之上叠加多视角扩散 + 3D 重建，从而从单个提示词或照片生成物体和场景。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**时间：** ~45 分钟

## 问题所在

3D 内容制作很痛苦：

- **表示方式。** 网格、点云、体素网格、有符号距离场（SDFs）、神经辐射场（NeRFs）、3D 高斯。每种都有权衡。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL, 2023）有约 1000 万个物体，大多数质量低下。
- **内存。** 一个 512³ 的体素网格是 1.28 亿个体素；一个有用的场景 NeRF 需要每条光线 100 万次采样。生成比重建更难。
- **监督信号。** 对于 2D 图像，你有像素。对于 3D，你通常只有少量 2D 视图，必须提升到 3D。

2026 年的技术栈将这两个问题分开。首先，用扩散模型生成 *2D 多视角图像*。其次，将 *3D 表示*（通常是 Gaussian splatting）拟合到这些图像上。

## 核心概念

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 表示方式：3D Gaussian Splatting（Kerbl 等, 2023）

将场景表示为约 100 万个 3D 高斯的点云。每个高斯有 59 个参数：位置（3）、协方差（6，或四元数 4 + 缩放 3）、不透明度（1）、球谐颜色（3 阶 48，0 阶 3）。

渲染 = 投影 + alpha 合成。速度快（4090 上约 100 fps @ 1080p）。可微分。通过梯度下降拟合真实照片。一个场景在消费级 GPU 上 5-30 分钟拟合完成。

2023-2024 年的两项创新：
- **生成式高斯 splat。** LGM、LRM、InstantMesh 等模型从一张或几张图像直接预测高斯点云。
- **4D Gaussian Splatting。** 高斯带有每帧偏移，用于动态场景。

### 多视角扩散

微调预训练的图像扩散模型，从文本提示或单张图像生成同一物体的多个一致视角。Zero123（Liu 等, 2023）、MVDream（Shi 等, 2023）、SV3D（Stability, 2024）、CAT3D（Google, 2024）。通常输出物体周围的 4-16 个视角，通过 Gaussian splatting 或 NeRF 提升到 3D。

### 文本到 3D 流程

| 模型 | 输入 | 输出 | 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 小时/资产 |
| Magic3D | text | mesh + texture | ~40 分钟 |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 分钟 |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 分钟 |
| LRM (Meta, 2023) | image | triplane | ~5 秒 |
| InstantMesh (2024) | image | mesh | ~10 秒 |
| SV3D (Stability, 2024) | image | novel views | ~2 分钟 |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 分钟 |
| TripoSR (2024) | image | mesh | ~1 秒 |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 秒 |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 秒 |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 秒 |

2025-2026 方向：直接文本到 mesh 的模型，带有适用于游戏引擎的 PBR 材质。多视角扩散中间步骤仍然是通用物体的最佳方案。

### NeRF（背景知识）

Neural Radiance Field（Mildenhall 等, 2020）。一个微型 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿光线积分来渲染。在新视角合成质量上优于基于 mesh 的方法，但渲染速度慢 100-1000 倍。在大多数实时应用中被 Gaussian splatting 取代，但在研究中仍占主导。

## 动手实现

`code/main.py` 实现了一个玩具级的 2D "Gaussian splatting" 拟合：将合成目标图像（平滑渐变）表示为 2D 高斯 splat 的和。通过梯度下降优化位置、颜色和协方差来匹配目标。你会看到两个核心操作：前向渲染（splat + alpha 合成）和梯度下降拟合。

### 步骤 1：2D 高斯 splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 步骤 2：通过 splat 求和渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D Gaussian splatting 按深度对高斯排序并按顺序 alpha 合成。我们的 2D 玩具只是简单求和。

### 步骤 3：梯度下降拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 常见陷阱

- **视角不一致。** 如果你独立生成 4 个视角而它们对物体结构有分歧，3D 拟合会变模糊。修复：使用共享注意力的多视角扩散。
- **背面幻觉。** 单张图像 → 3D 必须想象不可见的一面。质量差异很大。
- **高斯 splat 爆炸。** 无约束训练会增长到 1000 万个 splat 并过拟合。致密化 + 剪枝启发式（来自 3D-GS 原始论文）至关重要。
- **拓扑问题。** 来自隐式场（SDFs）的 mesh 常有孔洞或自相交。发布前运行 remesher（如 blender 的 voxel remesh）。
- **训练数据许可。** Objaverse 许可混杂；商业使用因模型而异。

## 实际应用

| 任务 | 2026 年选择 |
|------|-----------|
| 从照片重建场景 | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| 游戏用文本到 3D 物体 | Meshy 4 或 Rodin Gen-1.5（PBR 输出） |
| 图像到 3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 少量图像新视角合成 | CAT3D, SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 虚拟形象/着装人体 | Gaussian Avatar, HUGS |
| 研究 / SOTA | 上周刚发布的任何东西 |

用于游戏或电商流程中的生产级 3D：Meshy 4 或 Rodin Gen-1.5 输出可直接导入 Unity / Unreal 的 PBR mesh。

## 交付

保存 `outputs/skill-3d-pipeline.md`。该技能接收一个 3D 需求（输入：文本/单张图像/少量图像；输出：mesh/splat/NeRF；用途：渲染/游戏/VR）并输出：流程（多视角扩散 + 拟合，或直接 mesh 模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## 练习

1. **简单。** 运行 `code/main.py`，分别使用 4、16、64 个高斯。报告与目标的最终 MSE。
2. **中等。** 扩展到彩色高斯（RGB）。确认重建与目标颜色模式匹配。
3. **困难。** 使用 gsplat 或 Nerfstudio，从 50 张照片捕获重建真实物体。报告拟合时间和保留视角的最终 SSIM。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 场景表示为 3D 高斯点云；可微分 alpha 合成渲染。 |
| NeRF | "Neural radiance field" | MLP 输出 3D 点的颜色 + 密度；通过光线积分渲染。 |
| Triplane | "Three 2-D planes" | 将 3D 分解为三个 2D 轴对齐特征网格；比体素计算更便宜。 |
| SDS | "Score distillation sampling" | 使用 2D 扩散分数作为伪梯度来训练 3D 模型。 |
| Multi-view diffusion | "Many views at once" | 输出一批一致相机视角的扩散模型。 |
| PBR | "Physically-based rendering" | 带有 albedo、roughness、metallic、normal 通道的材质。 |
| Densification | "Grow splats" | 3DGS 训练启发式：在高梯度区域分裂/克隆 splat。 |

## 生产备注：3D 尚未有共享基础架构

与图像（latent diffusion + DiT）和视频（时空 DiT）不同，3D 在 2026 年还没有单一的 dominant runtime。生产决策树因表示方式而分叉：

- **NeRF / triplane。** 推理是光线步进 + 每个样本的 MLP 前向。512² 渲染需要数百万次 MLP 前向。积极批量化光线样本；SDPA/xformers 适用。
- **多视角扩散 + LRM 重建。** 两阶段流程。阶段 1（多视角 DiT）是与 Lesson 07 类似的扩散服务器。阶段 2（LRM transformer）是对视角的一次性前向传播。整体延迟特征是"扩散 + 一次性"——按阶段选择相应的 serving 原语。
- **SDS / DreamFusion。** 每资产优化，不是推理。构建作业，不是请求处理器。

对于 2026 年大多数产品，正确答案是"按需运行多视角扩散模型，异步重建到 3DGS，实时 serving 3DGS"。这将在 GPU 推理服务器（快）和离线优化器（慢）之间清晰分割工作负载。

## 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
