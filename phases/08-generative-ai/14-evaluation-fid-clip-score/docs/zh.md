# 评估 — FID、CLIP Score、人类偏好

> 每个生成模型排行榜都会引用 FID、CLIP score 和人类偏好竞技场的胜率。每个数字都有可以被有决心的研究者利用的失效模式。如果你不了解这些失效模式，就无法区分真正的改进和利用漏洞的操作。

**类型：** Build
**语言：** Python
**前置知识：** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**时间：** ~45 分钟

## 问题

生成模型的评判标准是*样本质量*和*条件遵循度*。两者都没有闭式度量。你的模型需要渲染 10,000 张图像；必须有某种方法给它们打分；你必须在不同模型家族、不同分辨率、不同架构之间信任这些数字。有三种指标在 2014-2026 年的考验中存活了下来：

- **FID (Fréchet Inception Distance)。** 两个分布之间的距离——真实图像和生成图像——在 Inception 网络的特征空间中。越低越好。
- **CLIP score。** 生成图像的 CLIP-image embedding 与提示词的 CLIP-text embedding 之间的余弦相似度。越高越好。衡量提示词遵循度。
- **人类偏好。** 对同一提示词用两个模型头对头比较，让人类（或 GPT-4 级别的模型）挑选更好的一个，聚合成 Elo 分数。

你还会看到：IS (inception score, 基本已退役)、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一个都修正了前一个的某个缺陷。

## 概念

![FID、CLIP 和偏好：三个维度，不同的失效模式](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人 (2017)。步骤：

1. 对 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2. 对每个池拟合高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯分布之间的 Fréchet 距离。越低 = 分布越相似。

失效模式：
- **小 N 偏差。** FID 是特征分布上的均方——小 N 会低估协方差，给出虚假的低 FID。始终使用 N ≥ 10,000。
- **Inception 依赖。** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的领域（人脸、艺术、文字图像）会产生无意义的 FID。使用领域特定的特征提取器。
- **利用漏洞。** 对 Inception 先验过拟合可以在没有视觉质量提升的情况下获得低 FID。用 CMMD 来对抗（见下文）。

### CLIP score — 提示词遵循度

Radford 等人 (2021)。对于生成图像 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 30k 生成图像上取平均 → 一个可在模型间比较的标量。

失效模式：
- **CLIP 自身的盲点。** CLIP 的组合推理能力较弱（"a red cube on a blue sphere" 经常失败）。模型可以在没有真正遵循复杂提示词的情况下获得高 CLIP score。
- **短提示词偏差。** 短提示词在野外有更多 CLIP-image 匹配。长提示词的 CLIP score 机械性地更低。
- **提示词利用漏洞。** 在提示词中加入 "high quality, 4k, masterpiece" 会虚高 CLIP score，而不会改善图文绑定。

CMMD (Jayasumana 等人, 2024) 修正了其中一些问题：使用 CLIP 特征替代 Inception，使用最大均值差异（maximum-mean discrepancy）替代 Fréchet。在检测细微质量差异方面更好。

### 人类偏好 —  ground truth

选取一组提示词池。用模型 A 和模型 B 生成。将成对结果展示给人类（或强大的 LLM 评判者）。将胜场聚合成 Elo 或 Bradley-Terry 分数。基准测试：

- **PartiPrompts (Google)**：1,600 个多样化提示词，12 个类别。
- **HPSv2**：107k 人类标注，广泛用作自动化代理。
- **ImageReward**：137k 提示词-图像偏好对，MIT 许可证。
- **PickScore**：在 Pick-a-Pic 的 2.6M 偏好上训练。
- **类 Chatbot-Arena 图像竞技场**：https://imagearena.ai/ 等。

失效模式：
- **评判者方差。** 非专家与专家的偏好不同。两者都要使用。
- **提示词分布。** 精心挑选的提示词会偏袒某个家族。必须始终记录。
- **LLM-judge 奖励黑客。** GPT-4-judge 会被漂亮但错误的输出欺骗。与人类评判三角验证。

## 联合使用

生产环境评估报告应包含：

1. FID 在 10-30k 样本上与 held-out 真实分布的对比（样本质量）。
2. CLIP score / CMMD 在相同样本上与对应提示词的对比（遵循度）。
3. 与上一版本模型在盲测竞技场中的胜率（整体偏好）。
4. 失效模式分析：50 个随机采样输出，标记已知问题（手部解剖、文字渲染、一致的对象数量）。

任何单一指标都是谎言。三个相互印证的指标 + 定性审查才能构成一个主张。

## 动手实现

`code/main.py` 实现了 FID、类 CLIP-score 和 Elo 聚合，使用合成"特征向量"（我们用 4 维向量作为 Inception 特征的替代）。你可以看到：

- 小 N 和大 N 下的 FID 计算 —— 偏差。
- 作为特征池之间余弦相似度的 "CLIP score"。
- 来自合成偏好流的 Elo 更新规则。

### 步骤 1：四行代码实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤 2：类 CLIP 余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤 3：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 时的 FID。** 启发式方法在 N=10k 以下不可靠。报告低 N FID 的论文是在利用漏洞。
- **跨分辨率比较 FID。** Inception 的 299×299 resize 会改变特征分布。仅在匹配分辨率下比较。
- **报告单一 seed。** 至少运行 3 个 seed。报告标准差。
- **通过负提示词虚高 CLIP score。** 某些 pipeline 通过过拟合提示词来提升 CLIP。检查视觉饱和度。
- **提示词重叠导致的 Elo 偏差。** 如果两个模型在训练期间都见过某个基准提示词，Elo 就失去意义。使用 held-out 提示词集。
- **人类评估的付费众包偏差。** Prolific、MTurk 标注者偏向年轻 / 科技友好人群。与招募的艺术/设计专家混合使用。

## 应用

2026 年生产环境评估协议：

| 支柱 | 最低要求 | 推荐 |
|------|---------|------|
| 样本质量 | FID on 10k vs held-out real | + CMMD on 5k + 按类别子集的 FID |
| 提示词遵循度 | CLIP score on 30k | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好 | 200 对盲测 vs baseline | + 2000 对人类 + LLM-judge + Chatbot Arena |
| 失效分析 | 50 手动标记 | 500 手动标记 + 自动化安全分类器 |

报告中包含四个支柱 = 主张。只有其中一个 = 营销。

## 交付

保存 `outputs/skill-eval-report.md`。该技能接受一个新模型检查点 + baseline，输出完整评估计划：样本量、指标、失效模式探针、签核标准。

## 练习

1. **简单。** 运行 `code/main.py`。在同一合成分布上比较 N=100 和 N=1000 的 FID。报告偏差幅度。
2. **中等。** 从合成 CLIP 风格特征实现 CMMD（参见 Jayasumana 等人, 2024 的公式）。与 FID 相比，对质量差异的敏感度如何。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 的子集中取 1000 个图像-提示词对，在偏好上微调一个小型 CLIP-based 评分器，测量其与 held-out 集的一致性。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| FID | "Fréchet Inception Distance" | 真实 vs 生成 Inception 特征高斯拟合的 Fréchet 距离。 |
| CLIP score | "Text-image similarity" | CLIP image 和 text embedding 之间的余弦相似度。 |
| CMMD | "FID's replacement" | CLIP-feature MMD；偏差更小，无高斯假设。 |
| IS | "Inception score" | Exp KL(p(y|x) || p(y))；与现代模型相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "Learned preference proxies" | 在人类偏好上训练的小型模型；用作自动评判。 |
| Elo | "Chess rating" | 成对胜场的 Bradley-Terry 聚合。 |
| PartiPrompts | "The benchmark prompt set" | Google 策划的 1,600 个提示词，跨 12 个类别。 |
| FD-DINO | "Self-sup replacement" | 使用 DINOv2 特征的 FD；对 ImageNet 外领域更好。 |

## 生产备注：评估也是一种推理工作负载

对 10k 样本运行 FID 意味着生成 10k 张图像。对于在单张 L4 上 1024² 分辨率的 50 步 SDXL base，这大约是 11 小时的单请求推理。评估预算是真实的，其场景正是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **大力批处理，忘掉延迟。** 离线评估 = 静态批处理，使用内存能容纳的最大 batch size。在 80GB H100 上使用 `pipe(...).images` 和 `num_images_per_prompt=8`，wall-clock 比单请求快 4-6 倍。
- **缓存真实特征。** 对真实参考集的 Inception (FID) 或 CLIP (CLIP-score, CMMD) 特征提取只运行*一次*，存储为 `.npz`。每次评估不要重新计算。

对于 CI / 回归门控：每个 PR 在 500 样本子集上运行 FID + CLIP score（~30 分钟）；每晚运行完整 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式综述。
