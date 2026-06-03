# 缩放定律

> 2020 年 Kaplan 论文说：模型越大，损失越低。2022 年 Hoffmann 论文说：你们训练不足。计算量分为两个桶——参数量和 token 数——而如何分配并不明显。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer），Phase 7 · 07（GPT）
**时间：** ~45 分钟

## 问题所在

当你拥有 C FLOPs 的训练计算量，并希望获得最佳模型时，你面临两个旋钮：

1. **多少参数（N）？** 模型越大，容量越高。
2. **多少训练 token（D）？** 数据越多，容量利用越好。

FLOPs 大致按 `6 × N × D` 缩放。你可以把 N 推高、D 降低，或者把 D 推高、N 降低。哪个更好？

2022 年之前，答案是"猛推 N"。GPT-3（2020）是 175B 参数，训练了约 300B token。比例约为每个参数 1.7 个 token。Kaplan 缩放定律支持这一点。

Hoffmann 等人（2022）训练了一个名为 Chinchilla 的小型模型家族，发现了不同的结果：最优比例接近**每个参数 20 个 token**。GPT-3 的训练量低了 10 倍。Chinchilla（70B 参数，1.4T token）在每个基准测试上都击败了 GPT-3（175B，300B token），而推理成本只有 2.5 分之一。

2026 年是 Chinchilla 的世界——但有一个重要的转折。Llama 3 8B 训练了 15 万亿 token，比例为每个参数 1,875 个 token。超出 Chinchilla 最优值 94 倍。对于大规模使用的模型，推理成本比训练成本更重要，因此过度训练（超过 Chinchilla）以获得更小的可部署 footprint 是 2026 年的默认选择。

## 核心概念

![Chinchilla 曲线：不同 N/D 比例下的损失与计算量关系](../assets/scaling-laws.svg)

### Hoffmann 定律

根据 Chinchilla 论文，损失遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数（非嵌入）。
- `D` = 训练 token。
- `α ≈ 0.34`、`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约损失上限。
- `A ≈ 406`、`B ≈ 411`。

两个项在缩放时相互权衡。在固定计算量（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优：每个参数 20 个 token。

### 为什么要过度训练

Chinchilla 最优值最小化了每训练 FLOP 的训练损失。但训练成本只付一次；推理成本永远存在。

对于每月服务一万亿 token 的聊天机器人，推理主导总成本。Llama 的做法：训练更小、更久。8B 参数训练 15T token 是深度推理优化的：

- 能装进消费级 GPU。
- 延迟是 70B Chinchilla 最优模型的一小部分。
- 对大多数任务来说质量足够接近。

DeepMind 2024 年的论文（"Over-training is the new optimal"）形式化了这一点。对于推理主导的工作负载，合适的比例更接近每个参数 100–500 个 token，具体取决于服务量。

### 涌现与平滑性

主张：某些能力（算术、多步推理、思维链跟随）在某个规模下会"突然涌现"。

Schaeffer 等人（2023）认为这是测量伪影：涌现指标使用不连续评分（精确匹配、阈值准确率），掩盖了底层 logits 中平滑的改进。连续指标（交叉熵）显示平滑曲线。

2026 年的共识是：通过连续损失的预测是可靠的。基准测试的跳跃通常是评分器伪影。按连续指标规划预算。

### 2026 年的图景

缩放定律仍然有效，但：

| 因素 | 变化方式 |
|------|---------|
| 数据质量 | 筛选"好"token（Phi 风格）使有效计算量偏移 >2× |
| MoE | 总参数与活跃 FLOP 解耦；按每活跃 FLOP 的缩放定律 |
| 后训练 | 某些能力（指令遵循、代码）随 SFT+RLHF 的变化大于预训练 |
| 多模态 | 图像 + 文本 token 一起缩放；每种模态有单独的曲线 |
| 合成数据 | 模型生成训练数据；有效计算量可以复合 |

Muon 优化器（Kimi Moonlight，2024）在匹配数据下展示了相比 AdamW 约 2× 的有效计算量增益。一些 2026 年的训练运行默认使用 Muon。改变了缩放定律中的绝对常数，而非其形状。

## 动手实现

参见 `code/main.py`。我们实现 Chinchilla 损失方程，并在多个计算预算下求解计算最优的 `(N, D)`。

### 步骤 1：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

将 `L` 绘制为 `(N, D)` 在固定 `C = 6ND` 下的等高线图。找到最小值。

### 步骤 2：计算最优前沿

对于从 `1e17` 到 `1e25` FLOPs 的计算预算，找到在 `6ND = C` 约束下最小化损失的 `(N, D)`。验证比例 `D/N ≈ 20`。

### 步骤 3：过度训练成本

计算将模型缩小 10 倍（最优 N 的 1/10，最优 D 的 10 倍）时额外支付的损失。报告换取的推理 FLOP 节省（与 N 成正比）。

### 步骤 4：与真实模型对比

填入已知的 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（活跃参数）的 `(N, D)` 对，比较预测损失与实际报告损失。

## 实际应用

你自己不太可能训练前沿模型。但缩放定律告诉你：

1. **你的微调是否有足够数据。** 如果你的任务特定数据低于基础模型每参数 20 个 token，预计会在某个损失下限饱和。
2. **是否应该选择更大的基础模型。** 如果你把所有预算都花在推理上，优先选择更小、训练更久的模型。
3. **收益何时递减。** 超过 Chinchilla 最优值 1000 倍后，对数损失的变化会变成噪声。

**2026 年的研究轨迹：**

- **数据受限 regime。** 网络中高质量 token 的数量有限（过滤后约 5–10 万亿英语 token）。前沿预训练正接近这一上限。合成数据、多语言、多模态和 RLHF 放大的微调是下一个杠杆。
- **计算量倍增技巧。** Muon 优化器、MoE、更好的数据筛选——每个都改变绝对常数，而非渐近线。
- **RL 的缩放定律。** 开放问题。早期证据表明 RL 样本存在幂律，但指数与预训练非常不同。

## 交付

参见 `outputs/skill-training-budget-estimator.md`。该技能在给定计算预算、部署约束和目标损失的情况下，为新的训练运行选择 `(N, D, hours, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印计算预算为 `1e20`、`1e22`、`1e24` 时的 Chinchilla 最优 `(N, D)`。与真实模型表对比。
2. **中等。** 实现 Hoffmann 损失关于计算量的函数。绘制计算最优前沿的损失与 `log10(C)` 的关系。确定定律预测我们需要 `>10^28` FLOPs 才能将交叉降低 0.1 的时间点。
3. **困难。** 在 5 个微型模型（100K 到 10M 参数）上拟合你自己的缩放定律，使用相同数据集训练。估计 `α` 和 `E`。你的指数与发表的指数匹配程度如何？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 参数（N） | "模型大小" | 非嵌入权重数量；决定容量。 |
| Token（D） | "训练数据" | 见过的训练 token 数量；决定参数利用程度。 |
| 计算量（C） | "花费的 FLOPs" | 标准 transformer 大致为 `6 × N × D`。 |
| Chinchilla 最优 | "D/N ≈ 20" | 最小化预训练每 FLOP 损失的比率。 |
| 过度训练 | "超过 Chinchilla" | 花费额外训练 FLOPs 以节省推理 FLOPs；D/N >> 20。 |
| 不可约损失 | "地板" | 缩放定律中的 `E` 项；数据本身的熵。 |
| 涌现能力 | "规模上的突然跳跃" | 通常是评分器伪影；连续损失是平滑的。 |
| 有效计算量 | "训练效率乘数" | 更好的数据 / 优化器 / 架构放大每个 FLOP 的效果。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇缩放定律论文；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 涌现作为测量伪影。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的过度训练对其工作负载是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× 计算量乘数。
