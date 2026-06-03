# 近端策略优化（PPO）

> A2C 在每次更新后就丢弃 rollout。PPO 将策略梯度包裹在一个带裁剪的重要性比率中，因此你可以对相同数据进行 10 轮以上的训练，而不会导致策略爆炸。Schulman 等人（2017）。到 2026 年仍是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 06（REINFORCE），阶段 9 · 07（Actor-Critic）
**时间：** ~75 分钟

## 问题所在

A2C（第 07 课）是 on-policy 的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。进行一次更新后，`π_θ` 就会改变；你使用的数据现在变成了 off-policy。重复使用它，你的梯度就会有偏。

Rollout 代价高昂。在 Atari 上，一次 rollout 跨越 8 个环境 × 128 步 = 1024 个转移，需要十几秒的环境时间。一次梯度步后就丢弃这些数据是浪费的。

信任区域策略优化（TRPO，Schulman 2015）是第一个解决方案：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上很干净，但每次更新都需要共轭梯度求解。2026 年没人运行 TRPO。

PPO（Schulman 等人，2017）用简单的裁剪目标替代了硬信任区域约束。多一行代码。每轮 rollout 训练十轮。无需共轭梯度。理论保证足够好。九年后，它仍然是从 MuJoCo 到 RLHF 所有任务的默认策略梯度算法。

## 核心概念

![PPO 裁剪替代目标：比率裁剪在 1 ± ε 处](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据时的策略之间的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的可能性是旧策略的两倍。

**裁剪替代目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图超过 `1 + ε`，裁剪会使梯度变平——不要将好的动作推得比旧概率高 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图超过 `1 - ε`（意味着我们会增加坏动作的可能性，相对于其裁剪后的减少量），裁剪会限制梯度——不要将坏动作推得低于 `-ε`。

`min` 处理另一个方向：如果比率已经向*有益*方向移动，你仍然获得梯度（不会裁剪对你不利的那一侧）。

典型的 `ε = 0.2`。将目标函数作为 `r_t` 的函数绘制：一个分段线性函数，"好的一侧"有平顶，"坏的一侧"有平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常为 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中，每个环境收集 `T` 步，共 `N × T` 个转移。
2. 计算优势（GAE），将其冻结为常数。
3. 冻结 `π_{θ_old}`，作为当前 `π_θ` 的快照。
4. 对于 `K` 轮 epoch，对于每个大小为 `(s, a, A, V_target, log π_old(a|s))` 的 minibatch：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃 rollout。返回步骤 1。

`K = 10` 和大小为 64 的 minibatch 是标准超参数组合。PPO 很鲁棒：精确数值在 ±50% 范围内通常无关紧要。

**KL 惩罚变体。** 原始论文提出了一种使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观测到的 KL 进行调整。裁剪版本成为主流；KL 变体在 RLHF 中得以保留（其中与参考策略的 KL 是你无论如何都想要的独立约束）。

## 动手实现

### 步骤 1：在 rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在 rollout 时只拍摄一次。在更新 epoch 期间不会改变。

### 步骤 2：计算 GAE 优势（第 07 课）

与 A2C 相同。在整个 batch 上进行归一化。

### 步骤 3：裁剪替代更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪 → 零梯度" 模式是 PPO 的核心。如果新策略已经在有益方向上漂移得太远，更新就会停止。

### 步骤 4：价值和熵

对 critic 目标添加标准 MSE，对 actor 添加熵奖励，与 A2C 相同。

### 步骤 5：诊断指标

每次更新时关注三个指标：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例** — 比率落在 `[1-ε, 1+ε]` 之外的样本比例。应为 `~0.1-0.3`。如果 `~0`，裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，你在过拟合 rollout → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。应随着 critic 学习而趋近于 1。

## 常见陷阱

- **裁剪系数调错。** `ε = 0.2` 是事实上的标准。降到 `0.1` 会使更新过于保守；`0.3+` 会导致不稳定。
- **epoch 过多。** `K > 20` 经常导致不稳定，因为策略偏离 `π_old` 太远。限制 epoch 数，尤其是对大型网络。
- **没有奖励归一化。** 大的奖励尺度会侵蚀裁剪范围。在计算优势前归一化奖励（运行标准差）。
- **忘记优势归一化。** 每批零均值/单位标准差归一化是标准做法。跳过它会在大多数基准上破坏 PPO。
- **学习率不衰减。** PPO 受益于线性衰减到零的学习率。恒定学习率通常更差。
- **重要性比率计算错误。** 始终使用 `exp(log_new - log_old)` 以保证数值稳定性，而不是 `new / old`。
- **梯度符号错误。** 最大化替代目标 = *最小化* `-L^{CLIP}`。符号翻转是最常见的 PPO bug。

## 应用场景

PPO 是 2026 年跨多个领域的默认 RL 算法：

| 应用场景 | PPO 变体 |
|----------|----------|
| MuJoCo / 机器人控制 | PPO 配合高斯策略，GAE(0.95) |
| Atari / 离散游戏 | PPO 配合分类策略，滚动 128 步 rollout |
| LLM 的 RLHF | PPO 配合对参考模型的 KL 惩罚，奖励来自响应末尾的 RM |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理型 LLM | GRPO（第 12 课）— 无 critic 的 PPO 变体 |
| 仅偏好数据 | DPO — PPO+KL 的闭式坍缩，无需在线采样 |

PPO 的*损失形状* — 裁剪替代目标 + 价值 + 熵 — 是 DPO、GRPO 和几乎所有 RLHF 流水线的脚手架。

## 提交代码

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，使用 `ε=0.2, K=4`。在匹配的环境步数下，与 A2C（每轮 rollout 一个 epoch）比较样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报与环境步数的关系，并跟踪每次更新的平均 KL。在这个任务上，KL 在什么 `K` 时爆炸？
3. **困难。** 将裁剪替代目标替换为自适应 KL 惩罚（如果 `β` 则加倍，如果 `KL > 2·target` 则减半，如果 `KL < target/2`）。比较最终回报、稳定性和无裁剪程度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 重要性比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；与收集数据时的策略的偏离程度。 |
| 裁剪替代目标 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有益侧超过裁剪点后梯度变平。 |
| 信任区域 | "TRPO / PPO 的意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 | "软信任区域" | PPO 替代方案：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 裁剪比例 | "裁剪触发的频率" | 诊断指标 — 应为 0.1-0.3；超出范围意味着调参不当。 |
| 多轮训练 | "数据复用" | 每轮 rollout 训练 K 轮；用方差代价换取样本效率。 |
| 类 on-policy | "大部分 on-policy" | PPO 名义上是 on-policy，但 K>1 轮安全地使用轻微 off-policy 数据。 |
| PPO-KL | "另一种 PPO" | KL 惩罚变体；用于 RLHF，其中与参考策略的 KL 已是约束条件。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原始论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每个 PPO 超参数的消融实验。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中的 PPO 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 清晰的现代 PyTorch 讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文引用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型上 PPO 的生产级配方；与第 09 课（RLHF）一起阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 项代码级优化"论文；哪些 PPO 技巧是关键的，哪些是 folklore。
