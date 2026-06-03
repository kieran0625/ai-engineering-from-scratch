# Actor-Critic — A2C 与 A3C

> REINFORCE 噪声很大。添加一个学习 `V̂(s)` 的 critic，将其从 return 中减去，就得到了一个期望相同但方差远低的 advantage。这就是 actor-critic。A2C 以同步方式运行；A3C 跨线程运行。两者都是每个现代深度 RL 方法的心智模型。

**类型：** Build
**语言：** Python
**前置知识：** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**时间：** ~75 分钟

## 问题所在

原始 REINFORCE 可以工作，但它的方差很糟糕。蒙特卡洛 return `G_t` 在不同 episode 之间可能波动超过 10 倍。将该噪声乘以 `∇ log π` 再取平均，得到的梯度估计器需要数千个 episode 才能让策略移动与少量 DQN 更新相同的距离。

方差来自使用原始 return。如果减去一个基线 `b(s_t)` —— 任何关于状态的函数，包括学习得到的价值 —— 期望不变而方差下降。最佳的可行基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是 *advantage*：

`A(s, a) = G - V̂(s)`

如果某个动作产生了高于平均的 return，它就是好的；低于平均则是坏的。带有学习 critic 的 REINFORCE 就是 *actor-critic*。critic 为 actor 提供了一个低方差的老师。这是 2015 年之后每个深度策略方法的基础（A2C、A3C、PPO、SAC、IMPALA）。

## 核心概念

![Actor-critic: 策略网络加价值网络，TD 残差作为 advantage](../assets/actor-critic.svg)

**两个网络，一个联合损失：**

- **Actor** `π_θ(a | s)`：策略。用于采样动作。通过策略梯度训练。
- **Critic** `V_φ(s)`：估计从状态出发的期望 return。通过最小化 `(V_φ(s) - target)²` 训练。

**Advantage。** 两种标准形式：

- *MC advantage:* `A_t = G_t - V_φ(s_t)`。无偏，高方差。
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差远低。也称为 *TD residual* `δ_t`。

**n-step advantage。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现在 Atari 上使用 `n = 5`，在 MuJoCo 上的 PPO 使用 `n = 2048`。

**Generalized Advantage Estimation (GAE)。** Schulman 等人（2016）提出了对所有 n-step advantage 的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值 —— 调节到偏差/方差达到你想要的平衡点。

**A2C：同步 advantage actor-critic。** 在 `N` 个并行环境中收集 `T` 步。为每步计算 advantage。在组合批次上更新 actor 和 critic。重复。A3C 更简单、更可扩展的兄弟版本。

**A3C：异步 advantage actor-critic。** Mnih 等人（2016）。生成 `N` 个工作线程，每个运行一个环境。每个工人在自己的 rollout 上本地计算梯度，然后异步应用到共享参数服务器。不需要 replay buffer —— 工人通过运行不同轨迹来解相关。A3C 证明了可以在 CPU 上大规模训练。2026 年，基于 GPU 的 A2C（批量化并行环境）占主导，因为 GPU 需要大批次。

**联合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`、`c_e ~ 0.01` 是典型的起点。

## 动手实现

### 第一步：critic

用 MSE 更新的线性 critic `V_φ(s) = w · features(s)`：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中，critic 在几百个 episode 内收敛。在 Atari 上，将线性 critic 替换为共享 CNN 主干 + 价值头。

### 第二步：n-step advantage

给定长度为 `T` 的 rollout 和 bootstrap 的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 目标。`advantages` 是乘以 `∇ log π` 的量。

### 第三步：联合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每次更新一个 rollout，actor 和 critic 使用不同的学习率。

### 第四步：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境并执行自己的前向传播。定期将梯度更新推送到共享主节点。主节点不加锁 —— 竞争没关系，只是增加了噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠成 `[N, obs_dim]` 批次，批量化前向传播，批量化反向传播。更高的 GPU 利用率，确定性，更容易推理。2026 年的默认选择。

我们的示例代码为清晰起见是单线程的；改写成批量化 A2C 只需三行 numpy。

## 常见陷阱

- **Critic 先于 actor 梯度产生偏差。** 如果 critic 是随机的，其基线没有信息量，你就是在纯噪声上训练。在开启策略梯度前预热 critic 几百步，或使用较慢的 actor 学习率。
- **Advantage 归一化。** 将 advantage 按批次归一化为零均值/单位标准差。以接近零的代价极大稳定训练。
- **共享主干。** 在图像输入上为 actor 和 critic 使用共享的特征提取器。分离的头部。共享特征同时受益于两个损失。
- **On-policy 契约。** A2C 的数据恰好复用一次用于更新。更多次会导致梯度有偏（重要性采样校正正是 PPO 添加的内容）。
- **熵坍塌。** 没有 `c_e > 0`，策略在几百次更新后变得接近确定性并停止探索。
- **奖励尺度。** Advantage 的大小取决于奖励尺度。归一化奖励（例如，除以运行标准差）以获得跨任务的一致梯度幅度。

## 应用

A2C/A3C 在 2026 年很少是最终选择，但它们是后续所有方法的架构基础：

| 方法 | 与 A2C 的关系 |
|------|--------------|
| PPO | A2C + 裁剪的重要性比率，用于多 epoch 更新 |
| IMPALA | A3C + V-trace 离策略校正 |
| SAC (Phase 9 · 07) | 带有软价值 critic 的离策略 A2C（下一课） |
| GRPO (Phase 9 · 12) | 没有 critic 的 A2C —— 组相对 advantage |
| DPO | 压缩成偏好排序损失的 A2C，无需采样 |
| AlphaStar / OpenAI Five | A2C + 联盟训练 + 模仿预训练 |

如果你在 2026 年的论文中看到 "advantage"，请想到 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC advantage（`G_t - V(s_t)`）训练 actor-critic。与第 06 课的 REINFORCE-with-running-mean-baseline 比较样本效率。
2. **中等。** 切换到 TD-residual advantage（`r + γ V(s') - V(s)`）。测量 advantage 批次的方差。它下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终 return 与样本效率的关系图。对于这个任务，偏差/方差的甜蜜点在哪里？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Actor | "策略网络" | `π_θ(a\|s)`，通过策略梯度更新。 |
| Critic | "价值网络" | `V_φ(s)`，通过 MSE 回归到 return / TD 目标更新。 |
| Advantage | "比平均好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计器。`∇ log π` 的乘数。 |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`；单步 advantage 估计。 |
| GAE | "插值旋钮" | n-step advantage 的指数加权和，由 `λ` 参数化。 |
| A2C | "同步 actor-critic" | 跨环境批量化；每个 rollout 一次梯度步。 |
| A3C | "异步 actor-critic" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年较少见。 |
| Bootstrap | "在边界使用 V" | 截断 rollout，加上 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当 critic 是神经网络时，结合第 9 章关于函数逼近的内容阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 可扩展的分布式 actor-critic，带有 V-trace 离策略校正。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 双时间尺度 actor-critic 分解的基础收敛性结果。
