# 策略梯度 —— 从零实现 REINFORCE

> 别再估计价值了。直接参数化策略，计算期望回报的梯度，向高处迈步。Williams (1992) 用一个定理就写清楚了。PPO、GRPO 以及所有 LLM 强化学习循环之所以存在，就是因为这个定理。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3 · 03（反向传播）、Phase 9 · 03（蒙特卡洛）、Phase 9 · 04（时序差分学习）
**时间：** ~75 分钟

## 问题

Q-learning 和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 来选择动作。对于离散动作和离散状态，这没问题。但当动作是连续的（在 10 维扭矩上 `argmax`？）或者你想要随机策略时（`argmax` 按构造就是确定性的），它就失效了。

策略梯度改为参数化*策略*。`π_θ(a | s)` 是一个输出动作分布的神经网络。从中采样来执行动作。计算期望回报关于 `θ` 的梯度。向高处迈步。没有 `argmax`。没有贝尔曼递归。就是对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合。计算回报。在每一步乘以 `∇ log π_θ(a | s)`。取平均。梯度上升。完成。

2026 年的每一个 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进版。亲手理解它是本阶段其余内容的前提，也是 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的前提。

## 概念

![策略梯度：softmax 策略、log-π 梯度、回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任何由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从第 `t` 步开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ` 而言的。

**证明很短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。分解出 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`。环境项消失。两行代数就给你这个定理。

**方差削减技巧。** 原始 REINFORCE 的方差大得惊人——回报有噪声，`∇ log π` 有噪声，它们的乘积噪声更大。两个标准修正：

1. **基线减法。** 将 `G_t` 替换为 `G_t - b(s_t)`，其中基线 `b(s_t)` 不依赖于 `a_t`。无偏因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：由 critic 学习的 `b(s_t) = V̂(s_t)` → actor-critic（第 07 课）。
2. **未来回报（reward-to-go）。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。只有未来回报对给定动作重要——过去的回报贡献零均值噪声。

结合起来，你得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（第 07 课）和 PPO（第 08 课）的直接祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是输出每个动作分数的任意神经网络。梯度有简洁形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即，所采取动作的分数减去其在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。这就是 Phase 9 · 07 的 SAC 所需的全部。

## 动手实现

### 步骤 1：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格环境，使用线性策略（每个动作一个权重向量）。对于 Atari，换入 CNN 并保留 softmax 头部。

### 步骤 2：采样和对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 步骤 3：记录对数概率的回合推演

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 步骤 4：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 one-hot 减去概率）是 softmax 策略梯度的核心。把它刻进肌肉记忆。

### 步骤 5：基线

对最近回合的 `G` 取滑动平均，就足以将方差削减到让 4×4 GridWorld 跑起来；大约 500 个回合收敛。将基线升级为学习的 `V̂(s)`，你就得到了 actor-critic。

## 陷阱

- **梯度爆炸。** 回报可能巨大。总是在乘以 `∇ log π` 之前，将 `G` 在批次上归一化为 `~N(0, 1)`。
- **熵坍塌。** 策略过早收敛到近确定性动作，停止探索，陷入停滞。修复：向目标添加熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个回合。critic 基线（第 07 课）或 TRPO/PPO 的信任域（第 08 课）是标准修复。
- **样本效率低。** 同策略意味着每次更新后你就丢弃每个转移。通过重要性采样进行离策略修正可以回收数据，代价是方差（PPO 的比率就是裁剪后的 IS 权重）。
- **非平稳梯度。** 100 个回合前的梯度使用的是旧的 `π`。同策略方法因此每几个回合就更新一次。
- **信用分配。** 没有未来回报时，过去的回报贡献噪声。始终使用未来回报。

## 应用

2026 年，REINFORCE 很少直接运行，但它的梯度公式无处不在：

| 应用场景 | 衍生方法 |
|----------|---------|
| 连续控制 | PPO / SAC 配合高斯策略 |
| LLM RLHF | 带 KL 惩罚的 PPO，在 token 级策略上运行 |
| LLM 推理（DeepSeek） | GRPO —— 带组相对基线的 REINFORCE，无 critic |
| 多智能体 | 集中式 critic REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅偏好设置 | DPO —— 重写为偏好似然损失的 REINFORCE，无需采样 |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是建立在这一行之上的方差削减技巧。

## 交付

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 REINFORCE，使用线性 softmax 策略。训练 1,000 个回合，不使用基线。绘制学习曲线；测量方差（回报的标准差）。
2. **中等。** 添加滑动平均基线。再次训练。与原始运行比较样本效率和方差。基线将收敛步数减少了多少？
3. **困难。** 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。在这个任务上，最佳点在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 策略梯度 | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；源自对数导数技巧。 |
| REINFORCE | "原始 PG 算法" | Williams (1992)；蒙特卡洛回报乘以 log-策略梯度。 |
| 对数导数技巧 | "得分函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望的梯度可计算。 |
| 基线 | "方差削减" | 从 `G` 中减去的任意 `b(s)`；无偏因为 `E[b · ∇ log π] = 0`。 |
| 未来回报 | "只有未来回报算数" | `G_t^{from t}` 替代完整的 `G_0`；正确且方差更低。 |
| 熵奖励 | "鼓励探索" | `+β · H(π(·\|s))` 项防止策略坍塌。 |
| 同策略 | "用刚看到的训练" | 梯度期望相对于当前策略——不能直接复用旧数据。 |
| 优势 | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；带基线的 REINFORCE 所乘的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) —— 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) —— 带函数近代的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书式讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) —— 清晰的教学讲解，含 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) —— 方差削减与自然梯度视角，将 REINFORCE 连接到信任域家族（TRPO、PPO）。
