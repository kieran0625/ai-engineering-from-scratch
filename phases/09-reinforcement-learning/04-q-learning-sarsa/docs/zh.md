# 时序差分 —— Q-Learning 与 SARSA

> Monte Carlo 要等到回合结束才更新。TD 则通过自举（bootstrapping）下一步的价值估计，在每一步之后立即更新。Q-learning 是离策略（off-policy）且乐观的；SARSA 是在策略（on-policy）且谨慎的。两者都只需一行代码。本阶段所有深度强化学习方法都建立在它们之上。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**时间：** ~75 分钟

## 问题

Monte Carlo 方法有效，但有两个昂贵的需求。它需要能够终止的回合，并且只在最终回报确定后才更新。如果你的回合有 1,000 步，MC 就要等 1,000 步才更新任何内容。它是高方差、低偏差的，在实践中很慢。

动态规划则正好相反——零方差的自举备份——但需要已知的模型。

时序差分（TD）学习取两者之长。从单个转移 `(s, a, r, s')` 出发，构造一步目标 `r + γ V(s')`，并将 `V(s)` 向其推进。不需要模型，不需要完整的回合。代价是在右侧使用了近似的 `V` 而引入偏差，但方差远低于 MC，且从第一步起就能在线更新。

这是现代强化学习的转折点——DQN、A2C、PPO、SAC 都建立于此。Phase 9 的其余内容都是在本课所写的一步 TD 更新之上，叠加函数近似和各种技巧。

## 概念

![Q-learning vs SARSA: 离策略 max vs 在策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛需要 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`, `Σ α² < ∞`），且所有状态被无限次访问。

**Q-learning。** 一种离策略 TD 控制方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使 Q-learning 在学习 `Q*` 的同时，智能体通过 ε-greedy 进行探索。Mnih 等人（2015）将其转化为 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种在策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称即元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而非贪婪的 `argmax`。收敛到当前 ε-greedy `π` 的 `Q^π`，当 `ε → 0` 时 `Q*`。

**悬崖行走的区别。** 在经典的悬崖行走任务中（掉下悬崖 = 奖励 -100），Q-learning 学习沿着悬崖边缘的最优路径，但偶尔会在探索时受到惩罚。SARSA 学习离悬崖一步的安全路径，因为它将探索噪声纳入了 Q 值。经过训练，两者在 `ε → 0` 时都达到最优。在实践中这很重要：当部署时实际发生探索时，SARSA 的行为更为保守。

**Expected SARSA。** 将 `Q(s', a')` 替换为其在 `π` 下的期望值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（无需对 `a'` 采样），目标仍在策略内。现代教科书中常作为默认选择。

**n-step TD 与 TD(λ)。** 通过在自举前等待 `n` 步，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 以几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 取平均。大多数深度 RL 使用 3 到 20 之间的 `n`。

## 动手实现

### 步骤 1：ε-greedy 策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 的*唯一*区别就是目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号就是在策略与离策略的区别。

### 步骤 3：学习曲线

每 100 回合追踪平均回报。Q-learning 在简单的确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 的 4×4 GridWorld 上，使用 `α=0.1, ε=0.1` 经过约 2,000 回合后两者都接近最优。

### 步骤 4：与 DP 真值比较

运行值迭代（第 02 课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。在 4×4 GridWorld 上，健康的表格型 TD 智能体在 10,000 回合后应落在 `~0.5` 范围内。

## 常见陷阱

- **初始 Q 值很重要。** 乐观初始化（对负奖励任务设 `Q = 0`）鼓励探索。悲观初始化可能使贪婪策略永远被困。
- **α 调度。** 恒定的 `α` 对非平稳问题适用。衰减的 `α_n = 1/n` 理论上保证收敛，但实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从高值开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（极限贪婪无限探索）是收敛条件。
- **Q-learning 中的 max 偏差。** 当 `Q` 有噪声时，`max` 算子向上有偏。导致高估——Hasselt 的 Double Q-learning（第 05 课的 DDQN 使用）通过两个 Q 表修复此问题。
- **非终止回合。** TD 可以在没有终止的情况下学习，但需要么限制步数，要么在限制处正确处理自举。标准做法：将限制视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（元组而非列表；浮点数取整后的元组，而非原始值）。

## 应用场景

2026 年的 TD 图景：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在策略安全关键场景 | SARSA / Expected SARSA | 探索期间更保守。 |
| 高维状态 | DQN (Phase 9 · 05) | 带经验回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3 (Phase 9 · 07) | Q 网络上的 TD 更新；策略网络输出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO (Phase 9 · 08, 12) | 通过 GAE 实现类 TD 优势的 Actor-critic。 |
| 离线 RL | CQL / IQL (Phase 9 · 08) | 带保守正则化的 Q-learning。 |

2026 年论文中你读到的"RL"有百分之九十都是 Q-learning 或 SARSA 的某种变体。在深入阅读之前，先用手指理解表格型更新。

## 提交

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2,000 回合的学习曲线（每 100 回合平均回报）。谁收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最后一行是悬崖，奖励 -100 并重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截图各自的路径。哪个更靠近悬崖？
3. **困难。** 实现 Double Q-learning。在有噪声奖励的 GridWorld 上（每步奖励添加高斯噪声 σ=5），展示 Q-learning 明显高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| TD error | "更新信号" | `δ = r + γ V(s') - V(s)`，自举后的残差。 |
| TD(0) | "一步 TD" | 每次转移后使用下一状态的估计进行更新。 |
| Q-learning | "离策略 RL 入门" | 对下一状态动作取 `max` 的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "在策略的 Q-learning" | 使用实际下一动作的 TD 更新；学习当前 ε-greedy π 的 `Q^π`。 |
| Expected SARSA | "低方差 SARSA" | 将采样的 `a'` 替换为其在 π 下的期望。 |
| GLIE | "正确的探索调度" | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需。 |
| Bootstrapping | "用当前估计作为目标" | TD 区别于 MC 的特征。带来偏差但大幅降低方差。 |
| Maximization bias | "Q-learning 高估" | 对噪声估计取 `max` 向上有偏；Double Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) —— 原始论文与收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) —— 最大偏差的修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) —— Expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) —— 首创 SARSA 的论文（当时称为"modified connectionist Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) —— 将 TD(0) 推广到 TD(n)，从 Q-learning 到资格迹、再到 PPO 中 GAE 的路径。
