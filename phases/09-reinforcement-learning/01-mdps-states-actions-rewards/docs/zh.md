# 马尔可夫决策过程、状态、动作与奖励

> 马尔可夫决策过程包含五个要素：状态、动作、转移、奖励、折扣因子。强化学习中的所有内容——Q-learning、PPO、DPO、GRPO——都是在这个框架上进行优化。学会一次，后续所有强化学习内容都能免费读懂。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 1 · 06（概率与分布），阶段 2 · 01（机器学习分类学）
**时间：** ~45 分钟

## 问题

你在编写一个国际象棋机器人。或者一个库存规划系统。或者一个交易智能体。或者训练推理模型的 PPO 循环。四个不同的领域，一个令人惊讶的事实：它们都可以归结为同一个数学对象。

监督学习给你 `(x, y)` 对，让你拟合一个函数。强化学习没有标签——只有一连串的状态、你采取的动作，以及一个标量奖励。这步棋赢了棋吗？补货决策省钱了吗？交易盈利了吗？LLM 刚刚生成的 token 是否带来了评审者更高的奖励？

你无法直接从这条数据流中学习，除非你将其形式化。"我看到了什么"、"我做了什么"、"接下来发生了什么"、"这有多好"——每一项都必须变成可以推理的对象。这种形式化就是马尔可夫决策过程。本阶段的每个 RL 算法，包括最后的 RLHF 和 GRPO 循环，都是在这个框架上进行优化。

## 概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个要素。**

- **状态** `S`。智能体做出决策所需的全部信息。在 GridWorld 中，是格子位置。在国际象棋中，是棋盘局面。在 LLM 中，是上下文窗口加上任何记忆。
- **动作** `A`。可选的行为。向上/下/左/右移动。走一步棋。生成一个 token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a`，下一个状态的概率分布。国际象棋中是确定性的，库存问题中是随机的，LLM 解码中几乎是确定性的。
- **奖励** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减去成本。GRPO 中的对数似然比项。
- **折扣因子** `γ ∈ [0, 1)`。未来奖励相对于现在的权重。`γ = 0.99` 对应约 100 步的有效范围；`γ = 0.9` 对应约 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖于当前状态。如果不满足，说明状态表示不完整——这不是方法的失败，而是状态的失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射为动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。价值 `V^π(s) = E[G_t | s_t = s]` 是从 `s` 出发、在策略 `π` 下的期望回报。Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定动作开始的期望回报。每个 RL 算法都会估计这两者之一，然后相应地改进 `π`。

**贝尔曼方程。** 本阶段所有内容都会用到的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报拆分为"这一步的奖励"加上"落点状态的折扣价值"。递归的。阶段 9 中的每个算法要么迭代这个方程直至收敛（动态规划），要么从中采样（蒙特卡洛），要么单步自举（时序差分）。

## 动手实现

### 步骤 1：一个微型确定性 MDP

4×4 的 GridWorld。智能体从左上角开始，右下角为终止状态，每步奖励 -1，动作 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境。确定性转移、恒定的步数惩罚、吸收型终止状态。

### 步骤 2：执行策略

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。这个 4×4 棋盘的平均回报约为 -60 到 -80。最优回报是 -6（直线路径向右下）。缩小这个差距就是阶段 9 的全部内容。

### 步骤 3：通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到价值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是 Sutton & Barto 中的第一个算法，也是后续所有 RL 方法的理论基础。

### 步骤 4：`γ` 是具有物理含义的超参数

有效范围大致为 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低会导致智能体短视。太高则信用分配变得嘈杂，因为许多早期步骤共同承担了远期奖励的责任。LLM RLHF 通常使用 `γ = 1`，因为回合短且有界。控制任务使用 `0.95–0.99`。长程策略游戏使用 `0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果你需要最近三次观测才能决策，那么"状态"就不只是当前观测。修复方法：堆叠帧（Atari 上的 DQN 堆叠 4 帧）或使用循环状态（对观测序列使用 LSTM/GRU）。
- **稀疏奖励。** 只有获胜才有奖励使得在大型状态空间中几乎无法学习。塑造奖励（中间信号）或用模仿学习启动（阶段 9 · 09）。
- **奖励作弊。** 优化代理奖励往往会产生病态行为。OpenAI 的赛艇智能体原地转圈无限收集能量包，而不是完成比赛。始终从目标结果定义奖励，而非代理指标。
- **折扣因子设定错误。** 在无限范围任务上 `γ = 1` 会使每个价值变为无穷。始终用有限范围或 `γ < 1` 来限制。
- **奖励尺度。** {+100, -100} 与 {+1, -1} 的最优策略相同，但梯度幅度差异巨大。在输入 PPO/DQN 之前归一化到 `[-1, 1]` 量级。

## 应用

2026 年的技术栈将每个 RL 流水线都先归约为 MDP，再写代码：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（ locomotion、操作） | 关节角度 + 速度 | 连续扭矩 | 任务特定的塑形奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘 + 历史 | 合法着法 | 赢=+1 / 输=-1 | 1.0（有限） |
| 库存 / 定价 | 库存 + 需求 | 订购量 | 收入 - 成本 | 0.95 |
| LLM 的 RLHF | 上下文 token | 下一个 token | 回合结束时奖励模型评分 | 1.0（回合约 200 token） |
| 推理的 GRPO | 提示 + 部分响应 | 下一个 token | 回合结束时验证器 0/1 | 1.0 |

在写任何训练循环之前，先写出这五个元组。大多数"RL 不工作"的 bug 报告都可以追溯到纸面上就有问题的 MDP 形式化。

## 交付

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略执行。运行 10,000 个回合。报告回报的均值和标准差。与最优回报（-6）比较。
2. **中等。** 对均匀随机策略运行 `policy_evaluation`，使用 `γ ∈ {0.5, 0.9, 0.99}`。将 `V` 打印为 4×4 网格。解释为什么终止状态附近的状态价值随 `γ` 增大而增长更快。
3. **困难。** 将 GridWorld 变为随机的：每个动作以概率 `p = 0.1` 滑向相邻方向。重新评估均匀策略。`V[start]` 会变好还是变差？为什么？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| MDP | "强化学习设定" | 满足马尔可夫性质的元组 `(S, A, P, R, γ)`。 |
| 状态 | "智能体看到的东西" | 在选定策略类下对未来动态的充分统计量。 |
| 策略 | "智能体的行为" | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| 回报 | "总奖励" | 从当前步开始的折扣和 `Σ γ^t r_t`。 |
| 价值 | "一个状态有多好" | 从 `s` 出发、在 `π` 下的期望回报。 |
| Q 值 | "一个动作有多好" | 从 `s` 出发、采取首动作 `a`、在 `π` 下的期望回报。 |
| 贝尔曼方程 | "动态规划递归" | 将价值 / Q 分解为单步奖励加折扣后继价值的不动点分解。 |
| 折扣因子 `γ` | "未来 vs 现在" | 远期奖励的几何权重；有效范围 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 3 章涵盖 MDP 和贝尔曼方程；第 1 章阐述了支撑后续每节课的奖励假设。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度 RL 角度简洁介绍 MDP。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 运筹学视角下关于 MDP 和精确求解方法的经典参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDP 作为动态规划特例的最清晰推导。
