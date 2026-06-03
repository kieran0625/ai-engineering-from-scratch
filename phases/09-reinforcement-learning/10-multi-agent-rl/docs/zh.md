# 多智能体强化学习

> 单智能体强化学习假设环境是静态的。把两个学习中的智能体放进同一个世界，这个假设就失效了：每个智能体都是对方环境的一部分，而且双方都在不断变化。多智能体强化学习就是一系列技巧，用于在马尔可夫假设不再成立时让学习收敛。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 04 (Q-learning)，阶段 9 · 06 (REINFORCE)，阶段 9 · 07 (Actor-Critic)
**时间：** ~45 分钟

## 问题

一个机器人学习在房间中导航是单智能体强化学习问题。一支足球队不是。AlphaStar 对战星际争霸对手不是。一个由竞价智能体组成的市场不是。两辆车在十字路口协商通行不是。许多真实世界的多对多问题都不是。

在每个多智能体场景中，从任意一个智能体的视角来看，其他智能体*就是*环境的一部分。当它们学习并改变行为时，环境就变得非静态。马尔可夫性质——"下一状态仅取决于当前状态和我的动作"——被违反了，因为下一状态还取决于*其他*智能体选择了什么，而它们的策略是移动的目标。

这破坏了表格收敛证明（Q-learning 的保证假设环境是静态的）。它也破坏了朴素深度强化学习：智能体互相追逐，陷入循环，永远无法收敛到稳定的策略。你需要多智能体专用技术：集中训练/分散执行、反事实基线、联盟训练、自我对弈。

2026 年应用：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（阶段 16），以及任何有不止一个智能玩家的游戏。

## 概念

![四种 MARL 范式：独立、集中式评论家、自我对弈、联盟](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP 的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自身策略 `π_i` 下最大化自身回报。如果奖励相同，则为**完全合作**。如果零和，则为**对抗**。如果混合，则为**一般和**。

**核心挑战：**

- **非静态性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 取决于 `π_{-i}`，而它在不断变化。
- **信用分配。** 面对共享奖励，是哪个智能体造成的？
- **探索协调。** 智能体必须探索互补策略，而不是冗余地探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观测性。** 每个智能体只能看到自己的观测；全局状态是隐藏的。

**四种主要范式：**

**1. 独立 Q-learning / 独立 PPO (IQL, IPPO)。** 每个智能体学习自己的 Q 函数或策略，将其他智能体视为环境的一部分。简单，有时有效（尤其是经验回放起到了平滑智能体建模技巧的作用）。理论收敛性：无。实践中：适用于松耦合任务，不适用于紧耦合任务。

**2. 集中训练，分散执行 (CTDE)。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，以局部观测 `o_i` 为条件——部署时标准分散执行。在*训练*期间，集中式评论家 `Q(s, a_1, …, a_n)` 以完整全局状态和联合动作为条件。示例：
- **MADDPG** (Lowe et al. 2017)：带集中式评论家的 DDPG。
- **COMA** (Foerster et al. 2017)：反事实基线——问"如果我采取了动作 `a'` 而不是原来的动作，我的奖励会是多少？"——隔离我的贡献。
- **MAPPO** / **IPPO** with shared critic (Yu et al. 2022)：带集中式价值函数的 PPO。2026 年合作式 MARL 的主流方法。
- **QMIX** (Rashid et al. 2018)：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` 通过单调混合。

**3. 自我对弈。** 同一智能体的两个副本互相对战。对手的策略*就是*我过去某个快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适用于零和游戏；训练信号是对称的。

**4. 联盟训练。** 自我对弈在一般和/对抗环境中的扩展：保留过去和当前策略的种群，从联盟中采样对手进行训练。添加利用者（专门击败当前最优策略）和主利用者（专门击败利用者）。AlphaStar（星际争霸 II）。当游戏存在"石头剪刀布"策略循环时需要。

**通信。** 允许智能体互相发送学习到的消息 `m_i`。在合作场景中有效。Foerster et al. (2016) 表明智能体间的可微通信可以端到端训练。如今基于 LLM 的多智能体系统（阶段 16）本质上就是用自然语言通信。

## 构建

本课使用一个 6×6 的 GridWorld，包含两个合作智能体。它们从对角角落出发，必须到达共享目标。共享奖励：任一智能体仍在移动时 `-1` 每步，两者都到达时 `+10`。参见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 步骤 2：独立 Q-learning

每个智能体运行自己的 Q 表，以联合状态为键。每步：两者都选择 ε-贪婪动作，收集联合转移，各自用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在此任务上有效，因为奖励是密集且对齐的。在紧耦合任务上失败（例如，一个智能体必须*等待*另一个）。

### 步骤 3：集中式 Q 与分解价值更新

使用一个覆盖联合动作 `Q(s, a_1, a_2)` 的 Q。从共享奖励更新。执行时通过边缘化分散：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。以指数级联合动作空间为代价换取*正确的*全局视角。

### 步骤 4：简单自我对弈（对抗性 2 智能体）

同一智能体，两个角色。训练智能体 A 对抗智能体 B；`K` 轮后，将 A 的权重复制到 B。对称训练，持续进步。迷你版 AlphaZero 配方。

## 陷阱

- **非静态回放。** 独立智能体的经验回放比单智能体更差，因为旧的转移是由现在已过时的对手生成的。修复：重标注或按新近度加权。
- **信用分配模糊。** 长回合后的共享奖励；无法明确说清哪个智能体做出了贡献。修复：反事实基线（COMA），或每个智能体的奖励塑形。
- **策略漂移/追逐。** 每个智能体的最佳响应随对方的更新而变化。修复：集中式评论家、慢学习率，或一次冻结一个。
- **通过协调的奖励黑客。** 智能体发现设计者未预料到的协调漏洞。拍卖智能体收敛到出价零。修复：仔细的奖励设计、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或角色条件化。
- **联盟循环。** 纯自我对弈可能陷入优势循环。修复：用多样化对手的联盟训练。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数近似近似；分解动作空间（每个智能体一个策略输出头）。

## 应用

2026 年 MARL 应用地图：

| 领域 | 方法 | 说明 |
|------|------|------|
| 合作导航/操作 | MAPPO / QMIX | CTDE；共享评论家 + 分散执行者。 |
| 双人游戏（象棋、围棋、扑克） | 带 MCTS 的自我对弈 (AlphaZero) | 零和；对称训练。 |
| 复杂多人游戏（Dota、星际争霸） | 联盟训练 + 模仿预训练 | OpenAI Five, AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / PPO with attention | 部分观测；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时使用平均场 RL。 |
| LLM 多智能体系统（阶段 16） | 自然语言通信 + 角色条件化 | 智能体规划层的 RL 循环。 |

2026 年，MARL 最大的增长领域是基于 LLM 的：语言模型智能体集群协商、辩论、构建软件。RL 表现为对*轨迹级*输出的偏好优化，而非 token 级（阶段 16 · 03）。

## 交付

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2 智能体合作 GridWorld 上训练独立 Q-learning。平均回报 > 0 需要多少轮？绘制联合学习曲线。
2. **中等。** 添加"协调"任务：仅当两个智能体同时踏上目标时才达成目标。独立 Q 还能收敛吗？什么会失效？
3. **困难。** 实现 MAPPO 风格的集中式评论家，并与独立 PPO 在协调任务上的收敛速度进行比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Markov game | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "集中训练，分散执行" | 训练时联合评论家；每个智能体的策略仅使用局部观测。 |
| IPPO | "独立 PPO" | 每个智能体单独运行 PPO。简单基线；常被低估。 |
| MAPPO | "多智能体 PPO" | 以全局状态为条件的集中式价值函数的 PPO。 |
| QMIX | "单调价值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分散式 argmax。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我的动作边缘化的期望 Q。 |
| Self-play | "智能体对战过去的自己" | 单一智能体，两个角色；零和游戏的标准方法。 |
| League play | "种群训练" | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带集中式评论家的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的价值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出奇地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联盟训练。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和游戏中的纯自我对弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教材对多智能体场景及 CTDE 旨在解决的非静态性问题的简要处理。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 及收敛结果的综述。
