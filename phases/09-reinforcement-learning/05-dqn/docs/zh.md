# 深度 Q 网络（DQN）

> 2013 年：Mnih 在原始像素上训练了一个 Q-learning 网络，在七个 Atari 游戏上击败了所有经典 RL 智能体。2015 年：扩展到 49 个游戏，发表于 Nature，开启了深度 RL 时代。DQN 就是 Q-learning 加上三个让函数逼近稳定的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3 · 03（反向传播），Phase 9 · 04（Q-learning、SARSA）
**时间：** ~75 分钟

## 问题所在

表格型 Q-learning 需要为每个 (state, action) 对存储单独的 Q 值。国际象棋有约 10⁴³ 个状态。Atari 画面是 210×160×3 = 100,800 个特征。表格型 RL 在数千个状态时就已失效，更不用说数十亿了。

事后看来，解决方案显而易见：用神经网络 `Q(s, a; θ)` 替换 Q 表。但"事后显而易见"花了数十年。朴素的 Q-learning 函数逼近在"致命三元组"——函数逼近 + 自举 + 离策略学习——下会发散。Mnih 等人（2013, 2015）找到了三个稳定学习的工程技巧：

1. **经验回放** 消除转移的相关性。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度大小。

Atari 上的 DQN 首次用单一架构、单一超参数集解决了从原始像素出发的数十个控制问题。此后所有"深度 RL"的构建——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都堆叠在这三个技巧的基础之上。

## 核心概念

![DQN 训练循环：env、replay buffer、online net、target net、Bellman TD loss](../assets/dqn.svg)

**目标函数。** DQN 最小化神经网络 Q 函数上的一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（约每 10,000 步）。`D` = 过去转移的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个容量为 `~10⁶` 的环形缓冲区。每步训练时均匀随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络多次从稀有奖励转移中学习，并消除连续梯度更新的相关性。没有它，基于神经网络的同策略 TD 在 Atari 上会发散。

**目标网络。** 在 Bellman 方程两边使用同一个网络 `Q(·; θ)` 会让目标随每次更新而移动——"追逐自己的尾巴"。解决方案：保留第二个权重冻结的网络 `Q(·; θ^-)`。每 `C` 步，复制 `θ → θ^-`。这同时为数千次梯度步稳定了回归目标。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是一种更平滑的变体。

**奖励裁剪。** Atari 的奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可防止任何单个游戏主导梯度。当奖励幅度重要时这是错误的；对 Atari 则没问题，因为那里只有符号重要。

**Double DQN。** Hasselt（2016）修复了最大化偏差：用在线网络*选择*动作，用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用，始终更好。默认使用它。

**其他改进（Rainbow, 2017）：** 优先回放（更频繁采样高 TD 误差转移）、Dueling 架构（分离 `V(s)` 和优势头）、噪声网络（学习探索）、n 步回报、分布式 Q（C51/QR-DQN）、多步自举。每项增加几个百分点；收益大致可加。

## 动手实现

这里的代码仅使用标准库，无 numpy——我们在一个微小的连续 GridWorld 上用手工实现的单隐藏层 MLP，每步训练在微秒级完成。算法与大规模 Atari DQN 完全相同。

### 步骤 1：回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 约需 50,000 容量；我们的玩具环境 5,000 足够。

### 步骤 2：微型 Q 网络（手工 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 步骤 3：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形式与第 04 课的 Q-learning 相同，有两点区别：(a) 我们通过可微的 `Q(·; θ)` 反向传播，而非查表；(b) 目标使用 `Q(·; θ^-)`。

### 步骤 4：外层循环

对每个 episode，基于 `Q(·; θ)` 进行 ε-贪婪行动，将转移推入缓冲区，采样小批量，执行梯度步，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的 16 维 one-hot 状态的微小 GridWorld 上，智能体约 500 个 episode 即可学到接近最优的策略。在 Atari 上，将其扩展到 2 亿帧并添加 CNN 特征提取器。

## 常见陷阱

- **致命三元组。** 函数逼近 + 离策略 + 自举可能发散。DQN 通过目标网络 + 回放来缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常从 1.0 降到 0.01，在前 ~10% 的训练过程中完成。早期探索不足会导致 Q 网络收敛到局部盆地。
- **过高估计。** `max` 对噪声 Q 是向上有偏的。生产环境务必使用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度大小与奖励幅度成正比。
- **回放缓冲区冷启动。** 缓冲区有几千个转移前不要训练。早期对 ~20 个样本的梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太稀疏 ≈ 目标陈旧。Atari DQN 使用 10,000 环境步。经验法则：每 ~1/100 的训练周期同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧使状态具有马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 实际应用

2026 年，DQN 很少是最先进的，但仍是离策略算法的参考基准：

| 任务 | 首选方法 | 为何不用 DQN？ |
|------|----------|--------------|
| 离散动作类 Atari | Rainbow DQN 或 Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有策略网络。 |
| 同策略 / 高吞吐 | PPO（Phase 9 · 08） | 无回放缓冲区；更易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，无自举爆炸。 |
| 大离散动作空间（推荐） | 带动作嵌入的 DQN，或 IMPALA | 可行；装饰很重要。 |
| LLM RL | PPO / GRPO | 序列级而非步级；不同损失。 |

经验教训仍然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自对弈缓冲区以及每个离线 RL 方法中。奖励裁剪作为优势归一化延续到 PPO 中。该架构是蓝图。

## 提交代码

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每 episode 回报曲线。运行均值超过 -10 需要多少 episode？
2. **中等。** 禁用目标网络（在线网络用于 Bellman 目标的两边）。测量训练不稳定性——回报是否震荡或发散？
3. **困难。** 添加 Double DQN：用在线网络选择 `argmax a'`，目标网络评估。在有/无 Double DQN 的情况下，比较噪声奖励 GridWorld 上 1,000 个 episode 后 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| DQN | "Deep Q-learning" | 带神经网络 Q 函数、回放缓冲区和目标网络的 Q-learning。 |
| 经验回放 | "Shuffled transitions" | 每梯度步均匀采样的环形缓冲区；消除数据相关性。 |
| 目标网络 | "Frozen bootstrap" | Q 的周期性副本，用于 Bellman 目标；稳定训练。 |
| 致命三元组 | "Why RL diverges" | 函数逼近 + 自举 + 离策略 = 无收敛保证。 |
| Double DQN | "Fix for maximization bias" | 在线网络选动作，目标网络评估它。 |
| Dueling DQN | "V and A heads" | 分解 Q = V + A - mean(A)；相同输出，更好梯度流。 |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional 合一。 |
| PER | "Prioritized Replay" | 按 TD 误差幅度比例采样转移。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 开启深度 RL 的 2013 NeurIPS  workshop 论文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 游戏 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — Dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代阐述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 关于"致命三元组"（函数逼近 + 自举 + 离策略）的教科书式处理，DQN 的目标网络和回放缓冲区正是为驯服它而设计。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 消融研究中使用的参考单文件 DQN；与本课从零实现版本对照阅读为佳。
