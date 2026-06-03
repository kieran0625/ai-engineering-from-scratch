# 面向游戏的强化学习 —— AlphaZero、MuZero 与大语言模型推理时代

> 1992年：TD-Gammon 凭借纯时序差分学习在西洋双陆棋上击败人类冠军。2016年：AlphaGo 击败李世石。2017年：AlphaZero 从零开始统治国际象棋、日本将棋和围棋。2024年：DeepSeek-R1 证明同样的方法，用 GRPO 替代 PPO，在推理任务上行之有效。游戏是驱动这一阶段每一次突破的基准。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 05 (DQN)、阶段 9 · 08 (PPO)、阶段 9 · 09 (RLHF)、阶段 9 · 10 (MARL)
**时间：** ~120 分钟

## 问题背景

游戏具备强化学习所需的一切条件。清晰的奖励（赢/输）。无限的对局（自我对弈重置）。完美的模拟环境（游戏本身就是模拟器）。离散或小型连续动作空间。迫使对抗鲁棒性的多智能体结构。

而且，游戏是每一次重大强化学习突破的试验场。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）—— 最新证明游戏强化学习技术适用于文本的范例。

本综合项目通过统一的视角审视三种里程碑式架构 —— **AlphaZero、MuZero 和 GRPO**：**自我对弈 + 搜索 + 策略改进**。每一种都泛化了前一种；GRPO 尤其像是将 AlphaZero 的方法应用于大语言模型推理，以 token 作为动作，以数学验证作为获胜信号。

## 核心概念

![AlphaZero ↔ MuZero ↔ GRPO：相同的循环，不同的环境](../assets/rl-games.svg)

**统一的循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017)。** Silver 等人。给定一个具有已知规则的游戏（国际象棋、日本将棋、围棋）：

- 策略-价值网络：一个塔式结构 `f_θ(s) → (p, v)`。`p` 是合法动作的先验分布。`v` 是预期的对局结果。
- 蒙特卡洛树搜索 (MCTS)：每步棋时，展开一棵可能续着树的搜索。使用 `(p, v)` 作为先验 + 引导。通过 UCB (PUCT) 选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：智能体之间进行对局。在第 `t` 步，MCTS 的访问分布 `π_t` 成为策略训练目标。
- 损失函数：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是对局结果（+1 / 0 / -1）。

零人类知识。零手工启发式。单一方法，在数千万盘自我对弈后掌握国际象棋、日本将棋和围棋。

**MuZero (2019)。** Schrittwieser 等人。去除了必须已知规则的要求。

- 不再使用固定环境，而是学习一个*潜在动态模型* `(h, g, f)`：
  - `h(s)`：将观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习的潜在空间*中运行。相同的搜索，相同的训练循环。
- 适用于围棋、国际象棋、日本将棋*以及* Atari —— 一种算法，无需规则知识。

**Stochastic MuZero (2022)。** 增加随机动态和机会节点；扩展至西洋双陆棋类游戏。

**Muesli、Gumbel MuZero (2022-2024)。** 在样本效率和确定性搜索方面的改进。

**GRPO (2024-2025)。** DeepSeek-R1 的方法。相同的 AlphaZero 式循环，应用于语言模型推理：

- "游戏"：解答数学 / 编程 / 推理问题。"获胜" = 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：大语言模型。动作：token。状态：提示词 + 已生成的回复。
- 无评论家（PPO 风格的 V_φ）。相反，对每个提示词，从策略中采样 `G` 个补全。计算每个补全的奖励。使用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 式更新的信号。
- KL 惩罚指向参考策略以防止漂移（类似 RLHF）。
- 完整损失函数：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无评论家，无 MCTS。组相对基线替代了这三者。在推理基准上以远低于 PPO-RLHF 的计算量达到或超过其质量。

**完整的 R1 方法。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。无 SFT。直接应用 GRPO，包含两个奖励组件：*准确性奖励*（基于规则 —— 最终答案是否解析为正确数字 / 代码是否通过单元测试）和*格式奖励*（补全是否将思维链包裹在 `<think>…</think>` 标签中）。经过数千步，平均回复长度从约 100 增长到约 10,000 个 token，数学基准分数攀升至接近 o1-preview 水平。模型从零开始学习推理。缺点：其思维链通常难以阅读、混合语言、缺乏风格润色。
- **R1。** 通过四阶段流程修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千条格式清晰的长思维链演示。对基础模型进行监督微调。这提供了一个可读的起点。
  2. **面向推理的 GRPO。** 应用带有准确性+格式奖励的 GRPO，并增加*语言一致性*奖励以防止代码切换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 600K 条推理轨迹，仅保留最终答案正确且思维链可读的，并与约 200K 条非推理 SFT 示例（写作、问答、自我认知）合并。再次微调基础模型。
  4. **全谱 GRPO。** 最后一轮 RL，同时覆盖推理（基于规则的奖励）和通用对齐（基于帮助性/无害性偏好的奖励）。

结果在 AIME 和 MATH-500 上匹配 o1，以开放权重发布，且足够小可以进行蒸馏。同一论文还发布了六个蒸馏后的稠密模型（Qwen-1.5B 到 Llama-70B），通过对 R1 的推理轨迹进行 SFT 得到 —— 学生端无需 RL。强 RL 教师的蒸馏始终优于学生规模下从零开始的 RL。

**为何在推理中使用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：(1) 无需训练价值网络，内存减半；(2) 组基线自然处理推理任务产生的稀疏轨迹末端奖励；(3) 每提示词归一化使优势在不同难度的问题间可比，而 PPO 的单一评论家无法做到。

**无搜索 vs 基于搜索。** 游戏领域已出现分化：

- *具有长时域的完美信息游戏*（围棋、国际象棋）：仍基于搜索。AlphaZero / MuZero 占主导。
- *大语言模型推理*：生产中尚无 MCTS；GRPO 用于完整 rollout，推理时使用 best-of-N。过程奖励模型 (PRM) 暗示可能重新引入步骤级搜索。

## 动手实现

`code/main.py` 中的代码实现了**微型 GRPO** —— 一个具有多组样本的多臂老虎机。算法与大语言模型上的相同；只是策略和环境更简单。它教授*损失函数*和*组相对优势*，这是 2025 年的创新。

### 步骤 1：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个提示词对 K 个答案 token 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于大语言模型在给定提示词条件下的最终层输出。

### 步骤 3：组采样和组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的诀窍。无需评论家。"基线"是组均值，归一化使用组标准差。

### 步骤 4：与 REINFORCE 基线（无价值函数）比较

相同的设置，相同的计算量，纯 REINFORCE。GRPO 收敛更快更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同的诊断指标：与参考策略的平均 KL、策略熵、随时间变化的奖励。一旦这些稳定，训练即完成。

## 常见陷阱

- **通过欺骗验证器实现的奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器有误或可被利用，大语言模型会找到漏洞。鲁棒的验证器（多个测试用例、形式化证明）至关重要。
- **组大小过小。** 组基线的方差约为 `1/√G`。低于 `G = 4` 时，优势信号嘈杂；标准选择为 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的大语言模型补全具有不同的对数概率。按 token 数量归一化，或使用序列级对数概率，或截断至最大长度。
- **纯自我对弈循环。** AlphaZero 式训练在一般和博弈中可能陷入支配循环。通过多样化的对手池缓解（联赛制，第 10 课）。
- **搜索-策略不匹配。** AlphaZero 训练策略网络模仿搜索输出。如果策略网络太小，无法表示搜索的分布，训练会停滞。
- **计算门槛。** MuZero / AlphaZero 需要大量计算。单次消融通常需要数百 GPU 小时。存在微型演示（例如，Connect Four 上的 AlphaZero）用于学习。
- **验证器覆盖率。** 对错误解决方案通过的单元测试会强化该错误。设计能捕捉边界情况的验证器。

## 实际应用

2026 年游戏强化学习领域概览，按领域划分：

| 领域 | 主导方法 |
|--------|---------|
| 双人零和棋类（围棋、国际象棋、日本将棋） | AlphaZero / MuZero / KataGo |
| 非完美信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略游戏（Dota、星际争霸） | PPO + 自我对弈 + 联赛（OpenAI Five、AlphaStar） |
| 大语言模型数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| 大语言模型对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证的） |
| 机器人 | PPO + DR（非游戏强化学习，但使用相同的策略梯度工具） |
| 组合优化问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

*方法* —— 自我对弈、搜索增强改进、策略蒸馏 —— 跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多正在路上。

## 交付物

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习题

1. **简单。** 在 `code/main.py` 中实现 GRPO 老虎机。在 2 个提示词 × 每个 4 个答案 token 上训练。在 < 1,000 次更新内收敛，使用 `G=8`。
2. **中等。** 接入 PPO（裁剪）和普通 REINFORCE。在相同老虎机上比较与 GRPO 的样本效率和奖励方差。
3. **困难。** 扩展为长度 2 的"推理链"：智能体发出两个 token，验证器奖励该组合。测量 GRPO 如何处理两步序列间的信用分配。（提示：计算*完整序列*的组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；使用学习到的 `(p, v)` 先验的 UCB1/PUCT 选择。 |
| AlphaZero | "自我对弈 + MCTS" | 策略-价值网络，训练目标为匹配 MCTS 访问分布和对局结果。 |
| MuZero | "学习模型的 AlphaZero" | 相同循环，但在潜在空间中通过学习的动态进行。 |
| GRPO | "无评论家 PPO" | 组相对策略优化；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` —— 平衡价值估计与先验。 |
| 自我对弈 | "智能体 vs 过去的自己" | 零和博弈的标准；对称训练信号。 |
| 联赛制 | "基于种群的自我对弈" | 过去 + 当前 + 剥削者作为对手采样。 |
| 验证器奖励 | "可验证强化学习" | 奖励来自确定性检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 为每个推理步骤打分，而非仅最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) —— 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) —— 完整的四阶段 R1 方法及 R1-Zero 消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) —— 大规模的 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) —— 一切的开山之作。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) —— 使用自定义奖励函数应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) —— 多规模开源复现 R1 方法。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— 关于自我对弈、搜索和"设计奖励"的教科书式论述，R1 在大语言模型规模上实例化了这些概念。
