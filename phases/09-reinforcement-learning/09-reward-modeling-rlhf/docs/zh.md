# 奖励建模与 RLHF

> 人类无法为“优质助手回复”写出奖励函数，但他们可以比较两个回复并选出更好的一个。用这些比较数据拟合一个奖励模型，然后让语言模型通过 RL 对其进行优化。Christiano 2017。InstructGPT 2022。将 GPT-3 变成 ChatGPT 的配方。到 2026 年，它大多已被 DPO 取代——但核心思维模型依然适用。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 05（情感分析），Phase 9 · 08（PPO）
**时间：** ~45 分钟

## 问题所在

你用 next-token-prediction 目标训练了一个语言模型。它能写出语法正确的英文。但它也会撒谎、啰嗦、该拒绝时不拒绝。更多预训练无法解决这个问题——网络文本就是病因，不是解药。

你想要一个*标量奖励*来表达“对于指令 X，回复 A 比回复 B 更好”。手写这个奖励函数是不可能的。“有用性”不是关于 token 的闭式表达式。但人类可以比较两个输出并标记偏好。这可以大规模低成本收集。

RLHF（Christiano et al. 2017; Ouyang et al. 2022）将偏好转化为奖励模型，然后通过 PPO 针对该奖励优化语言模型。分三步：SFT → RM → PPO。这是 ChatGPT、Claude、Gemini 以及 2023–2025 年间所有对齐 LLM 的配方。

到 2026 年，PPO 步骤大多被 DPO（Phase 10 · 08）取代，因为它更便宜，在对齐调优上几乎同样有效。但*奖励模型*这一组件仍然支撑着每一个 Best-of-N 采样器、每一个基于可验证奖励的 RL 流水线，以及每一个使用过程奖励模型的推理模型。理解 RLHF，你就理解了对齐的整个技术栈。

## 核心概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（SFT）。** 从预训练基模型开始。在人类编写的目标行为演示数据（指令遵循回复、有帮助的回复等）上进行微调。结果：模型 `π_SFT` *偏向良好行为*，但动作空间仍然无界。

**阶段 2：奖励模型训练。**

- 收集对提示 `x` 的成对回复 `(y_+, y_-)`，由人类标注为“y_+ 优于 y_-”。
- 训练奖励模型 `R_φ(x, y)`，使其对 `y_+` 给出更高分数。
- 损失函数：**Bradley-Terry 成对逻辑回归**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid 函数。奖励之差意味着偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来就是标准，也是现代 RLHF 中的主流选择。

- `R_φ` 通常从 SFT 模型初始化，顶部加一个标量头。相同的 transformer 主干；单个线性层输出奖励。

**阶段 3：针对带 KL 惩罚的 RM 运行 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留一个冻结的*参考*模型 `π_ref = π_SFT`。
- 回复结束时的奖励 `y`：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT`——它是一个*正则化项*，而非硬信任域。`β` 通常为 `0.01`-`0.05`。
- 用此奖励运行 PPO（第 08 课）。优势在 token 级别轨迹上计算，但 RM 只对完整回复打分。

**为什么需要 KL？** 没有它，PPO 会乐于发现奖励黑客策略——RM 只在分布内完成上训练。分布外的回复可能比任何人类写的回复得分更高。KL 将 `π_θ` 保持在 RM 训练所在的流形附近。它是 RLHF 中最重要的调节旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：闭式代数将阶段 2+3 折叠为基于偏好数据的单一监督损失。没有 RM，没有 PPO。在对齐基准上质量相同，计算量却少得多。将在 Phase 10 · 08 中讲解。
- **GRPO**（DeepSeek 2024–2025）：用组相对基线替代 critic 的 PPO，奖励来自*验证器*（代码运行/数学答案匹配）而非人工训练的 RM。在推理模型中占主导。将在 Phase 9 · 12 中讲解。
- **过程奖励模型（PRM）：** 对部分解（每个推理步骤）打分，用于 RLHF 和 GRPO 的推理变体。
- **Constitutional AI / RLAIF：** 用对齐的 LLM 生成偏好而非人类。扩展偏好预算。

## 动手实现

本课使用由字符串表示的微型合成“提示”和“回复”。RM 是基于词袋表示的线性打分器。没有真正的 LLM——重要的是流水线的*结构*，而非规模。参见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中，这由人类标注员完成。结构——`(prompt, preferred_response, rejected_response)`——完全相同。

### 步骤 2：Bradley-Terry 奖励模型

线性分数：`R(x, y) = w · bag(y)`。训练目标是最小化 BT 成对 log-loss：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

经过几百次更新后，`w` 会给“好词” token 赋予正权重，“坏词”赋予负权重。

### 步骤 3：基于 RM 的类 PPO 策略

我们的玩具策略从词汇表中生成单个 token。我们在 RM 下对该 token 打分，计算 `log π_θ(token | prompt)`，加上到参考模型的 KL 惩罚，并应用裁剪后的 PPO 替代目标。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 步骤 4：监控 KL

每次更新跟踪平均 `KL(π_θ || π_ref)`。如果它超过 `~5-10`，说明策略已大幅偏离 `π_SFT`——要么 `β` 在上升，要么奖励黑客行为开始。这是真实 RLHF 中的首要诊断指标。

### 步骤 5：使用 TRL 的生产配方

理解玩具流水线后，以下是真实库用户编写的相同循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于阶段 2，`PPOTrainer`（内置到参考模型的 KL）用于阶段 3。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库为你做的三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观测到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按约定冻结——你不能意外地与 `policy` 共享参数。而价值头与策略共享同一主干（`AutoModelForCausalLMWithValueHead` 附加一个标量 MLP 头），这就是为什么 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## 常见陷阱

- **过度优化 / 奖励黑客。** RM 不完美；`π_θ` 会找到得分高但实际很差的对抗性完成。症状：奖励持续上升，而人工评估分数停滞或下降。修复：早停、提高 `β`、扩展 RM 训练数据。
- **长度黑客。** 在有帮助的回复上训练的 RM 往往隐式奖励长度。策略学会填充回复。修复：长度归一化奖励，或用长度感知 RM 进行 RLAIF。
- **RM 太小。** RM 至少需要与策略一样大。微小的 RM 无法忠实打分策略的输出。
- **KL 调参。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是使用*自适应* β，目标为每步固定 KL。
- **偏好数据噪声。** ~30% 的人类标注有噪声或模糊。通过在一致性情过滤数据上训练 RM，或在 BT 上使用温度来校准。
- **离策略问题。** 第一轮之后，PPO 数据略微离策略。如第 08 课所述，监控裁剪比例。

## 应用场景

2026 年的 RLHF 是分层的：

| 层级 | 目标 | 方法 |
|------|------|------|
| 指令遵循、有用性、无害性 | 对齐 | DPO（Phase 10 · 08）优先于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 带验证器奖励的 GRPO（Phase 9 · 12）。 |
| 长程多步任务 | 智能体 | 基于步骤的过程奖励模型的 PPO / GRPO。 |
| 安全性 / 拒绝行为 | 安全 | 带独立安全 RM 的 RLHF-PPO，或 Constitutional AI。 |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练小型“奖励头”。 |

RLHF 曾是 2022–2024 年的*核心*方法。到 2026 年，生产对齐流水线以 DPO 优先，仅在需要密集 RM 或安全关键步骤时使用 PPO。

## 交付

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 `code/main.py` 上用 500 对合成偏好数据训练 Bradley-Terry 奖励模型。在 100 对 held-out 数据上测量成对准确率。应超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个设置，绘制 RM 分数与到参考模型的 KL 随更新的变化。哪些设置出现奖励黑客？
3. **困难。** 在相同偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流水线在计算量和最终 RM 分数上进行比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| RLHF | “对齐 RL” | 三阶段 SFT + RM + PPO 流水线（Christiano 2017, Ouyang 2022）。 |
| 奖励模型（RM） | “打分网络” | 通过 Bradley-Terry 拟合成对偏好的学习标量函数。 |
| Bradley-Terry | “成对逻辑损失” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准的 RM 目标函数。 |
| KL 惩罚 | “靠近参考模型” | 奖励中的 `β · KL(π_θ \|\| π_ref)`；防止奖励黑客的正则化项。 |
| 奖励黑客 | “古德哈特定律” | 策略利用 RM 缺陷；症状：奖励上升，人工评估持平。 |
| RLAIF | “AI 标注的偏好” | 标签来自另一个 LM 而非人类的 RLHF。 |
| PRM | “过程奖励模型” | 对部分推理步骤打分；用于推理流水线。 |
| Constitutional AI | “Anthropic 的方法” | 由显式规则指导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) —— 开启 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) —— 更早的摘要 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) —— DPO；2026 年后 RLHF 时代的默认选择。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) —— RLAIF 与自我批判循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) —— HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) —— 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码了解自适应 KL 和价值头细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla —— 三阶段流水线配图表的经典 walk-through。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) —— 该库；`examples/` 有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) —— 奖励假设视角；思考奖励黑客的必备前置知识。
