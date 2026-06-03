# 从仿真到现实的迁移

> 一个在仿真器中训练但在硬件上失效的策略，本质上只是记住了仿真器。域随机化、域自适应和系统辨识是让学习控制器跨越现实鸿沟的三大工具。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 9 · 08 (PPO), Phase 2 · 10 (偏差/方差)
**时间：** ~45 分钟

## 问题所在

训练真实机器人缓慢、危险且昂贵。双足机器人需要数百万个训练回合才能学会行走；而真实的双足机器人即使只摔倒一次也会损坏硬件。仿真提供无限次重置、确定性可复现性、并行环境以及无物理损伤。

但仿真器是不完美的。轴承的摩擦力大于 MuJoCo 模型的设定。相机存在镜头畸变，而仿真器并未包含。电机存在延迟、间隙和饱和现象，99% 的仿真模型都忽略了这些。风、灰尘和可变光照会破坏在无菌渲染环境下训练的策略。**现实鸿沟**——仿真分布与真实分布之间的系统性差异——是机器人强化学习部署中的核心问题。

你需要一个对仿真到现实的分布偏移具有*鲁棒性*的策略。历史上三种方法：随机化仿真器（域随机化）、用少量真实数据调整策略（域自适应/微调），或辨识真实系统的参数并匹配它们（系统辨识）。到 2026 年，主流方案将三者结合，并配合大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## 概念

![三种仿真到现实的方案：域随机化、自适应、系统辨识](../assets/sim-to-real.svg)

**域随机化（DR）。** Tobin 等人 2017，Peng 等人 2018。训练期间，随机化每一个可能在真实机器人上不同的仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习一个关于"今天它在哪个仿真器中"的条件分布，并在整个范围内泛化。如果真实机器人落在训练包络之内，策略就能工作。

- **优点：** 不需要真实数据。一套方案，适用多种机器人。
- **缺点：** 过度随机化的训练会产生"通用"但过于保守的策略。噪声太多 ≈ 正则化过强。

**系统辨识（SI）。** 将仿真器的参数拟合到真实世界数据，然后再训练。如果你能测量真实机器人臂关节的摩擦力，就将其输入仿真器。然后训练一个期望这些值的策略。需要接触真实系统，但能直接缩小现实鸿沟。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残余模型误差对策略不可见；小的未辨识效应（如电机死区）仍会导致部署失败。

**域自适应。** 在仿真中训练，用少量真实数据微调。两种变体：

- **Real2Sim2Real：** 使用真实 rollout 学习一个残差仿真器 `f(s, a, z) - f_sim(s, a)`，在修正后的仿真器中训练。无需大量真实数据即可缩小差距。
- **观测自适应：** 训练一个策略，通过学习的特征提取器（如 GAN 像素到像素）将真实观测映射为类仿真观测。控制器保持在仿真中。

**特权学习 / 教师-学生。** Miki 等人 2022（ANYmal 四足机器人）。在仿真中训练一个*教师*，它可以访问特权信息（真实摩擦、地形高度、IMU 漂移）。蒸馏一个*学生*，它只能看到真实传感器观测。学生学会从历史中推断特权特征，在不同物理参数下保持鲁棒。

**大规模并行仿真。** 2024–2026 年。Isaac Lab、Mujoco MJX、Brax 都能在单个 GPU 上运行数千个并行机器人。PPO 配合 4,096 个并行人形机器人在数小时内收集数年的经验。随着训练分布的扩大，"现实鸿沟"缩小；当这 4,096 个环境每个都有不同的随机化参数时，DR 几乎免费。

**真实世界 2026 年方案（四足行走示例）：**

1. 大规模并行仿真，域随机化重力、摩擦、电机增益、负载。
2. 使用特权信息（地形图、身体速度真实值）训练教师策略。
3. 仅使用本体感知（腿部关节编码器）从教师蒸馏学生策略。
4. 可选：通过真实 IMU 上的自编码器进行观测自适应。
5. 部署。在 10 多种环境中零样本迁移。如果失败，用安全约束 PPO 进行数分钟的真实世界微调。

## 动手实现

本课的代码是在一个具有*噪声*转移的 GridWorld 上演示域随机化的微型示例。我们训练一个策略，它在"仿真"中经历随机的打滑概率，并在训练期间从未见过的"真实"打滑水平上评估。其形式直接对应于 MuJoCo 到硬件的迁移。

### 步骤 1：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露的一个参数。在真实机器人中，它可以是摩擦、质量、电机增益——任何在仿真和真实之间变化的东西。

### 步骤 2：使用 DR 训练

在每个回合开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何算法。持续多个回合。

### 步骤 3：在"真实"打滑上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支撑集内；`0.5` 和 `0.7` 在外部。DR 训练的策略应在支撑集内保持接近最优，在外部 graceful 退化。固定打滑训练的策略在其训练打滑之外会很脆弱。

### 步骤 4：与窄范围训练对比

用仅 `slip = 0.0` 训练第二个策略。在相同的 `slip` 扫描上评估。一旦真实打滑 > 0，你应该会看到灾难性下降。

## 陷阱

- **随机化过度。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略会过于规避风险，从不尝试最优路径。匹配*预期的*真实世界分布，而非"任何事都可能发生"。
- **随机化不足。** 在薄切片上训练，策略完全无法泛化。使用自适应课程（自动域随机化），随着策略提升而扩大分布。
- **参数空间辨识错误。** 随机化错误的东西（相机色调，而真实鸿沟是电机延迟），DR 帮不上忙。先分析真实机器人。
- **特权信息泄漏。** 教师使用全局状态而非仅观测来决策，会导致学生无法跟上。确保教师的策略对于给定观测历史的学生是可实现的。
- **仿真到仿真迁移失败。** 如果你的策略对更难的仿真变体不鲁棒，它对真实世界也不会鲁棒。部署前务必在留出仿真变体上测试。
- **缺乏真实世界安全包络。** 一个在仿真中工作、"在真实中工作"的策略，如果没有底层安全护盾，仍可能损坏硬件。在非学习控制器中加入速率限制、扭矩限制、关节限制。

## 应用

2026 年仿真到现实技术栈：

| 领域 | 技术栈 |
|--------|--------|
| 足式运动（ANYmal、Spot、人形） | Isaac Lab + DR + 特权教师/学生 |
| 操作（灵巧手、抓取放置） | Isaac Lab + DR + 视觉 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线自适应 |
| 指尖/手中操作 | OpenAI Dactyl（史无前例规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

对于所有尺度的控制，工作流程是一致的：尽可能拟合仿真，对无法拟合的做随机化，训练巨大的策略，蒸馏，用安全护盾部署。

## 交付

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定打滑 GridWorld（slip=0.0）上训练 Q-learning 智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报 vs 打滑图。
2. **中等。** 训练一个采样 `slip ~ Uniform[0, 0.3]` 的 DR Q-learning 智能体。在相同的扫描上评估。在 slip=0.5（分布外）时，DR 能带来多少提升？
3. **困难。** 实现一个课程：从 slip=0.0 开始，每次策略达到最优的 90% 时扩大 DR 范围。测量达到 slip=0.3 零样本所需的总环境步数，与固定 DR 基线对比。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| 现实鸿沟 | "仿真到真实的差异" | 训练与部署之间物理/感知的分布偏移。 |
| 域随机化（DR） | "在随机仿真中训练" | 训练期间随机化仿真参数，使策略泛化。 |
| 系统辨识（SI） | "测量真实并拟合仿真" | 估计真实物理参数；设置仿真以匹配。 |
| 域自适应 | "用真实数据微调" | 仿真训练后少量真实世界微调；可能自适应观测或动力学。 |
| 特权信息 | "教师的真实值" | 只有仿真拥有的信息；学生必须从观测历史中推断。 |
| 教师/学生 | "蒸馏特权 -> 可观测" | 教师用捷径训练；学生学会在没有捷径的情况下模仿。 |
| ADR | "自动域随机化" | 随着策略提升而扩大 DR 范围的课程。 |
| Real2Sim | "用真实数据缩小差距" | 学习残差使仿真模仿真实 rollout。 |

## 延伸阅读

- [Tobin 等人 (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) —— 原始 DR 论文（机器人视觉）。
- [Peng 等人 (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) —— 动力学 DR，四足运动。
- [OpenAI 等人 (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) —— Dactyl，大规模 ADR。
- [Miki 等人 (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) —— ANYmal 的教师-学生方法。
- [Makoviychuk 等人 (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) —— 驱动 2025–2026 年部署的大规模并行仿真。
- [Akkaya 等人 (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) —— ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— Dyna 框架（用模型进行规划 + rollout），支撑现代仿真到现实流水线。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) —— 仿真到现实方法的分类及基准结果。
