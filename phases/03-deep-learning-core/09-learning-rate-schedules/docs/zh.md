# 学习率调度与预热

> 学习率是单一最重要的超参数。不是架构，不是数据集大小，不是激活函数。是学习率。如果你什么都不调，就调这个。

**类型：** 构建
**语言：** Python
**前置知识：** 课程 03.06（优化器）、课程 03.08（权重初始化）
**时间：** ~90 分钟

## 学习目标

- 从零实现恒定、阶梯衰减、余弦退火、预热+余弦以及 1cycle 学习率调度
- 演示学习率选择的三种失效模式：发散（过高）、停滞（过低）和振荡（无衰减）
- 解释为什么预热对基于 Adam 的优化器是必需的，以及它如何稳定早期训练
- 在同一任务上比较五种调度的收敛速度，并根据给定的训练预算选择合适的调度

## 问题所在

将学习率设为 0.1。训练发散——损失在 3 步内跳到无穷大。设为 0.0001。训练爬行——100 个 epoch 后，模型几乎还在原地。设为 0.01。训练前 50 个 epoch 正常，然后损失围绕一个最小值振荡，永远到不了，因为步长太大。

最优学习率不是恒定的。它在训练过程中会变化。早期，你想要大步长来快速覆盖地面。训练后期，你想要小步长来 settle 进尖锐的最小值。90% 准确率的模型和 95% 准确率的模型之间的差别，往往就在于调度。

过去三年发布的每个主要模型都使用学习率调度。Llama 3 使用峰值 lr=3e-4，预热 2000 步，余弦衰减到 3e-5。GPT-3 使用 lr=6e-4，预热 3.75 亿个 token。这些不是随意选择。它们是花费数百万美元的广泛超参数搜索的结果。

你需要理解调度，因为默认设置对你的问题不适用。当你微调预训练模型时，合适的调度与从头训练不同。当你增大 batch size 时，预热周期需要改变。当训练在第 10,000 步崩溃时，你需要知道这是调度问题还是其他原因。

## 核心概念

### 恒定学习率

最简单的方法。选一个数，每一步都用。

```
lr(t) = lr_0
```

很少最优。它对训练末期来说要么太高（围绕最小值振荡），要么对初期来说太低（在小步长上浪费计算）。对小模型和调试来说还行。对任何训练超过一小时的东西都是糟糕选择。

### 阶梯衰减

ResNet 时代的老派方法。在固定 epoch 将学习率按一个因子（通常 10 倍）削减。

```
lr(t) = lr_0 * gamma^(floor(epoch / step_size))
```

其中 gamma = 0.1 且 step_size = 30 表示：每 30 个 epoch 学习率降低 10 倍。ResNet-50 使用了这个——lr=0.1，在 epoch 30、60 和 90 时各降 10 倍。

问题：最优衰减点取决于数据集和架构。换到不同问题，你需要重新调整何时衰减。过渡是突然的——学习率突然变化时损失可能飙升。

### 余弦退火

从最大学习率平滑衰减到最小值，遵循余弦曲线：

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

其中 t 是当前步数，T 是总步数。

在 t=0 时，余弦项为 1，所以 lr = lr_max。在 t=T 时，余弦项为 -1，所以 lr = lr_min。衰减初期平缓，中期加速，末期再次平缓。

这是大多数现代训练的默认选择。除了 lr_max 和 lr_min 之外没有超参数要调。余弦形状符合经验观察，即大部分学习发生在训练中期——你希望在那个关键阶段有合理的步长。

### 预热：为什么要从小开始

Adam 和其他自适应优化器维护梯度的均值和方差的运行估计。在第 0 步，这些估计初始化为零。最初几次梯度更新基于糟糕的统计量。如果这段时间学习率很大，模型会迈出巨大且方向错误的步伐。

预热解决这个问题。从极小的学习率开始（通常是 lr_max / warmup_steps 甚至零），在前 N 步线性 ramp 到 lr_max。当你达到完整学习率时，Adam 的统计量已经稳定。

```
lr(t) = lr_max * (t / warmup_steps)     for t < warmup_steps
```

典型预热：总训练步数的 1-5%。Llama 3 训练了约 1.8 万亿 token，预热 2000 步。GPT-3 预热了 3.75 亿个 token。

### 线性预热 + 余弦衰减

现代默认方案。线性 ramp 上升，然后余弦衰减：

```
if t < warmup_steps:
    lr(t) = lr_max * (t / warmup_steps)
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

这就是 Llama、GPT、PaLM 和大多数现代 transformer 使用的方案。预热防止早期不稳定。余弦衰减让模型 settle 进好的最小值。

### 1cycle 策略

Leslie Smith 的发现（2018）：在训练前半段将学习率从低值 ramp 到高值，后半段再 ramp 回来。反直觉——为什么要在中途*提高*学习率？

理论：高学习率通过给优化轨迹添加噪声来起到正则化作用。模型在 ramp-up 阶段探索更多损失 landscape，找到更好的 basin。然后 ramp-down 阶段在找到的最好的 basin 内 refine。

```
Phase 1 (0 to T/2):    lr ramps from lr_max/25 to lr_max
Phase 2 (T/2 to T):    lr ramps from lr_max to lr_max/10000
```

1cycle 在固定计算预算下通常比余弦退火训练更快。代价：你必须提前知道总步数。

### 调度形状

```mermaid
graph LR
    subgraph "Constant"
        C1["lr"] --- C2["lr"] --- C3["lr"]
    end

    subgraph "Step Decay"
        S1["0.1"] --- S2["0.1"] --- S3["0.01"] --- S4["0.001"]
    end

    subgraph "Cosine Annealing"
        CS1["lr_max"] --> CS2["gradual"] --> CS3["steep"] --> CS4["lr_min"]
    end

    subgraph "Warmup + Cosine"
        WC1["0"] --> WC2["lr_max"] --> WC3["cosine"] --> WC4["lr_min"]
    end
```

### 决策流程图

```mermaid
flowchart TD
    Start["Choosing a LR schedule"] --> Know{"Know total<br/>training steps?"}

    Know -->|"Yes"| Budget{"Compute budget?"}
    Know -->|"No"| Constant["Use constant LR<br/>with manual decay"]

    Budget -->|"Large (days/weeks)"| WarmCos["Warmup + Cosine Decay<br/>(Llama/GPT default)"]
    Budget -->|"Small (hours)"| OneCycle["1cycle Policy<br/>(fastest convergence)"]
    Budget -->|"Moderate"| Cosine["Cosine Annealing<br/>(safe default)"]

    WarmCos --> Warmup["Warmup = 1-5% of steps"]
    OneCycle --> FindLR["Find lr_max with LR range test"]
    Cosine --> MinLR["Set lr_min = lr_max / 10"]
```

### 已发布模型的真实数据

```mermaid
graph TD
    subgraph "Published LR Configs"
        L3["Llama 3 (405B)<br/>Peak: 3e-4<br/>Warmup: 2000 steps<br/>Schedule: Cosine to 3e-5"]
        G3["GPT-3 (175B)<br/>Peak: 6e-4<br/>Warmup: 375M tokens<br/>Schedule: Cosine to 0"]
        R50["ResNet-50<br/>Peak: 0.1<br/>Warmup: none<br/>Schedule: Step decay x0.1 at 30,60,90"]
        B["BERT (340M)<br/>Peak: 1e-4<br/>Warmup: 10K steps<br/>Schedule: Linear decay"]
    end
```

## 动手实现

### 步骤 1：调度函数

每个函数接收当前步数，返回该步的学习率。

```python
import math


def constant_schedule(step, lr=0.01, **kwargs):
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=1e-5, **kwargs):
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress
```

### 步骤 2：可视化所有调度

打印基于文本的图表，展示每种调度在训练过程中的变化。

```python
def visualize_schedule(name, schedule_fn, total_steps=500, **kwargs):
    steps = list(range(0, total_steps, total_steps // 20))
    if total_steps - 1 not in steps:
        steps.append(total_steps - 1)

    lrs = [schedule_fn(s, total_steps=total_steps, **kwargs) for s in steps]
    max_lr = max(lrs) if max(lrs) > 0 else 1.0

    print(f"\n{name}:")
    for s, lr_val in zip(steps, lrs):
        bar_len = int(lr_val / max_lr * 40)
        bar = "#" * bar_len
        print(f"  Step {s:4d}: lr={lr_val:.6f} {bar}")
```

### 步骤 3：训练网络

一个简单的两层网络，使用圆形数据集，与之前课程相同，但这次我们改变调度。

```python
import random


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return max(0.0, x)


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def train_with_schedule(schedule_fn, schedule_name, data, epochs=300, base_lr=0.05, **kwargs):
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)

    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, target in data:
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)

            z1 = []
            h = []
            for i in range(hidden_size):
                z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                z1.append(z)
                h.append(relu(z))

            z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)

            error = out - target
            d_out = error * out * (1 - out)

            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out

            total_loss += (out - target) ** 2
            if (out >= 0.5) == (target >= 0.5):
                correct += 1
            step += 1

        avg_loss = total_loss / len(data)
        accuracy = correct / len(data) * 100
        epoch_losses.append(avg_loss)

    return epoch_losses
```

### 步骤 4：比较所有调度

用每种调度训练同一个网络，比较最终损失和收敛行为。

```python
def compare_schedules(data):
    configs = [
        ("Constant", constant_schedule, {}),
        ("Step Decay", step_decay_schedule, {"step_size": 15000, "gamma": 0.1}),
        ("Cosine", cosine_schedule, {"lr_min": 1e-5}),
        ("Warmup+Cosine", warmup_cosine_schedule, {"warmup_steps": 3000, "lr_min": 1e-5}),
        ("1cycle", one_cycle_schedule, {}),
    ]

    print(f"\n{'Schedule':<20} {'Start Loss':>12} {'Mid Loss':>12} {'End Loss':>12} {'Best Loss':>12}")
    print("-" * 70)

    for name, schedule_fn, extra_kwargs in configs:
        losses = train_with_schedule(schedule_fn, name, data, epochs=300, base_lr=0.05, **extra_kwargs)
        mid_idx = len(losses) // 2
        best = min(losses)
        print(f"{name:<20} {losses[0]:>12.6f} {losses[mid_idx]:>12.6f} {losses[-1]:>12.6f} {best:>12.6f}")
```

### 步骤 5：学习率过高 vs 过低

演示三种失效模式：过高（发散）、过低（爬行）和刚好。

```python
def lr_sensitivity(data):
    learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]

    print("\nLR Sensitivity (constant schedule, 100 epochs):")
    print(f"  {'LR':>10} {'Start Loss':>12} {'End Loss':>12} {'Status':>15}")
    print("  " + "-" * 52)

    for lr in learning_rates:
        losses = train_with_schedule(constant_schedule, f"lr={lr}", data, epochs=100, base_lr=lr)
        start = losses[0]
        end = losses[-1]

        if end > start or math.isnan(end) or end > 1.0:
            status = "DIVERGED"
        elif end > start * 0.9:
            status = "BARELY MOVED"
        elif end < 0.15:
            status = "CONVERGED"
        else:
            status = "LEARNING"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>12.6f} {end_str:>12} {status:>15}")
```

## 实际应用

PyTorch 在 `torch.optim.lr_scheduler` 中提供了调度器：

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

for step in range(1000):
    loss = train_step(model, optimizer)
    scheduler.step()
```

对于预热 + 余弦，使用 lambda 调度器或 HuggingFace 的 `get_cosine_schedule_with_warmup`：

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
)
```

HuggingFace 函数是大多数 Llama 和 GPT 微调脚本使用的。不确定时，使用预热 + 余弦，预热步数 = 总步数的 3-5%。它几乎适用于所有情况。

## 交付成果

本课程产出：
- `outputs/prompt-lr-schedule-advisor.md` —— 一个为你的训练配置推荐合适学习率调度和超参数的 prompt

## 练习

1. 实现指数衰减：lr(t) = lr_0 * gamma^t，其中 gamma = 0.999。在圆形数据集上与余弦退火比较。

2. 实现学习率范围测试（Leslie Smith）：训练几百步，同时将 LR 从 1e-7 指数增加到 1。绘制损失 vs LR 的图。最优最大 LR 就在损失开始增加之前。

3. 用预热 + 余弦训练，但改变预热长度：总步数的 0%、1%、5%、10%、20%。找到训练最稳定的最佳点。

4. 实现带热重启的余弦退火（SGDR）：每 T 步将学习率重置为 lr_max 并再次衰减。在更长的训练运行中与标准余弦比较。

5. 构建一个"调度外科医生"，监控训练损失，在损失稳定时自动从预热切换到余弦，如果损失停滞太久则降低 lr。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 学习率 | "模型学习的速度" | 乘以梯度以确定参数更新大小的标量 |
| 调度 | "随时间改变 LR" | 将训练步数映射到学习率的函数，旨在优化收敛 |
| 预热 | "从小 LR 开始" | 在前 N 步将 LR 从接近零线性 ramp 到目标值，以稳定优化器统计量 |
| 余弦退火 | "平滑 LR 衰减" | 按照余弦曲线从 lr_max 衰减到 lr_min |
| 阶梯衰减 | "在里程碑处降低 LR" | 在固定 epoch 间隔将 LR 乘以一个因子（通常 0.1） |
| 1cycle 策略 | "先升后降" | Leslie Smith 的方法，在一个周期内 ramp LR 上升再下降以实现更快收敛 |
| LR 范围测试 | "找到最佳学习率" | 短暂训练同时增加 LR，找到损失开始发散的值 |
| 带热重启的余弦 | "重置并重复" | 周期性将 LR 重置为 lr_max 并再次衰减（SGDR） |
| Eta min | "LR 的下限" | 调度衰减到的最小学习率 |
| 峰值学习率 | "最大 LR" | 训练期间达到的最高学习率，通常在预热之后 |

## 延伸阅读

- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017) —— 引入余弦退火和热重启
- Smith, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (2018) —— 1cycle 策略论文
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) —— 记录大规模使用的预热 + 余弦调度
- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017) —— 大 batch 训练的线性缩放规则和预热
