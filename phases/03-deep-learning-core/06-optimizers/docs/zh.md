# 优化器

> 梯度下降告诉你朝哪个方向走。但它没说走多远、走多快。SGD 是指南针。Adam 是带交通数据的 GPS。

**类型：** Build
**语言：** Python
**前置知识：** 课程 03.05（损失函数）
**时间：** ~75 分钟

## 学习目标

- 用 Python 从零实现 SGD、带动量的 SGD、Adam 和 AdamW 优化器
- 解释 Adam 的偏差修正如何在训练早期补偿零初始化的矩估计
- 演示为什么 AdamW 在相同任务上比带 L2 正则化的 Adam 产生更好的泛化效果
- 为 transformers、CNN、GAN 和微调任务选择合适的优化器及默认超参数

## 问题所在

你计算出了梯度。你知道权重 #4,721 应该减少 0.003 来降低损失。但 0.003 是什么单位？按什么比例缩放？第 1 步和第 1,000 步应该移动相同的量吗？

普通梯度下降对每一步的每个参数应用相同的学习率：w = w - lr * gradient。这在实践中造成了三个让神经网络训练变得痛苦的问题。

第一，振荡。损失景观很少是光滑的碗状。它更像一条狭长山谷。梯度指向山谷的"对岸"（陡峭方向），而不是沿着山谷（平缓方向）。梯度下降在狭窄维度上来回弹跳，而在有用的方向上进展甚微。你见过这种情况：损失快速下降然后停滞，不是因为模型收敛了，而是因为它在振荡。

第二，所有参数共享一个学习率是错误的。有些权重需要大幅更新（它们处于早期、欠拟合阶段）。有些只需要微小更新（它们已接近最优值）。适合前者的学习率会毁掉后者，反之亦然。

第三，鞍点。在高维空间中，损失景观存在广阔的平坦区域，梯度接近零。普通 SGD 以梯度的速度在这些区域爬行，而梯度实际上为零。模型看起来卡住了。它没有卡住——它处于平坦区域，另一侧有有用的下降方向。但 SGD 没有机制来推动它前进。

Adam 解决了所有这三个问题。它为每个参数维护两个运行平均值——平均梯度（动量，处理振荡）和平均平方梯度（自适应速率，处理不同尺度）。结合对最初几步的偏差修正，它提供了一个在 80% 的问题上使用默认超参数就能工作的单一优化器。本课程从零构建它，让你准确理解它在另外 20% 的情况下何时以及为什么会失败。

## 核心概念

### 随机梯度下降（SGD）

最简单的优化器。在小批量数据上计算梯度，朝相反方向迈步。

```
w = w - lr * gradient
```

"随机"意味着你使用数据的随机子集（小批量）来估计梯度，而不是完整数据集。这种噪声实际上是有用的——它有助于逃离尖锐的局部最小值。但噪声也会导致振荡。

学习率是唯一的调节旋钮。太高：损失发散。太低：训练永远进行不完。最优值取决于架构、数据、批量大小和训练当前阶段。对于现代网络上的普通 SGD，典型值范围从 0.01 到 0.1。但即使在单次训练中，理想的学习率也会变化。

### 动量

小球滚下山坡的类比被过度使用但确实准确。不是仅按梯度迈步，而是维护一个累积过去梯度的速度。

```
m_t = beta * m_{t-1} + gradient
w = w - lr * m_t
```

Beta（通常为 0.9）控制保留多少历史信息。当 beta = 0.9 时，动量大致是最近 10 个梯度的平均值（1 / (1 - 0.9) = 10）。

为什么这能修复振荡：指向相同方向的梯度会累积。方向翻转的梯度会相互抵消。在那个狭窄山谷中，"横跨"分量每步都变号并被衰减。"沿谷"分量保持一致并被放大。结果是沿有用方向平滑加速。

真实数据：在条件恶劣的损失景观上，单独使用 SGD 可能需要 10,000 步。带动量的 SGD（beta=0.9）在相同问题上通常需要 3,000-5,000 步。加速效果不是边际的。

### RMSProp

第一个实际有效的逐参数自适应学习率方法。由 Hinton 在 Coursera 讲座中提出（从未正式发表）。

```
s_t = beta * s_{t-1} + (1 - beta) * gradient^2
w = w - lr * gradient / (sqrt(s_t) + epsilon)
```

s_t 跟踪平方梯度的运行平均值。梯度持续较大的参数会被除以较大的数（有效学习率更小）。梯度较小的参数会被除以较小的数（有效学习率更大）。

这解决了"一个学习率适用于所有参数"的问题。一个已经获得大幅更新的权重可能已接近目标——减慢它。一个一直获得微小更新的权重可能训练不足——加快它。

Epsilon（通常为 1e-8）防止在参数尚未更新时除以零。

### Adam：动量 + RMSProp

Adam 结合了两种思想。它为每个参数维护两个指数移动平均：

```
m_t = beta1 * m_{t-1} + (1 - beta1) * gradient        (first moment: mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2       (second moment: variance)
```

**偏差修正** 是大多数解释都会跳过的关键细节。在第 1 步，m_1 = (1 - beta1) * gradient。当 beta1 = 0.9 时，这是 0.1 * gradient——太小了十倍。移动平均尚未"预热"。偏差修正进行补偿：

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

在第 1 步且 beta1 = 0.9 时：m_hat = m_1 / (1 - 0.9) = m_1 / 0.1 = 实际梯度。在第 100 步：(1 - 0.9^100) 约等于 1.0，因此修正消失。偏差修正在前 ~10 步很重要，在 ~50 步后无关紧要。

更新公式：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

Adam 默认值：lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8。这些默认值适用于 80% 的问题。当不适用时，先改 lr。然后改 beta2。几乎不需要改 beta1 或 epsilon。

### AdamW：正确的权重衰减

L2 正则化向损失添加 lambda * w^2。在普通 SGD 中，这等价于权重衰减（每步从权重中减去 lambda * w）。在 Adam 中，这种等价性被打破。

Loshchilov & Hutter 的洞见：当你将 L2 加入损失，然后 Adam 处理梯度时，自适应学习率也会缩放正则化项。梯度方差大的参数获得更少的正则化。方差小的参数获得更多。这不是你想要的——你希望无论梯度统计如何，正则化都是均匀的。

AdamW 通过在 Adam 更新后直接对权重应用权重衰减来修复这个问题：

```
w = w - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * lambda * w
```

权重衰减项（lr * lambda * w）不会被 Adam 的自适应因子缩放。每个参数获得相同比例的收缩。

这看起来是个小细节。它不是。AdamW 在几乎所有任务上都比 Adam + L2 正则化收敛到更好的解。它是 PyTorch 中训练 transformers、扩散模型和大多数现代架构的默认优化器。BERT、GPT、LLaMA、Stable Diffusion——都用 AdamW 训练。

### 学习率：最重要的超参数

```mermaid
graph TD
    LR["Learning Rate"] --> TooHigh["Too high (lr > 0.01)"]
    LR --> JustRight["Just right"]
    LR --> TooLow["Too low (lr < 0.00001)"]

    TooHigh --> Diverge["Loss explodes<br/>NaN weights<br/>Training crashes"]
    JustRight --> Converge["Loss decreases steadily<br/>Reaches good minimum<br/>Generalizes well"]
    TooLow --> Stall["Loss decreases slowly<br/>Gets stuck in suboptimal minimum<br/>Wastes compute"]

    JustRight --> Schedule["Usually needs scheduling"]
    Schedule --> Warmup["Warmup: ramp from 0 to max<br/>First 1-10% of training"]
    Schedule --> Decay["Decay: reduce over time<br/>Cosine or linear"]
```

如果你只调一个超参数，就调学习率。学习率 10 倍的变化比任何架构决策都重要。常见默认值：

- SGD: lr = 0.01 到 0.1
- Adam/AdamW: lr = 1e-4 到 3e-4
- 微调预训练模型: lr = 1e-5 到 5e-5
- 学习率预热：在前 1-10% 的步数上线性 ramp 到 max_lr

### 优化器对比

```mermaid
flowchart LR
    subgraph "Optimization Path"
        SGD_P["SGD<br/>Oscillates across valley<br/>Slow but finds flat minima"]
        Mom_P["SGD + Momentum<br/>Smoother path<br/>3x faster than SGD"]
        Adam_P["Adam<br/>Adapts per-parameter<br/>Fast convergence"]
        AdamW_P["AdamW<br/>Adam + proper decay<br/>Best generalization"]
    end
    SGD_P --> Mom_P --> Adam_P --> AdamW_P
```

### 每种优化器的优势场景

```mermaid
flowchart TD
    Task["What are you training?"] --> Type{"Model type?"}

    Type -->|"Transformer / LLM"| AdamW["AdamW<br/>lr=1e-4, wd=0.01-0.1"]
    Type -->|"CNN / ResNet"| SGD_M["SGD + Momentum<br/>lr=0.1, momentum=0.9"]
    Type -->|"GAN"| Adam2["Adam<br/>lr=2e-4, beta1=0.5"]
    Type -->|"Fine-tuning"| AdamW2["AdamW<br/>lr=2e-5, wd=0.01"]
    Type -->|"Don't know yet"| Default["Start with AdamW<br/>lr=3e-4, wd=0.01"]
```

## 动手实现

### 步骤 1：普通 SGD

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
```

### 步骤 2：带动量的 SGD

```python
class SGDMomentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None

    def step(self, params, grads):
        if self.velocities is None:
            self.velocities = [0.0] * len(params)
        for i in range(len(params)):
            self.velocities[i] = self.beta * self.velocities[i] + grads[i]
            params[i] -= self.lr * self.velocities[i]
```

### 步骤 3：Adam

```python
import math

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
```

### 步骤 4：AdamW

```python
class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            params[i] -= self.lr * self.weight_decay * params[i]
```

### 步骤 5：训练对比

用课程 05 中的圆形数据集，用同一个两层网络配合四种优化器进行训练。比较收敛情况。

```python
import random

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class OptimizerTestNetwork:
    def __init__(self, optimizer, hidden_size=8):
        random.seed(0)
        self.hidden_size = hidden_size
        self.optimizer = optimizer

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def set_params(self, params):
        idx = 0
        for i in range(self.hidden_size):
            for j in range(2):
                self.w1[i][j] = params[idx]
                idx += 1
        for i in range(self.hidden_size):
            self.b1[i] = params[idx]
            idx += 1
        for i in range(self.hidden_size):
            self.w2[i] = params[idx]
            idx += 1
        self.b2 = params[idx]

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def compute_grads(self, target):
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        grads = [0.0] * (self.hidden_size * 2 + self.hidden_size + self.hidden_size + 1)
        idx = 0
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_out * self.w2[i] * d_relu
            grads[idx] = d_h * self.x[0]
            grads[idx + 1] = d_h * self.x[1]
            idx += 2

        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            grads[idx] = d_out * self.w2[i] * d_relu
            idx += 1

        for i in range(self.hidden_size):
            grads[idx] = d_out * self.h[i]
            idx += 1

        grads[idx] = d_out
        return grads

    def train(self, data, epochs=300):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                grads = self.compute_grads(y)
                params = self.get_params()
                self.optimizer.step(params, grads)
                self.set_params(params)

                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append((avg_loss, accuracy))
            if epoch % 75 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses
```

## 实际应用

PyTorch 优化器处理参数组、梯度裁剪和学习率调度：

```python
import torch
import torch.optim as optim

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(torch.randn(32, 784))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 10, (32,)))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

模式永远是：zero_grad, forward, loss, backward, (clip), step, (schedule)。记住这个顺序。搞错它（例如，在 optimizer.step() 之前调用 scheduler.step()）是微妙 bug 的常见来源。

对于 CNN，许多从业者仍偏好 SGD + 动量（lr=0.1, momentum=0.9, weight_decay=1e-4）配合步进或余弦调度。SGD 找到更平坦的最小值，往往泛化更好。对于 transformers 和 LLM，AdamW 配合预热 + 余弦衰减是通用默认。没有实测理由不要违背这个共识。

## 产出

本课程产出：
- `outputs/prompt-optimizer-selector.md` —— 一个为任何架构选择合适优化器和学习率的决策提示

## 练习

1. 实现 Nesterov 动量，即在"前瞻"位置（w - lr * beta * v）而非当前位置计算梯度。在圆形数据集上将其与标准动量的收敛情况进行对比。

2. 实现学习率预热调度：在前 10% 的训练步数内从 0 线性 ramp 到 max_lr，然后余弦衰减到 0。用 Adam + 预热与无预热的 Adam 训练。测量达到圆形数据集 90% 准确率需要多少 epoch。

3. 在 Adam 训练期间跟踪每个参数的有效学习率。有效率为 lr * m_hat / (sqrt(v_hat) + eps)。绘制 10、50 和 200 步后有效率的分布。所有参数都以相同速度更新吗？

4. 实现梯度裁剪（按全局范数裁剪）。将最大梯度范数设为 1.0。使用高学习率（Adam 的 lr=0.01）进行有裁剪和无裁剪的训练。统计在 10 个随机种子下，有裁剪和无裁剪时有多少次运行发散（损失变为 NaN）。

5. 在大权重网络上对比 Adam 与 AdamW。将所有权重初始化为 [-5, 5] 中的随机值（远大于正常值）。用 weight_decay=0.1 训练 200 个 epoch。绘制两种优化器训练过程中权重的 L2 范数。AdamW 应显示更快的权重收缩。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Learning rate | "步长" | 梯度更新上的标量乘数；训练中影响最大的单一超参数 |
| SGD | "基础梯度下降" | 随机梯度下降：在小批量上计算 lr * gradient 并从权重中减去来更新 |
| Momentum | "滚球类比" | 过去梯度的指数移动平均；衰减振荡并加速一致方向 |
| RMSProp | "自适应学习率" | 将每个参数的梯度除以其近期梯度的运行 RMS；均衡学习率 |
| Adam | "默认优化器" | 结合动量（一阶矩）和 RMSProp（二阶矩），并对初始步骤进行偏差修正 |
| AdamW | "正确的 Adam" | 解耦权重衰减的 Adam；直接对权重应用正则化而非通过梯度 |
| Bias correction | "运行平均的预热" | 除以 (1 - beta^t) 以补偿 Adam 矩估计的零初始化 |
| Weight decay | "收缩权重" | 每步减去权重值的一个分数；惩罚大权重的正则化器 |
| Learning rate schedule | "随时间改变 lr" | 在训练期间调整学习率的函数；预热 + 余弦衰减是现代默认 |
| Gradient clipping | "限制梯度范数" | 当梯度向量的范数超过阈值时将其缩小；防止梯度爆炸更新 |

## 延伸阅读

- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014) —— 原始 Adam 论文，包含收敛分析和偏差修正推导
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017) —— 证明 L2 正则化和权重衰减在 Adam 中不等价，并提出 AdamW
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) —— 引入 LR 范围测试和循环调度，消除调优固定学习率的需要
- Ruder, "An Overview of Gradient Descent Optimization Algorithms" (2016) —— 对所有优化器变体最全面的单一综述，包含清晰的对比和直观解释
