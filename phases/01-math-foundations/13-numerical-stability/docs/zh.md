# 数值稳定性

> 浮点数是一种会泄漏的抽象。它会在训练过程中反咬你一口，而你却毫无防备。

**类型：** 构建
**语言：** Python
**前置知识：** 第一阶段，第 01-04 课
**时间：** ~120 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性抵消
- 使用中心有限差分验证解析梯度与数值梯度
- 解释为什么 bfloat16 比 float16 更适合训练，以及损失缩放如何防止梯度下溢

## 问题所在

你的模型训练了三小时，然后损失变成了 NaN。你添加了一条打印语句。在第 9,000 步时 logits 还是正常的。在第 9,001 步时它们变成了 `inf`。到第 9,002 步时，每个梯度都是 `nan`，训练彻底崩溃。

或者：你的模型训练完成了，但准确率比论文声称的低 2%。你检查了所有地方。架构匹配。超参数匹配。数据匹配。问题在于论文使用了 float32，而你使用了 float16 却没有正确的缩放。累积的三十二位舍入误差悄然吞噬了你的准确率。

或者：你从零实现了交叉熵损失。它在小的 logits 上工作正常。当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 大于 float32 能表示的范围。每个 ML 框架都用两行技巧来处理这个问题。你不知道这个技巧存在。

数值稳定性不是理论上的担忧。它是训练成功与静默失败之间的区别。你最终要调试的每个严重的 ML bug，归根结底都是浮点数问题。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。一个浮点数有三个部分：符号位、指数和尾数（有效数字）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度（有多少位有效数字）。指数决定范围（数字能有多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 提供约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但不能区分 1.00000001 和 1.00000002。超过 7 位之后，一切都是舍入噪声。

float16 提供约 3 位精度。它能表示的最大数字是 65,504。对于 ML 来说这小得惊人，因为 logits、梯度和激活值 routinely 超过这个值。

bfloat16 是 Google 对 float16 范围问题的回答。它有与 float32 相同的 8 位指数（相同范围，最大到 3.4e38），但只有 7 位尾数（比 float16 精度更低）。对于神经网络训练，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点中精确表示。在二进制中，它是一个无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 将其截断为 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 存储为约 0.200000002980232。它们的和是 0.300000004470348，不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. 像 `if loss < threshold` 这样的损失比较可能给出错误答案
2. 累积许多小值（数千步的梯度更新）会偏离真实总和
3. 如果用 `==` 比较浮点数，校验和与可复现性测试会失败

解决方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你减去两个几乎相等的浮点数时，有效数字相互抵消，剩下的舍入噪声被提升为前导数字。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

这就是单次减法带来的 19% 相对误差。在 ML 中，这种情况发生在：

- 计算具有大均值的数据方差：`E[x^2] - E[x]^2` 当 E[x] 很大时
- 减去几乎相等的对数概率
- 用过小的 epsilon 计算有限差分梯度

解决方法：重新排列公式以避免减去大的、几乎相等的数字。对于方差，使用 Welford 算法或先对数据进行中心化处理。对于对数概率，始终在对数空间中工作。

### 溢出和下溢

溢出发生在结果太大而无法表示时。下溢发生在结果太小（比最小的可表示正数更接近零）时。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是 ML 中溢出的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数则朝另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。组合 `log(exp(x))` 如果没有正确的技巧就是雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 会溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 下溢为零，`log(0)` 变成 `-inf`。

技巧：在取指数前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么这有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`。不可能溢出。求和中至少有一项是 1，所以总和至少为 1，`log(1) = 0`。不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

设 `c = max(x)`，溢出就被消除了。

这个技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有技巧时，[100, 101, 102] 的 logits 会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算是安全的。这不是优化。这是正确性的要求。

### NaN 和 Inf：检测与预防

`nan`（非数字）和 `inf`（无穷大）会像病毒一样在计算中传播。梯度更新中的一个 `nan` 会使权重变成 `nan`，进而使每个后续输出变成 `nan`。训练在一步之内就死了。

`inf` 如何出现：
- 大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- 累加中的 `float32` 溢出

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何涉及现有 `nan` 的运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 将输入限制到 `exp()`：`exp(clamp(x, -80, 80))`
2. 在分母中添加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部添加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 调试时在每次前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能存在 bug。数值梯度检查通过有限差分计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h²) 精度，比前向差分 `(f(x+h) - f(x)) / h` 好得多，后者只有 O(h)。

选择 h：太大则近似错误。太小则灾难性抵消破坏答案。`h = 1e-5` 到 `1e-7` 是典型的。

检查：计算解析梯度和数值梯度之间的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新层或损失函数时务必检查梯度。PyTorch 为此提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 有专门的硬件（Tensor Core），计算 float16 矩阵乘法比 float32 快 2-8 倍。混合精度训练利用这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。Float16 将低于 ~6e-8 的任何值下溢为零。你的模型停止学习，因为所有梯度更新都是零。

解决方法是损失缩放：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放自动调整缩放因子。从较大值（65536）开始。如果梯度溢出到 `inf`，则减半。如果 N 步没有溢出，则加倍。

### bfloat16 vs float16：为什么 bfloat16 在训练中胜出

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 有更多精度（10 位尾数 vs 7 位）但范围有限（最大 ~65,504）。bfloat16 精度较低，但范围与 float32 相同（最大 ~3.4e38）。

对于神经网络训练：

- 激活值和 logits 在训练峰值期间经常超过 65,504。float16 溢出；bfloat16 能处理。
- float16 需要损失缩放，但 bfloat16 通常不需要，因为它的范围覆盖了梯度大小范围。
- bfloat16 是 float32 的简单截断：丢弃尾数的低 16 位。转换是平凡的，且在指数上无损。

float16 更适合推理，因为值有界且精度更重要。bfloat16 更适合训练，因为范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16。

### 梯度裁剪

梯度爆炸发生在梯度通过多层指数增长时（在 RNN、深度网络和 transformer 中很常见）。单个大的梯度可以在一步内破坏所有权重。

两种裁剪类型：

**按值裁剪：** 独立地限制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的。它是标准选择。

典型值：transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

梯度裁剪不是权宜之计。它是安全机制。没有它，单个异常批次就可能产生足够大的梯度，毁掉数周的训练。

### 归一化层作为数值稳定器

批量归一化、层归一化和 RMS 归一化通常被介绍为帮助训练收敛的正则化器。它们也是数值稳定器。

没有归一化时，激活值可以通过层指数增长或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化在每层重新定中心和重新缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）防止所有激活值相同时除以零。学习到的参数 `gamma` 和 `beta` 让网络恢复它需要的任何尺度。

这使值在整个网络中保持在数值安全的范围内，防止前向传播中的溢出和后向传播中的梯度爆炸。

### 常见 ML 数值 Bug

**Bug：几个 epoch 后损失变成 NaN。**
原因：logits 变得太大，softmax 溢出。或者学习率太高，权重发散。
解决：使用稳定 softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：损失卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没有学习。
解决：检查数据标签是否正确，验证损失函数，检查死 ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：混合精度没有正确的损失缩放。梯度下溢静默地将小更新归零。
解决：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死 ReLU 神经元（所有输入为负），或 float16 下溢。
解决：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个 GPU 上工作，但在另一个上给出不同结果。**
原因：非确定性的浮点累加顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
解决：接受小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：`exp()` 在损失计算中返回 `inf`。**
原因：原始 logits 直接传给 `exp()` 而没有最大值减法技巧。
解决：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度大小或高于 65,504 的激活值。
解决：使用带损失缩放的混合精度（AMP），或使用 bfloat16。

## 动手实现

### 步骤 1：演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素 vs 稳定 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步骤 3：实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步骤 4：实现稳定的交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 应用实践

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

参见 `code/numerical.py` 获取完整实现，所有边界情况均已演示。

## 交付成果

本课程产出：
- `code/numerical.py`，包含稳定的 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现会在第三阶段构建训练循环和第四阶段实现注意力机制时再次出现。

## 练习题

1. **灾难性抵消。** 使用朴素公式 `E[x^2] - E[x]^2` 在 float32 中计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 在线算法计算。将误差与真实方差（0.6667）比较。

2. **精度探索。** 找到最小的正 float32 值 `x`，使得 `1.0 + x == 1.0` 在 Python 中成立。这就是机器 epsilon。验证它匹配 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp 边界情况。** 用以下情况测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值非常负（-1000）。验证它在朴素版本失败的地方给出正确结果。

4. **神经网络层的梯度检查。** 实现单个线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 验证 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建范围 [1e-9, 1e-3] 的随机梯度，转换为 float16，测量有多少比例变成零。然后应用损失缩放（乘以 1024），转换为 float16，缩放回，再次测量零的比例。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| 机器 epsilon | "精度极限" | 在给定浮点格式中，使得 1.0 + e != 1.0 的最小值 e。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 减去几乎相等的浮点数时，有效数字抵消，舍入噪声主导结果。 |
| 溢出 | "数字太大" | 结果超过最大可表示值，变成 inf。exp(89) 使 float32 溢出。 |
| 下溢 | "数字太小" | 结果比最小的可表示正数更接近零，变成 0.0。exp(-104) 使 float32 下溢。 |
| Log-sum-exp 技巧 | "先减去最大值" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，防止溢出和下溢。用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在取指数前减去 max(logits)。数值上完全相同的结果，不可能溢出。 |
| 梯度检查 | "验证你的反向传播" | 将反向传播的解析梯度与有限差分的数值梯度比较，以发现实现错误。 |
| 混合精度 | "Float16 前向，float32 后向" | 对速度关键的操作使用低精度浮点，对数值敏感的操作使用高精度浮点。典型加速比为 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播前将损失乘以一个大常数，使梯度保持在 float16 的可表示范围内，然后在权重更新前除以相同常数。 |
| bfloat16 | "Brain 浮点" | Google 的 16 位格式，8 位指数（与 float32 相同范围）和 7 位尾数（比 float16 精度低）。训练时优先使用。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量，使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数字" | 未定义操作产生的特殊浮点值（0/0、inf-inf、sqrt(-1)）。传播到所有后续运算中。 |
| Inf | "无穷大" | 溢出或除以零产生的特殊浮点值。可以组合产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。慢但可靠，用于验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，密集但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 引入 float16 训练损失缩放的论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为 TPU 选择此格式的原因
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和舍入误差的算法
