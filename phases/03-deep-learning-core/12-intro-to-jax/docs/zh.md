# JAX 简介

> PyTorch 会修改张量。TensorFlow 会构建计算图。JAX 会编译纯函数。最后一种方式会改变你对深度学习的理解。

**类型：** 构建
**语言：** Python
**前置条件：** 阶段 03 第 01-10 课，基础 NumPy
**时间：** ~90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的即时求值与 JAX 的函数式编译模型之间的关键设计差异
- 应用 JIT 编译和 vmap 向量化来加速训练循环，与朴素 Python 实现进行对比
- 在 JAX 中训练简单网络，并将其显式状态管理与 PyTorch 的面向对象方法进行对比

## 问题所在

你已经知道如何在 PyTorch 中构建神经网络。你定义一个 `nn.Module`，调用 `.backward()`，执行优化器步进。它能工作。数百万人在使用它。

但 PyTorch 有一个深植于其 DNA 中的限制：它在 Python 中逐条追踪操作，一次执行一个。每个 `tensor + tensor` 都是一次独立的内核启动。每个训练步骤都重新解释相同的 Python 代码。这在需要训练之前都没问题，直到你需要在 2,048 个 TPU 上训练一个 5400 亿参数的模型。那时，开销会拖垮你。

Google DeepMind 使用 JAX 训练 Gemini。Anthropic 使用 JAX 训练 Claude。这些都不是小规模操作——它们是地球上规模最大的神经网络训练任务。他们选择 JAX，是因为 JAX 将训练循环视为可编译的程序，而不是一系列 Python 调用。

JAX 是 NumPy 加上三个超能力：自动微分、JIT 编译到 XLA，以及自动向量化。你编写一个处理单个样本的函数。JAX 给你一个能处理批次、计算梯度、编译为机器代码并在多个设备上运行的函数。而且不需要修改原始函数。

## 核心概念

### JAX 的设计哲学

JAX 是一个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` 带状态的类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 即时执行 | 通过 XLA 进行 JIT 编译 |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变的 `model.parameters()` | 不可变的数组 pytree |

这不是风格偏好。这是编译器的约束。JIT 编译需要纯函数——相同的输入总是产生相同的输出，没有副作用。正是这个限制使得 100 倍加速成为可能。

### jax.numpy：熟悉的接口

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名。相同的广播规则。相同的切片语义。但数组驻留在 GPU/TPU 上，且每个操作都可以被编译器追踪。

一个关键区别：JAX 数组是不可变的。没有 `a[0] = 5`。取而代之的是：`a = a.at[0].set(5)`。一开始会觉得别扭，但一周后就会豁然开朗——不可变性正是 `grad`、`jit` 和 `vmap` 可组合的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到张量上（`.grad`）。JAX 将梯度附加到函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数并返回一个计算梯度的新函数。没有 `.backward()` 调用。没有存储在张量上的计算图。梯度只是另一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。雅可比矩阵。海森矩阵。全部通过组合 `grad` 实现。PyTorch 也能做到（`torch.autograd.functional.hessian`），但那是后来添加的。在 JAX 中，这是基础。

约束条件：`grad` 只对纯函数有效。内部不能有 print 语句（它们在追踪时运行，而非执行时）。不能修改外部状态。没有显式密钥管理就不能生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

在第一次调用时，JAX 追踪函数——它记录发生了哪些操作，但不执行它们。然后将该追踪结果交给 XLA（Accelerated Linear Algebra），Google 为 TPU 和 GPU 打造的编译器。XLA 融合操作、消除冗余内存拷贝，并生成优化的机器代码。

后续调用完全跳过 Python。编译后的代码以 C++ 速度在加速器上运行。

JIT 适用场景：
- 训练步骤（相同的计算重复数千次）
- 推理（相同的模型，不同的输入）
- 任何使用相似形状输入被调用多次的函数

JIT 不适用场景：
- 包含依赖于值的 Python 控制流的函数（`if x > 0`，其中 x 是被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪隐藏了实际执行）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` 循环。这些不是可选的——它们是编译的代价。

### vmap：自动向量化

你编写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升为处理批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不要对 `params` 进行批处理（共享的），对 `x` 的第 0 轴进行批处理。没有手动的 `for` 循环。没有 reshape。没有批次维度的传递。JAX 自动找出批次维度并对整个计算进行向量化。

这不是语法糖。`vmap` 生成融合的向量化代码，比 Python 循环快 10-100 倍。而且它可以与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度。一行代码。这在 PyTorch 中几乎不可能实现，除非使用 hack。

### pmap：跨设备的数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 在所有可用设备（GPU/TPU）上复制函数并分割批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 跨设备同步梯度。

Google 使用 `pmap`（及其后继者 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型是：编写单设备版本，用 `pmap` 包装，完成。

### Pytrees：通用数据结构

JAX 操作"pytrees"——列表、元组、字典和数组的嵌套组合。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 将 `f` 应用到每个叶子节点。这就是优化器如何一次性更新所有参数：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数被传入。没有东西被存储。没有东西被修改。这使得每个函数都可测试、可组合、可编译。这也意味着你自己管理参数——或者使用 Flax 或 Equinox 这样的库。

### JAX 生态系统

JAX 提供原语。库提供易用性：

| 库 | 作用 | 风格 |
|---------|------|-------|
| **Flax** (Google) | 神经网络层 | `nn.Module` 带显式状态 |
| **Equinox** (Patrick Kidger) | 神经网络层 | 基于 Pytree，Pythonic |
| **Optax** (DeepMind) | 优化器 + 学习率调度 | 可组合的梯度变换 |
| **Orbax** (Google) | 检查点保存 | 保存/恢复 pytrees |
| **CLU** (Google) | 指标 + 日志 | 训练循环工具 |

Optax 是标准的优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，使得组合变得简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX vs PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一等公民（Google 构建了两者） | 社区维护（torch_xla） |
| GPU 支持 | 良好（通过 XLA 的 CUDA） | 最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 简单（即时执行，逐行） |
| 生态系统 | 研究导向（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（无处不在） |
| 大规模训练 | 优秀（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型速度 | 较慢（函数式开销） | 较快（修改即走） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

诚实的答案：除非你有特定理由使用 JAX，否则使用 PyTorch。这些理由是——TPU 访问、需要逐样本梯度、大规模多设备训练，或在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要显式的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这很烦人。但它保证了跨设备和跨编译的可复现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的属性。

## 动手实现

### 步骤 1：设置和数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐藏层分别为 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤 2：初始化参数

没有类。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 初始化，手动完成。三个 PRNG 密钥从一个种子分裂而来。每个权重都是嵌套字典中的不可变数组。

### 步骤 3：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数进，预测出。没有 `self`，没有存储状态。`loss_fn` 从头计算交叉熵——softmax、log、负均值。

### 步骤 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 一次性返回损失值和梯度。`@jax.jit` 装饰器将两个函数都编译到 XLA。第一次调用后，每个训练步骤都不接触 Python 运行。

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch。~97% 测试准确率。第一个 epoch 很慢（JIT 编译）。第 2-10 个 epoch 很快。

注意缺少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合函数调用。梯度被计算、被 Adam 变换、应用到参数——全部在 `train_step` 内部。

## 实际应用

### Flax：Google 标准

Flax 是最常用的 JAX 神经网络库。它重新引入了 `nn.Module`，但带有显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型分离。`model.init()` 创建参数。`model.apply(params, x)` 运行前向传播。模型对象没有状态。

### Equinox：Pythonic 的替代方案

Equinox（由 Patrick Kidger 开发）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一个 pytree。不需要 `.apply()`。参数就是模型的叶子节点。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度变换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——全部组合为变换链。每个变换看到梯度，修改它们，传递给下一个。没有单一的优化器类。

## 部署上线

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用很慢（编译）。基准测试前先预热。
- 避免在 JIT 内部对 JAX 数组使用 Python 循环。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部工作。普通的 `print()` 不行。
- 使用 `jax.profiler` 或 TensorBoard 进行性能分析。XLA 编译可能隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 来禁用。

**检查点保存：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` —— 选择正确 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md` —— 涵盖 JAX 函数式模式的技能

## 练习

1. 为 MLP 添加 dropout。在 JAX 中，dropout 需要 PRNG 密钥——将密钥贯穿前向传播，并为每个 dropout 层分裂密钥。对比有和没有 dropout 的测试准确率。

2. 使用 `jax.vmap` 计算 32 张 MNIST 图像批次的逐样本梯度。计算每个样本的梯度范数。哪些样本的梯度最大，为什么？

3. 将手动前向函数替换为适用于任意层数的通用 `mlp_forward(params, x)`。使用 `jax.tree.leaves` 自动确定深度。

4. 对有和没有 `@jax.jit` 的训练步骤进行基准测试。各计时 100 步。在你的硬件上加速比是多少？第一次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。在有和没有裁剪的情况下训练。绘制训练过程中的梯度范数以观察效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| XLA | "让 JAX 变快的东西" | Accelerated Linear Algebra——一个编译器，融合操作并从计算图生成优化的 GPU/TPU 内核 |
| JIT | "即时编译" | JAX 在第一次调用时追踪函数，编译到 XLA，然后在后续调用中运行编译版本 |
| 纯函数 | "没有副作用" | 输出仅依赖于输入的函数——没有全局状态，没有修改，没有显式密钥就不产生随机性 |
| vmap | "自动批处理" | 将处理单个样本的函数转换为处理批次的函数，无需重写 |
| pmap | "自动并行" | 在多个设备上复制函数并分割输入批次 |
| Pytree | "嵌套的字典数组" | 任何列表、元组、字典和数组的嵌套结构，JAX 可以遍历和变换 |
| 追踪 | "记录计算" | JAX 用抽象值执行函数以构建计算图，不计算真实结果 |
| 函数式自动微分 | "函数的梯度" | 通过变换函数而非将梯度存储附加到张量上来计算导数 |
| Optax | "JAX 的优化器库" | 可组合的梯度变换库——Adam、SGD、裁剪、调度——它们链式组合 |
| Flax | "JAX 的 nn.Module" | Google 的 JAX 神经网络库，添加层抽象同时保持状态显式 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) —— 解释设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 的 JAX 神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) —— Flax 的 Pythonic 替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" —— 标准优化器库
- "You Don't Know JAX" (Colin Raffel, 2020) —— 来自 T5 作者之一的 JAX 陷阱和模式实用指南
