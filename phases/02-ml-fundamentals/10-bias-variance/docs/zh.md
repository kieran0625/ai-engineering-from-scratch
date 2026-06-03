# 偏差-方差权衡

> 每个模型误差都来自三个来源之一：偏差、方差或噪声。你只能控制前两个。

**类型：** 学习
**语言：** Python
**前置知识：** 第 2 阶段，第 01-09 课（机器学习基础、回归、分类、评估）
**时间：** ~75 分钟

## 学习目标

- 推导期望预测误差的偏差-方差分解，并解释不可约噪声的作用
- 利用训练误差和测试误差的模式诊断模型是高偏差还是高方差
- 解释正则化技术（L1、L2、dropout、早停）如何以偏差换取方差
- 实现实验，可视化模型复杂度增加时的偏差-方差权衡

## 问题所在

你训练了一个模型。它在测试数据上有一些误差。这些误差从何而来？

如果你的模型太简单（用线性回归拟合曲线数据集），它会持续地偏离真实模式。这就是偏差。如果你的模型太复杂（用 20 次多项式拟合 15 个数据点），它会完美拟合训练数据，但在新数据上给出 wildly different 的预测。这就是方差。

对于固定的模型容量，你无法同时最小化两者。压低偏差，方差就会上升。压低方差，偏差就会上升。理解这一权衡是机器学习中最有用的诊断技能。它告诉你应该让模型更复杂还是更简单，应该获取更多数据还是设计更好的特征，应该更多还是更少地进行正则化。

## 核心概念

### 偏差：系统性误差

偏差衡量你的模型平均预测与真实值之间的差距。如果你在来自同一分布的许多不同训练集上训练同一模型并平均其预测，偏差就是这个平均值与真实值之间的差距。

高偏差意味着模型过于僵化，无法捕捉真实模式。用直线拟合抛物线，无论给多少数据，它都会错过曲线。这就是欠拟合。

```
High bias (underfitting):
  Model always predicts roughly the same wrong thing.
  Training error: HIGH
  Test error: HIGH
  Gap between them: SMALL
```

### 方差：对训练数据的敏感性

方差衡量在不同数据子集上训练时，预测结果的变化程度。如果训练集的微小变化导致模型的大幅变化，方差就高。

高方差意味着模型在拟合训练数据中的噪声，而非底层信号。20 次多项式会穿过每个训练点，但在点之间剧烈振荡。这就是过拟合。

```
High variance (overfitting):
  Model fits training data perfectly but fails on new data.
  Training error: LOW
  Test error: HIGH
  Gap between them: LARGE
```

### 分解

对于任意点 x，在平方损失下的期望预测误差可以精确分解为：

```
Expected Error = Bias^2 + Variance + Irreducible Noise

where:
  Bias^2   = (E[f_hat(x)] - f(x))^2
  Variance = E[(f_hat(x) - E[f_hat(x)])^2]
  Noise    = E[(y - f(x))^2]             (sigma^2)
```

- `f(x)` 是真实函数
- `f_hat(x)` 是模型的预测
- `E[...]` 是对不同训练集的期望
- `y` 是观测标签（真实函数加上噪声）

噪声项是不可约的。在噪声数据上，任何模型都无法做得比 sigma^2 更好。你的任务是找到 bias^2 和 variance 之间的正确平衡。

### 模型复杂度与误差

```mermaid
graph LR
    A[Simple Model] -->|increase complexity| B[Sweet Spot]
    B -->|increase complexity| C[Complex Model]

    style A fill:#f9f,stroke:#333
    style B fill:#9f9,stroke:#333
    style C fill:#f99,stroke:#333
```

经典的 U 型曲线：

| 复杂度 | 偏差 | 方差 | 总误差 |
|-----------|------|----------|-------------|
| 过低 | 高 | 低 | 高（欠拟合） |
| 适中 | 中等 | 中等 | 最低 |
| 过高 | 低 | 高 | 高（过拟合） |

### 正则化作为偏差-方差控制

正则化故意增加偏差以减少方差。它约束模型，使其无法追逐噪声。

- **L2（Ridge）：** 将所有权重向零收缩。保留所有特征但降低其影响。
- **L1（Lasso）：** 将部分权重精确推至零。执行特征选择。
- **Dropout：** 在训练期间随机禁用神经元。强制冗余表示。
- **早停：** 在模型完全拟合训练数据之前停止训练。

正则化强度（lambda、dropout 比率、epoch 数量）直接控制你在偏差-方差曲线上的位置。更多正则化意味着更多偏差、更少方差。

### 双下降：现代视角

经典理论认为：过了最优点后，更多的复杂度总是有害的。但 2019 年以来的研究揭示了一个意外现象。如果你持续增加模型容量，远超插值阈值（模型有足够参数完美拟合训练数据的点），测试误差可能再次下降。

```mermaid
graph LR
    A[Underfit Zone] --> B[Classical Sweet Spot]
    B --> C[Interpolation Threshold]
    C --> D[Double Descent - Error Drops Again]

    style A fill:#fdd,stroke:#333
    style B fill:#dfd,stroke:#333
    style C fill:#fdd,stroke:#333
    style D fill:#dfd,stroke:#333
```

这种"双下降"现象解释了为什么 massively overparameterized 的神经网络（参数远多于训练样本）仍然能很好地泛化。经典的偏差-方差权衡并非错误，但在现代 regime 下是不完整的。

关于双下降的关键观察：
- 它出现在线性模型、决策树和神经网络中
- 在插值区域，更多数据实际上可能有害（样本层面的双下降）
- 更多训练 epoch 也会导致这种现象（epoch 层面的双下降）
- 正则化可以平滑峰值，但无法消除它

为什么会这样？在插值阈值处，模型刚好有足够容量拟合所有训练点。它被迫进入一个穿过每个点的非常特定的解，数据的微小扰动就会导致拟合的大幅变化。这就是方差达到峰值的地方。超过阈值后，模型有许多可能完美拟合数据的解。学习算法（例如具有隐式正则化的梯度下降）倾向于选择其中最简单的一个。这种对简单解的隐式偏好就是过参数化模型能够泛化的原因。

| 区域 | 参数 vs 样本 | 行为 |
|--------|----------------------|----------|
| 欠参数化 | p << n | 经典权衡适用 |
| 插值阈值 | p ~ n | 方差达到峰值，测试误差激增 |
| 过参数化 | p >> n | 隐式正则化生效，测试误差下降 |

实际应用建议：如果你在使用神经网络或大型树集成，不要在插值阈值处停止。要么远低于它（使用显式正则化），要么远高于它。最糟糕的位置就是恰好在阈值处。

### 诊断你的模型

```mermaid
flowchart TD
    A[Compare train error vs test error] --> B{Large gap?}
    B -->|Yes| C[High variance - overfitting]
    B -->|No| D{Both errors high?}
    D -->|Yes| E[High bias - underfitting]
    D -->|No| F[Good fit]

    C --> G[More data / Regularize / Simpler model]
    E --> H[More features / Complex model / Less regularization]
    F --> I[Deploy]
```

| 症状 | 诊断 | 解决方案 |
|---------|-----------|-----|
| 训练误差高，测试误差高 | 偏差 | 更多特征、更复杂的模型、减少正则化 |
| 训练误差低，测试误差高 | 方差 | 更多数据、正则化、更简单的模型、dropout |
| 训练误差低，测试误差低 | 拟合良好 | 部署上线 |
| 训练误差下降，测试误差上升 | 过拟合进行中 | 早停 |

### 实用策略

**当偏差是问题时：**
- 添加多项式或交互特征
- 使用更灵活的模型（树集成替代线性模型）
- 降低正则化强度
- 延长训练时间（如果尚未收敛）

**当方差是问题时：**
- 获取更多训练数据
- 使用 bagging（随机森林）
- 增加正则化（更高的 lambda、更多 dropout）
- 特征选择（移除噪声特征）
- 使用交叉验证尽早发现

### 集成方法与方差降低

集成方法是应对方差最实用的工具。

**Bagging（Bootstrap Aggregating）** 在训练数据的不同 bootstrap 样本上训练多个模型，然后平均它们的预测。每个单独模型具有高方差，但平均后的方差低得多。随机森林就是将 bagging 应用于决策树。

其数学原理：如果你平均 N 个独立预测，每个预测的方差为 sigma^2，则平均的方差为 sigma^2 / N。模型并非真正独立（它们看到相似的数据），所以降低幅度小于 1/N，但仍然相当显著。

**Boosting** 通过顺序构建模型来降低偏差，每个新模型关注当前集成的误差。梯度提升和 AdaBoost 是主要例子。如果添加太多模型，Boosting 可能过拟合，因此需要早停或正则化。

| 方法 | 主要效果 | 偏差变化 | 方差变化 |
|--------|---------------|-------------|-----------------|
| Bagging | 降低方差 | 不变 | 降低 |
| Boosting | 降低偏差 | 降低 | 可能增加 |
| Stacking | 两者都降 | 取决于元学习器 | 取决于基模型 |
| Dropout | 隐式 bagging | 轻微增加 | 降低 |

**实用规则：** 如果你的基模型方差高（深树、高次多项式），使用 bagging。如果你的基模型偏差高（浅树桩、简单线性模型），使用 boosting。

### 学习曲线

学习曲线绘制训练误差和验证误差随训练集大小的变化。它们是你最实用的诊断工具。与单一的训练/测试比较不同，学习曲线展示模型的轨迹，并告诉你更多数据是否有帮助。

```mermaid
flowchart TD
    subgraph HB["High Bias Learning Curve"]
        direction LR
        HB1["Small N: both errors high"]
        HB2["Large N: both errors converge to HIGH error"]
        HB1 --> HB2
    end

    subgraph HV["High Variance Learning Curve"]
        direction LR
        HV1["Small N: train low, test high (big gap)"]
        HV2["Large N: gap shrinks but slowly"]
        HV1 --> HV2
    end

    subgraph GF["Good Fit Learning Curve"]
        direction LR
        GF1["Small N: some gap"]
        GF2["Large N: both converge to LOW error"]
        GF1 --> GF2
    end
```

如何解读：

| 场景 | 训练误差 | 验证误差 | 差距 | 含义 | 应对措施 |
|----------|---------------|-----------------|-----|---------------|------------|
| 高偏差 | 高 | 高 | 小 | 模型无法捕捉模式 | 更多特征、更复杂的模型、减少正则化 |
| 高方差 | 低 | 高 | 大 | 模型记忆训练数据 | 更多数据、正则化、更简单的模型 |
| 拟合良好 | 中等 | 中等 | 小 | 模型泛化良好 | 部署上线 |
| 高方差，正在改善 | 低 | 随数据增加而降低 | 缩小 | 数据可以解决的方差问题 | 收集更多数据 |
| 高偏差，平坦 | 高 | 高且平坦 | 小且平坦 | 更多数据无济于事 | 改变模型架构 |

关键洞见：如果两条曲线都已平稳且差距很小，但两者误差都高，更多数据无用。你需要更好的模型。如果差距很大且仍在缩小，更多数据会有帮助。

### 如何生成学习曲线

有两种方法：

**方法 1：改变训练集大小，固定模型。** 保持模型和超参数不变。在逐渐增大的训练数据子集上训练。在每个大小处测量训练误差和验证误差。这是标准的学习曲线。

**方法 2：改变模型复杂度，固定数据。** 保持数据不变。扫描复杂度参数（多项式次数、树深度、层数）。在每个复杂度处测量训练误差和验证误差。这是验证曲线，直接展示偏差-方差权衡。

两种方法互为补充。第一种告诉你更多数据是否有帮助。第二种告诉你不同的模型是否有帮助。在决定下一步之前，两种都运行。

```mermaid
flowchart TD
    A[Model underperforming] --> B[Generate learning curve]
    B --> C{Gap between train and val?}
    C -->|Large gap, val still decreasing| D[More data will help]
    C -->|Small gap, both high| E[More data will NOT help]
    C -->|Large gap, val flat| F[Regularize or simplify]
    E --> G[Generate validation curve]
    G --> H[Try more complex model]
```

## 动手实现

`code/bias_variance.py` 中的代码运行完整的偏差-方差分解实验。以下是分步方法。

### 步骤 1：从已知函数生成合成数据

我们使用 `f(x) = sin(1.5x) + 0.5x` 并添加高斯噪声。知道真实函数让我们能够计算精确的偏差和方差。

```python
def true_function(x):
    return np.sin(1.5 * x) + 0.5 * x

def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y
```

### 步骤 2：Bootstrap 采样和多项式拟合

对于每个多项式次数，我们抽取多个 bootstrap 训练集，拟合多项式，并在固定测试网格上记录预测。这给出了每个测试点处预测的分布。

```python
def fit_polynomial(x_train, y_train, degree, lam=0.0):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w
```

我们在 200 个不同的 bootstrap 样本上拟合。每个 bootstrap 样本来自相同的底层分布，但包含不同的点。

### 步骤 3：计算 Bias^2、方差分解

有了每个测试点处 200 组预测，我们可以直接从定义计算分解：

```python
mean_pred = predictions.mean(axis=0)
bias_sq = np.mean((mean_pred - y_true) ** 2)
variance = np.mean(predictions.var(axis=0))
total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1))
```

- `mean_pred` 是从 bootstrap 样本估计的 E[f_hat(x)]
- `bias_sq` 是平均预测与真实值之间的平方差距
- `variance` 是预测在 bootstrap 样本间的平均离散程度
- `total_error` 应近似等于 bias^2 + variance + noise

### 步骤 4：学习曲线

学习曲线在固定模型复杂度的情况下扫描训练集大小。它们展示你的模型是受数据限制还是容量限制。

```python
def demo_learning_curves():
    sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
    degree = 5

    for n in sizes:
        train_errors = []
        test_errors = []
        for seed in range(50):
            x_train, y_train = generate_data(n_samples=n, seed=seed * 100)
            w = fit_polynomial(x_train, y_train, degree)
            train_pred = predict_polynomial(x_train, w)
            train_mse = np.mean((train_pred - y_train) ** 2)
            test_pred = predict_polynomial(x_test, w)
            test_mse = np.mean((test_pred - y_test) ** 2)
            train_errors.append(train_mse)
            test_errors.append(test_mse)
        # Average over runs gives the learning curve point
```

对于高方差模型（小数据下的 5 次多项式），你会看到：
- 训练误差起始较低，随数据增加而上升，因为更多数据使记忆更困难
- 测试误差起始较高，随模型获得更多信号而下降
- 差距随数据增加而缩小

对于高偏差模型（1 次），两条曲线快速收敛到相同的高值，更多数据无济于事。

### 步骤 5：正则化扫描

代码还包含 `demo_regularization_sweep()`，它固定高次多项式（15 次）并扫描 Ridge 正则化强度从 0.001 到 100。这从不同角度展示偏差-方差权衡：不是改变模型复杂度，而是改变约束强度。

```python
def demo_regularization_sweep():
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    for alpha in alphas:
        results = bias_variance_decomposition([15], lam=alpha)
        r = results[15]
        print(f"alpha={alpha:.3f}  bias={r['bias_sq']:.4f}  var={r['variance']:.4f}")
```

在低 alpha 处，15 次多项式几乎无约束。方差占主导，因为模型追逐每个 bootstrap 样本中的噪声。在高 alpha 处，惩罚如此强烈，模型实际上变成了接近常数的函数。偏差占主导。最优 alpha 位于这两个极端之间。

这与改变多项式次数的 U 型曲线相同，但由连续旋钮而非离散值控制。实际应用中，正则化是控制权衡的首选方式，因为它允许精细控制而无需改变特征集。

## 应用实践

sklearn 提供了 `learning_curve` 和 `validation_curve` 来自动化这些诊断，无需编写 bootstrap 循环。

### 验证曲线：扫描模型复杂度

```python
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

degrees = list(range(1, 16))
train_scores_all = []
val_scores_all = []

for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(d), Ridge(alpha=0.01))
    train_scores, val_scores = validation_curve(
        pipe, X, y, param_name="polynomialfeatures__degree",
        param_range=[d], cv=5, scoring="neg_mean_squared_error"
    )
    train_scores_all.append(-train_scores.mean())
    val_scores_all.append(-val_scores.mean())
```

这直接给出偏差-方差权衡曲线。验证分数相对于训练分数最差的地方，方差占主导。两者都差的地方，偏差占主导。

### 学习曲线：扫描训练集大小

```python
from sklearn.model_selection import learning_curve

pipe = make_pipeline(PolynomialFeatures(5), Ridge(alpha=0.01))
train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="neg_mean_squared_error"
)
train_mse = -train_scores.mean(axis=1)
val_mse = -val_scores.mean(axis=1)
```

绘制 `train_mse` 和 `val_mse` 随 `train_sizes` 的变化。形状告诉你关于模型的一切信息。

### 带正则化扫描的交叉验证

```python
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    pipe = make_pipeline(PolynomialFeatures(10), Ridge(alpha=alpha))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"alpha={alpha:>7.3f}  MSE={-scores.mean():.4f} +/- {scores.std():.4f}")
```

这扫描固定模型复杂度下的正则化强度。你会看到相同的偏差-方差权衡：低 alpha 意味着高方差，高 alpha 意味着高偏差。

### 综合应用：完整的诊断工作流

实际应用中，你按顺序运行这些诊断：

1. 训练你的模型。计算训练误差和测试误差。
2. 如果两者都高：你有偏差问题。跳到第 4 步。
3. 如果训练误差低但测试误差高：你有方差问题。生成学习曲线，看更多数据是否有帮助。如果没有，进行正则化。
4. 生成验证曲线，扫描你的主要复杂度参数。找到最优点。
5. 在最优点，生成学习曲线。如果差距仍然很大，你需要更多数据或正则化。
6. 使用 `cross_val_score` 尝试不同 alpha 值的 Ridge/Lasso。选择交叉验证误差最低的 alpha。

对于大多数表格数据集，这只需 10-15 分钟计算，却能节省数小时的猜测。

## 交付成果

本课程产出：`outputs/prompt-model-diagnostics.md`

## 练习题

1. 用 `noise_std=0`（无噪声）运行分解。不可约误差项会发生什么变化？最优复杂度会改变吗？

2. 将训练集大小从 30 增加到 300。这如何影响方差分量？最优多项式次数会移动吗？

3. 在实验中添加 L2 正则化（Ridge 回归）。对于固定的高次多项式（15 次），扫描 lambda 从 0 到 100。绘制 bias^2 和 variance 随 lambda 变化的函数。

4. 将真实函数从多项式改为 `sin(x)`。偏差-方差分解如何变化？是否仍有明确的最优次数？

5. 实现一个简单的 bootstrap aggregating（bagging）包装器：在 bootstrap 样本上训练 10 个模型并平均预测。证明这能在不大幅增加偏差的情况下降低方差。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|----------------------|
| 偏差 | "模型太简单" | 错误假设导致的系统性误差。平均模型预测与真实值之间的差距。 |
| 方差 | "模型过拟合" | 对训练数据敏感导致的误差。预测在不同训练集间的变化程度。 |
| 不可约误差 | "数据中的噪声" | 真实数据生成过程中的随机性导致的误差。任何模型都无法消除。 |
| 欠拟合 | "学得不够" | 模型具有高偏差。即使在训练数据上也错过了真实模式。 |
| 过拟合 | "记住了数据" | 模型具有高方差。它拟合了训练数据中的噪声，无法泛化。 |
| 正则化 | "约束模型" | 添加惩罚以降低模型复杂度，以偏差换取更低方差。 |
| 双下降 | "更多参数可能有帮助" | 当模型容量远超插值阈值时，测试误差再次下降。 |
| 模型复杂度 | "模型的灵活程度" | 模型拟合任意模式的能力。由架构、特征或正则化控制。 |

## 延伸阅读

- [Hastie, Tibshirani, Friedman: Elements of Statistical Learning, Ch. 7](https://hastie.su.domains/ElemStatLearn/) -- 偏差-方差分解的权威论述
- [Belkin et al., Reconciling modern machine learning practice and the bias-variance trade-off (2019)](https://arxiv.org/abs/1812.11118) -- 双下降论文
- [Nakkiran et al., Deep Double Descent (2019)](https://arxiv.org/abs/1912.02292) -- epoch 层面和样本层面的双下降
- [Scott Fortmann-Roe: Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html) -- 清晰的视觉解释
