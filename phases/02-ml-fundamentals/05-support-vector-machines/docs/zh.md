# 支持向量机

> 找到两类之间最宽的街道。这就是全部思想。

**类型：** Build
**语言：** Python
**前置知识：** Phase 1（第 08 课 优化、第 14 课 范数与距离、第 18 课 凸优化）
**时间：** ~90 分钟

## 学习目标

- 使用 hinge loss 和梯度下降在原始形式上从零实现线性 SVM
- 解释最大间隔原理并从训练好的模型中识别支持向量
- 比较线性核、多项式核和 RBF 核，并解释核技巧如何避免显式的高维映射
- 评估由 C 参数控制的间隔宽度与分类错误之间的权衡

## 问题

你有两类数据点，需要画一条线（或超平面）将它们分开。有无数条线可以做到。你应该选哪一条？

间隔最大的那条。间隔是决策边界与每侧最近数据点之间的距离。更宽的间隔意味着分类器更自信，对未见数据的泛化能力更好。

这一直觉引出了支持向量机（SVM），机器学习中最具数学优雅性的算法之一。SVM 在深度学习之前是占主导地位的分类方法，对于小数据集、高维数据以及需要原理清晰、理论保证充分的模型的问题，仍然是最佳选择。

SVM 与 Phase 1 直接相关：优化是凸的（第 18 课），间隔用范数度量（第 14 课），核技巧利用点积处理非线性边界，而无需在高维空间中进行任何计算。

## 概念

### 最大间隔分类器

给定标签 y_i ∈ {-1, +1} 和特征向量 x_i 的线性可分数据，我们希望找到一个超平面 w^T x + b = 0 来分隔两类。

点 x_i 到超平面的距离为：

```
distance = |w^T x_i + b| / ||w||
```

对于正确分类的点：y_i * (w^T x_i + b) > 0。间隔是超平面到任一侧最近点距离的两倍。

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["+ class points"] --> A
    E["- class points"] --> C
    B --- F["Decision boundary"]
```

优化问题：

```
maximize    2 / ||w||     (the margin width)
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

等价地（最小化 ||w||^2 更容易优化）：

```
minimize    (1/2) ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

这是一个凸二次规划问题。它有唯一的全局解。恰好位于间隔边界上的数据点（满足 y_i * (w^T x_i + b) = 1）就是支持向量。它们是唯一决定决策边界的点。移动或移除任何非支持向量点，边界都不会改变。

### 支持向量：关键的少数

```mermaid
graph TD
    subgraph Classification
        SV1["Support Vector (+ class)<br>y(w'x+b) = 1"] --- DB["Decision Boundary<br>w'x+b = 0"]
        DB --- SV2["Support Vector (- class)<br>y(w'x+b) = 1"]
    end
    O1["Other + points<br>(do not affect boundary)"] -.-> SV1
    O2["Other - points<br>(do not affect boundary)"] -.-> SV2
```

大多数训练点都是无关的。只有支持向量重要。这就是为什么 SVM 在预测时内存效率高：你只需要存储支持向量，而不是整个训练集。

支持向量的数量也给出了泛化误差的上界。相对于数据集大小而言，支持向量越少意味着泛化能力越好。

### 软间隔：用 C 参数处理噪声

真实数据很少能完美分离。有些点可能在边界错误的一侧，或者在间隔内部。软间隔公式通过引入松弛变量来允许违反。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i
```

松弛变量 xi_i 度量点 i 违反间隔的程度。C 控制权衡：

| C 值 | 行为 |
|---------|----------|
| 大 C | 严重惩罚违反。间隔窄，误分类少。过拟合 |
| 小 C | 允许更多违反。间隔宽，误分类多。欠拟合 |

C 是正则化强度的倒数。大 C = 较少正则化。小 C = 较多正则化。

### Hinge loss：SVM 的损失函数

软间隔 SVM 可以重写为无约束优化：

```
minimize    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
```

项 max(0, 1 - y_i * f(x_i)) 就是 hinge loss。当点被正确分类且超出间隔时为零。当点在间隔内或被误分类时为线性增长。

```
Hinge loss for a single point:

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

Zero loss when y*f(x) >= 1 (correctly classified, outside margin).
Linear penalty when y*f(x) < 1.
```

与 logistic loss（逻辑回归）比较：

```
Hinge:     max(0, 1 - y*f(x))          Hard cutoff at margin
Logistic:  log(1 + exp(-y*f(x)))        Smooth, never exactly zero
```

Hinge loss 产生稀疏解（只有支持向量有非零贡献）。Logistic loss 使用所有数据点。这使得 SVM 在预测时更节省内存。

### 用梯度下降训练线性 SVM

你可以使用 hinge loss 加 L2 正则化的梯度下降来训练线性 SVM，无需解约束 QP：

```
L(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - y_i * (w^T x_i + b)))

Gradient with respect to w:
  If y_i * (w^T x_i + b) >= 1:  dL/dw = lambda * w
  If y_i * (w^T x_i + b) < 1:   dL/dw = lambda * w - y_i * x_i

Gradient with respect to b:
  If y_i * (w^T x_i + b) >= 1:  dL/db = 0
  If y_i * (w^T x_i + b) < 1:   dL/db = -y_i
```

这称为原始形式（primal formulation）。每轮迭代复杂度为 O(n * d)，其中 n 是样本数，d 是特征数。对于大型稀疏高维数据（文本分类），这很快。

### 对偶形式与核技巧

SVM 问题的拉格朗日对偶（来自 Phase 1 第 18 课，KKT 条件）为：

```
maximize    sum(alpha_i) - (1/2) * sum_ij(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
subject to  0 <= alpha_i <= C
            sum(alpha_i * y_i) = 0
```

对偶形式只涉及数据点之间的点积 x_i . x_j。这是关键洞见。将每个点积替换为核函数 K(x_i, x_j)，SVM 就可以学习非线性边界，而无需显式计算变换。

```
Linear kernel:      K(x, z) = x . z
Polynomial kernel:  K(x, z) = (x . z + c)^d
RBF (Gaussian):     K(x, z) = exp(-gamma * ||x - z||^2)
```

RBF 核将数据映射到无限维空间。输入空间中相近的点核值接近 1。相距远的点核值接近 0。它可以学习任何平滑的决策边界。

```mermaid
graph LR
    subgraph "Input Space (not separable)"
        A["Data points in 2D<br>circular boundary"]
    end
    subgraph "Feature Space (separable)"
        B["Data points in higher dim<br>linear boundary"]
    end
    A -->|"Kernel trick<br>K(x,z) = phi(x).phi(z)"| B
```

核技巧在高维空间中计算点积，而无需真正到达那里。对于 D 维中 d 次多项式核，显式特征空间有 O(D^d) 维。但 K(x, z) 只需 O(D) 时间计算。

### 用于回归的 SVM（SVR）

支持向量回归在数据周围拟合一个宽度为 epsilon 的管道。管道内的点损失为零。管道外的点被线性惩罚。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i + xi_i*)
subject to  y_i - (w^T x_i + b) <= epsilon + xi_i
            (w^T x_i + b) - y_i <= epsilon + xi_i*
            xi_i, xi_i* >= 0
```

epsilon 参数控制管道宽度。管道越宽 = 支持向量越少 = 拟合越平滑。管道越窄 = 支持向量越多 = 拟合越紧密。

### 为什么 SVM 输给了深度学习（以及它们何时仍然获胜）

SVM 从 1990 年代末到 2010 年代初主导着机器学习。深度学习在几个方面超越了它们：

| 因素 | SVM | 深度学习 |
|--------|------|---------------|
| 特征工程 | 需要它 | 学习特征 |
| 可扩展性 | 核方法 O(n^2) 到 O(n^3) | 使用 SGD 每轮 O(n) |
| 图像/文本/音频 | 需要手工设计特征 | 从原始数据学习 |
| 大数据集 (>100k) | 慢 | 扩展性好 |
| GPU 加速 | 收益有限 | 大幅提速 |

SVM 在以下情况仍然获胜：
- 小数据集（数百到数千个样本）
- 高维稀疏数据（带 TF-IDF 特征的文本）
- 需要数学保证时（间隔界）
- 训练时间必须极短时（线性 SVM 非常快）
- 具有清晰间隔结构的二分类
- 异常检测（单类 SVM）

## 动手实现

### 步骤 1：Hinge loss 和梯度

基础。为批次计算 hinge loss 及其梯度。

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### 步骤 2：通过梯度下降实现线性 SVM

通过最小化正则化 hinge loss 来训练。无需 QP 求解器。

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### 步骤 3：核函数

实现线性核、多项式核和 RBF 核。

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### 步骤 4：间隔和支持向量识别

训练后，识别哪些点是支持向量并计算间隔宽度。

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

完整实现及所有演示请参见 `code/svm.py`。

## 使用

使用 scikit-learn：

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")
```

重要提示：训练 SVM 前务必对特征进行缩放。SVM 对特征量级敏感，因为间隔依赖于 ||w||，而未缩放的特征会扭曲几何结构。

对于大数据集，使用 `LinearSVC`（原始形式，每轮 O(n)）而非 `SVC`（对偶形式，O(n^2) 到 O(n^3)）：

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## 练习

1. 生成一个 2D 线性可分数据集。训练你的 LinearSVM 并识别支持向量。验证支持向量是距离决策边界最近的点。

2. 在噪声数据集上将 C 从 0.001 变化到 1000。为每个 C 值绘制决策边界。观察从宽间隔（欠拟合）到窄间隔（过拟合）的过渡。

3. 创建一个类边界为圆形（非线性）的数据集。展示线性 SVM 失败。计算 RBF 核矩阵并展示在核诱导的特征空间中类变得可分。

4. 在同一数据集上比较 hinge loss 和 logistic loss。训练线性 SVM 和逻辑回归。统计每个模型的决策边界由多少训练点贡献（支持向量 vs 所有点）。

5. 实现 SVR（epsilon-不敏感损失）。将其拟合到 y = sin(x) + 噪声。在预测周围绘制 epsilon 管道并突出显示支持向量（管道外的点）。

## 关键术语

| 术语 | 实际含义 |
|------|----------------------|
| 支持向量 | 距离决策边界最近的训练点。唯一决定超平面的点 |
| 间隔 | 决策边界与最近支持向量之间的距离。SVM 最大化这个值 |
| Hinge loss | max(0, 1 - y*f(x))。正确分类且超出间隔时为零。否则线性惩罚 |
| C 参数 | 间隔宽度与分类错误之间的权衡。大 C = 窄间隔，小 C = 宽间隔 |
| 软间隔 | 通过松弛变量允许间隔违反的 SVM 公式。处理不可分数据 |
| 核技巧 | 在高维特征空间中计算点积，而无需显式映射到该空间 |
| 线性核 | K(x, z) = x . z。等价于标准点积。用于线性可分数据 |
| RBF 核 | K(x, z) = exp(-gamma * \|\|x-z\|\|^2)。映射到无限维。学习任何平滑边界 |
| 多项式核 | K(x, z) = (x . z + c)^d。映射到多项式组合的特征空间 |
| 对偶形式 | 仅依赖于数据点之间点积的 SVM 问题重构。启用核方法 |
| SVR | 支持向量回归。在数据周围拟合 epsilon 管道。管道内的点损失为零 |
| 松弛变量 | xi_i：度量点违反间隔的程度。间隔外正确分类的点为零 |
| 最大间隔 | 选择最大化到每类最近点距离的超平面的原理 |

## 延伸阅读

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1) - SVM 和统计学习的奠基性著作
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018) - 原始 SVM 论文
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) - 使 SVM 训练实用的 SMO 算法
- [scikit-learn SVM documentation](https://scikit-learn.org/stable/modules/svm.html) - 包含实现细节的实用指南
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - 大多数 SVM 实现背后的 C++ 库
