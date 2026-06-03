# K 近邻与距离

> 存储所有数据。通过观察邻居来预测。最简单却真正有效的算法。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1（第 14 课 范数与距离）
**时间：** ~90 分钟

## 学习目标

- 从零实现 KNN 分类与回归，支持可配置的 K 值和距离加权投票
- 比较 L1、L2、余弦和闵可夫斯基距离度量，并为给定数据类型选择合适的度量
- 解释维度灾难，并演示为什么 KNN 在高维空间中性能下降
- 构建 KD 树以实现高效的最近邻搜索，并分析其何时优于暴力搜索

## 问题

你有一个数据集。一个新的数据点到来。你需要对其进行分类或预测其值。与其从数据中学习参数（如线性回归或 SVM），不如找到与新点最接近的 K 个训练点，让它们投票。

这就是 K 近邻。没有训练阶段。没有参数要学习。没有损失函数要最小化。你存储整个训练集，在预测时计算距离。

这听起来简单得不像能工作。但 KNN 对许多问题来说 surprisingly 有竞争力，尤其是在小到中等规模的数据集上，深入理解它能揭示基本概念：距离度量的选择（连接到阶段 1 第 14 课）、维度灾难，以及惰性学习与主动学习之间的区别。

KNN 在现代 AI 中也无处不在，只是换了不同的名字。向量数据库对嵌入做 KNN 搜索。检索增强生成（RAG）找到 K 个最近的文档块。推荐系统找到相似的用户或物品。算法是相同的。规模和数据结构不同。

## 概念

### KNN 如何工作

给定一个带标签点的数据集和一个新的查询点：

1. 计算查询点到数据集中每个点的距离
2. 按距离排序
3. 取 K 个最近的点
4. 分类：K 个邻居中的多数投票
5. 回归：K 个邻居值的平均（或加权平均）

```mermaid
graph TD
    Q["Query point ?"] --> D["Compute distances<br>to all training points"]
    D --> S["Sort by distance"]
    S --> K["Select K nearest"]
    K --> C{"Classification<br>or Regression?"}
    C -->|Classification| V["Majority vote"]
    C -->|Regression| A["Average values"]
    V --> P["Prediction"]
    A --> P
```

这就是整个算法。没有拟合。没有梯度下降。没有轮次。

### 选择 K

K 是唯一的超参数。它控制偏差-方差权衡：

| K | 行为 |
|---|------|
| K = 1 | 决策边界跟随每个点。训练误差为零。高方差。过拟合 |
| 小 K（3-5）| 对局部结构敏感。能捕捉复杂边界 |
| 大 K | 边界更平滑。对噪声更鲁棒。可能欠拟合 |
| K = N | 对每个点都预测多数类。最大偏差 |

常见的起点是 K = sqrt(N)，其中 N 是数据集中的点数。二分类时使用奇数 K 以避免平局。

```mermaid
graph LR
    subgraph "K=1 (overfitting)"
        A["Jagged boundary<br>follows every point"]
    end
    subgraph "K=15 (good)"
        B["Smooth boundary<br>captures true pattern"]
    end
    subgraph "K=N (underfitting)"
        C["Flat boundary<br>predicts majority class"]
    end
    A -->|"increase K"| B -->|"increase K"| C
```

### 距离度量

距离函数定义了"近"的含义。不同的度量产生不同的邻居，不同的预测。

**L2（欧几里得）** 是默认值。直线距离。

```
d(a, b) = sqrt(sum((a_i - b_i)^2))
```

对特征尺度敏感。使用 L2 的 KNN 前务必标准化特征。

**L1（曼哈顿）** 求绝对差之和。比 L2 更鲁棒，因为它不对差值平方。

```
d(a, b) = sum(|a_i - b_i|)
```

**余弦距离** 测量向量之间的夹角，忽略大小。对文本和嵌入数据至关重要。

```
d(a, b) = 1 - (a . b) / (||a|| * ||b||)
```

**闵可夫斯基** 用参数 p 泛化 L1 和 L2。

```
d(a, b) = (sum(|a_i - b_i|^p))^(1/p)

p=1: Manhattan
p=2: Euclidean
p->inf: Chebyshev (max absolute difference)
```

使用哪种度量取决于数据：

| 数据类型 | 最佳度量 | 原因 |
|---------|---------|------|
| 数值特征，尺度相近 | L2（欧几里得）| 默认值，适用于空间数据 |
| 数值特征，有异常值 | L1（曼哈顿）| 鲁棒，不放大较大差异 |
| 文本嵌入 | 余弦 | 大小是噪声，方向是语义 |
| 高维稀疏 | 余弦或 L1 | L2 受维度灾难影响 |
| 混合类型 | 自定义距离 | 按特征类型组合度量 |

### 加权 KNN

标准 KNN 对所有 K 个邻居赋予相同权重。但距离为 0.1 的邻居应该比距离为 5.0 的邻居更重要。

**距离加权 KNN** 按距离的倒数为每个邻居加权：

```
weight_i = 1 / (distance_i + epsilon)

For classification: weighted vote
For regression:     weighted average = sum(w_i * y_i) / sum(w_i)
```

epsilon 防止当查询点与训练点完全匹配时除以零。

加权 KNN 对 K 的选择不太敏感，因为无论 K 是多少， distant 邻居的贡献都很少。

### 维度灾难

KNN 在高维空间中性能下降。这不是模糊的担忧。这是数学事实。

**问题 1：距离收敛。** 随着维度增加，最大距离与最小距离的比值趋近于 1。所有点都变得与查询点"等距"。

```
In d dimensions, for random uniform points:

d=2:    max_dist / min_dist = varies widely
d=100:  max_dist / min_dist ~ 1.01
d=1000: max_dist / min_dist ~ 1.001

When all distances are nearly equal, "nearest" is meaningless.
```

**问题 2：体积爆炸。** 要在数据的固定比例内捕获 K 个邻居，你需要将搜索半径扩展到覆盖特征空间的更大比例。高维中的"邻域"涵盖了大部分空间。

**问题 3：角点主导。** 在 d 维单位超立方体中，大部分体积集中在角点附近，而非中心。内切于立方体的球体所包含的体积比例随着 d 增大而趋近于零。

实际后果：KNN 在大约 20-50 个特征以内工作良好。超过这个范围，你需要在应用 KNN 之前进行降维（PCA、UMAP、t-SNE），或者利用数据内在低维性的基于树的搜索结构。

### KD 树：快速最近邻搜索

暴力 KNN 计算查询点到每个训练点的距离。每次查询是 O(n * d)。对于大数据集，这太慢了。

KD 树沿特征轴递归划分空间。在每个层级，它沿一个维度在中位数处分割。

```mermaid
graph TD
    R["Split on x1 at 5.0"] -->|"x1 <= 5.0"| L["Split on x2 at 3.0"]
    R -->|"x1 > 5.0"| RR["Split on x2 at 7.0"]
    L -->|"x2 <= 3.0"| LL["Leaf: 3 points"]
    L -->|"x2 > 3.0"| LR["Leaf: 4 points"]
    RR -->|"x2 <= 7.0"| RL["Leaf: 2 points"]
    RR -->|"x2 > 7.0"| RRR["Leaf: 5 points"]
```

要找到最近邻，遍历到包含查询点的叶节点，然后回溯并检查相邻分区，仅当它们可能包含更近的点时。

平均查询时间：低维时 O(log n)。但 KD 树在高维（d > 20）时退化到 O(n)，因为回溯时剪掉的分支越来越少。

### 球树：更适合中等维度

球树将数据划分为嵌套的超球体，而非轴对齐的盒子。每个节点定义一个球（中心 + 半径），包含该子树中的所有点。

相对于 KD 树的优势：
- 在中等维度（最高约 50）表现更好
- 处理非轴对齐结构
- 更紧的包围体积意味着搜索时更多分支被剪枝

KD 树和球树都是精确算法。对于真正大规模的搜索（数百万点，数百维），使用近似最近邻方法（HNSW、IVF、乘积量化）。这些在阶段 1 第 14 课中介绍。

### 惰性学习与主动学习

KNN 是惰性学习器：训练时不做任何工作，预测时做所有工作。大多数其他算法（线性回归、SVM、神经网络）是主动学习器：它们在训练时进行大量计算来构建紧凑模型，然后预测很快。

| 方面 | 惰性（KNN） | 主动（SVM、神经网络）|
|------|-----------|----------------------|
| 训练时间 | O(1)，仅存储数据 | O(n * epochs) |
| 预测时间 | 每次查询 O(n * d) | O(d) 或 O(参数) |
| 预测时内存 | 存储整个训练集 | 仅存储模型参数 |
| 适应新数据 | 即时添加点 | 重新训练模型 |
| 决策边界 | 隐式，即时计算 | 显式，训练后固定 |

惰性学习适用于：
- 数据集频繁变化（添加/删除点无需重新训练）
- 只需要很少的查询预测
- 需要零训练时间
- 数据集足够小，暴力搜索很快

### KNN 回归

KNN 回归不是多数投票，而是对 K 个邻居的目标值取平均。

```
prediction = (1/K) * sum(y_i for i in K nearest neighbors)

Or with distance weighting:
prediction = sum(w_i * y_i) / sum(w_i)
where w_i = 1 / distance_i
```

KNN 回归产生分段常数（或加权时分段平滑）的预测。它无法外推到训练数据范围之外。如果训练目标都在 0 到 100 之间，KNN 永远不会预测 200。

## 构建

### 步骤 1：距离函数

实现 L1、L2、余弦和闵可夫斯基距离。这些直接连接到阶段 1 第 14 课。

```python
import math

def l2_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def l1_distance(a, b):
    return sum(abs(ai - bi) for ai, bi in zip(a, b))

def cosine_distance(a, b):
    dot_val = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot_val / (norm_a * norm_b)

def minkowski_distance(a, b, p=2):
    if p == float('inf'):
        return max(abs(ai - bi) for ai, bi in zip(a, b))
    return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)
```

### 步骤 2：KNN 分类器和回归器

构建完整的 KNN，支持可配置的 K、距离度量和可选的距离加权。

```python
class KNN:
    def __init__(self, k=5, distance_fn=l2_distance, weighted=False,
                 task="classification"):
        self.k = k
        self.distance_fn = distance_fn
        self.weighted = weighted
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_one(x) for x in X]
```

### 步骤 3：用于高效搜索的 KD 树

从零构建 KD 树，沿每个维度的中位数递归分割。

```python
class KDTree:
    def __init__(self, X, indices=None, depth=0):
        # Recursively partition the data
        self.axis = depth % len(X[0])
        # Split on median of the current axis
        ...

    def query(self, point, k=1):
        # Traverse to leaf, then backtrack
        ...
```

参见 `code/knn.py` 获取包含所有辅助方法和演示的完整实现。

### 步骤 4：特征缩放

KNN 需要特征缩放，因为距离对特征大小敏感。一个范围从 0 到 1000 的特征会主导一个范围从 0 到 1 的特征。

```python
def standardize(X):
    n = len(X)
    d = len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds = [
        max(1e-10, (sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n) ** 0.5)
        for j in range(d)
    ]
    return [[((X[i][j] - means[j]) / stds[j]) for j in range(d)] for i in range(n)], means, stds
```

## 使用

使用 scikit-learn：

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

当数据集足够大且维度足够低时，scikit-learn 自动使用 KD 树或球树。对于高维数据，它回退到暴力搜索。你可以通过 `algorithm` 参数控制这一点。

对于大规模最近邻搜索（数百万向量），使用 FAISS、Annoy 或向量数据库：

```python
import faiss

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
distances, indices = index.search(query_vectors, k=5)
```

## 练习

1. 在具有 3 个类别的 2D 数据集上实现 KNN 分类。绘制 K=1、K=5、K=15 和 K=N 时的决策边界。观察从过拟合到欠拟合的过渡。

2. 在 2、5、10、50、100 和 500 维中生成 1000 个随机点。对于每个维度，计算最大成对距离与最小成对距离的比值。绘制比值与维度的关系，以可视化维度灾难。

3. 在文本分类问题（使用 TF-IDF 向量）上比较 KNN 的 L1、L2 和余弦距离。哪种度量给出最佳准确率？为什么余弦往往在文本上获胜？

4. 实现 KD 树并测量查询时间 vs 暴力搜索，数据集为 2D、10D 和 50D 中的 1k、10k 和 100k 个点。在什么维度下 KD 树不再比暴力搜索快？

5. 为 y = sin(x) + 噪声 构建加权 KNN 回归器。将其与 K=3、10、30 的未加权 KNN 比较。证明加权产生更平滑的预测，尤其对于大 K。

## 关键术语

| 术语 | 实际含义 |
|------|---------|
| K 近邻 | 通过找到与查询最接近的 K 个训练点来预测的非参数算法 |
| 惰性学习 | 训练时不做计算。所有工作在预测时完成。KNN 是典型例子 |
| 主动学习 | 训练时进行大量计算以构建紧凑模型。大多数 ML 算法是主动的 |
| 维度灾难 | 在高维中，距离收敛且邻域扩展到覆盖大部分空间，使 KNN 失效 |
| KD 树 | 沿特征轴递归划分空间的二叉树。低维时 O(log n) 查询 |
| 球树 | 嵌套超球体的树。在中等维度（最高约 50）比 KD 树表现更好 |
| 加权 KNN | 邻居按距离倒数加权。更近的邻居对预测影响更大 |
| 特征缩放 | 将特征归一化到可比较的范围。对 KNN 等基于距离的方法是必需的 |
| 多数投票 | 通过计算 K 个邻居中哪个类别最常见来进行分类 |
| 暴力搜索 | 计算到每个训练点的距离。每次查询 O(n*d)。精确但大数据量时慢 |
| 近似最近邻 | 比精确搜索更快找到近似最近点的算法（HNSW、LSH、IVF）|
| Voronoi 图 | 空间的划分，每个区域包含所有离一个训练点比任何其他点更近的点。K=1 KNN 产生 Voronoi 边界 |

## 延伸阅读

- [Cover & Hart: Nearest Neighbor Pattern Classification (1967)](https://ieeexplore.ieee.org/document/1053964) - 奠基性 KNN 论文，证明其错误率最多是贝叶斯最优的两倍
- [Friedman, Bentley, Finkel: An Algorithm for Finding Best Matches in Logarithmic Expected Time (1977)](https://dl.acm.org/doi/10.1145/355744.355745) - 原始 KD 树论文
- [Beyer et al.: When Is "Nearest Neighbor" Meaningful? (1999)](https://link.springer.com/chapter/10.1007/3-540-49257-7_15) - 对最近邻维度灾难的正式分析
- [scikit-learn Nearest Neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - 算法选择的实用指南
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的十亿级近似最近邻搜索库
