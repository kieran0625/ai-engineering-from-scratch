# 范数与距离

> 你的距离函数定义了“相似”的含义。选错了，下游一切都会出问题。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 1，课程 01（线性代数直觉）、02（向量、矩阵与运算）
**时间：** ~90 分钟

## 学习目标

- 从零实现 L1、L2、余弦、马氏距离、Jaccard 和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量，并解释为什么其他替代方案会失败
- 将 L1 和 L2 范数与 LASSO 和 Ridge 正则化及其几何约束区域联系起来
- 演示同一数据集在不同度量下如何产生不同的最近邻

## 问题

你有两个向量。它们可能是词嵌入，可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择哪个距离函数。两个数据点在一种度量下可能是最近邻，在另一种度量下却可能相距甚远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——它们都依赖于这个选择。选错了，你的模型就会优化错误的目标。

不存在 universally best 的距离。L2 适用于空间数据。余弦相似度在 NLP 中占主导。Jaccard 处理集合。编辑距离处理字符串。马氏距离考虑相关性。Wasserstein 距离移动概率质量。每一种都编码了关于"相似"含义的不同假设。

本课程从零构建每一个主要距离函数，告诉你何时该用哪个工具，并演示同一数据在不同度量下如何产生完全不同的最近邻。

## 概念

### 范数：度量向量大小

范数度量向量的"大小"。两个向量之间的每个距离函数都可以写成它们差的范数：d(a, b) = ||a - b||。所以理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数对所有分量的绝对值求和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离，因为它度量你在城市网格中沿着坐标轴行走的距离。不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 当你需要对异常值鲁棒时（单个巨大差异不会主导结果）
- 特征选择问题（L1 正则化促进稀疏性）

与 L1 正则化（Lasso）的联系：向损失函数添加 ||w||_1 会对绝对权重值之和进行惩罚。这会将小权重精确推至零，实现自动特征选择。L1 惩罚在权重空间中形成菱形的约束区域，而菱形的角落在坐标轴上，此时某些权重为零。

与损失函数的联系：平均绝对误差（MAE）是预测值与目标值之间 L1 距离的平均值。它对所有误差线性惩罚，相比 MSE 对异常值更鲁棒。

### L2 范数（欧几里得距离）

L2 范数是直线距离。分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是你在几何课上学到的距离。n 维空间中的毕达哥拉斯定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用 L2：
- 低到中等维度的连续数据
- 特征尺度可比时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似度

与 L2 正则化（Ridge）的联系：向损失函数添加 ||w||_2^2 会对大权重进行惩罚。与 L1 不同，它不会将权重推至零。它会按比例将所有权重向零收缩。L2 惩罚形成圆形约束区域，因此坐标轴上没有角。权重变小但很少精确为零。

与损失函数的联系：均方误差（MSE）是 L2 距离平方的平均值。平方会对大误差施加比小误差更重的惩罚。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：一般族

L1 和 L2 是 Lp 范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值产生不同形状的"单位球"（所有与原点距离为 1 的点组成的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-无穷范数（切比雪夫距离）

当 p 趋近于无穷大时，Lp 范数收敛为最大绝对分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的那个维度决定。其他所有维度都被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-无穷：
- 当任何单一维度中的最坏情况偏差很重要时
- 棋盘（国际象棋中的王按 L-无穷移动：任何方向一步的代价都是 1）
- 制造公差（每个维度都必须在规格范围内）

### 余弦相似度与余弦距离

余弦相似度度量两个向量之间的夹角，忽略它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的范围从 -1（相反方向）到 +1（相同方向）。垂直向量的余弦相似度为 0。

余弦距离将其转换为距离：cosine_distance = 1 - cosine_similarity。范围从 0（相同方向）到 2（相反方向）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦相似度在 NLP 和嵌入中占主导：在文本中，文档长度不应影响相似度。一篇关于猫的文档长度是另一篇关于猫的文档的两倍，它们仍然应该是"相似的"。余弦相似度忽略大小（长度），只关心方向。两个词分布相同但长度不同的文档指向同一方向，余弦相似度为 1.0。

何时使用余弦相似度：
- 文本相似度（TF-IDF 向量、词嵌入、句子嵌入）
- 任何大小是噪声、方向是信号的领域
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎总是使用余弦或点积）

### 点积相似度 vs 余弦相似度

两个向量的点积为：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是点积除以两个向量的模长。当两个向量已经单位归一化（模长 = 1）时，点积和余弦相似度完全相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

它们的区别：点积包含大小信息。模长更大的向量获得更高的点积分数。这在某些检索系统中很重要，你希望"热门"项目排名更高。模长作为隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：
- 当你想要纯方向相似度时使用余弦相似度
- 当模长携带有意义信息时使用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在两者之间选择
- 如果你的嵌入是 L2 归一化的，选择无关紧要

### 马氏距离

欧几里得距离平等对待所有维度。但如果你的特征相关或具有不同尺度，L2 会产生误导性结果。

马氏距离考虑数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观理解：马氏距离首先对数据进行去相关和归一化（白化），然后在该变换后的空间中计算 L2 距离。如果 S 是单位矩阵（不相关、单位方差的特征），马氏距离退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：
- 异常值检测（与均值马氏距离很大的点是异常值）
- 特征具有不同尺度和相关性时的分类
- 当你有足够数据来估计可靠的协方差矩阵时
- 制造业质量控制（多变量过程监控）

### Jaccard 相似度（用于集合）

Jaccard 相似度度量两个集合之间的重叠。

```
J(A, B) = |A intersect B| / |A union B|
```

范围从 0（无重叠）到 1（相同集合）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：
- 比较标签、类别或特征的集合
- 基于词出现（而非频率）的文档相似度
- 近似重复检测（MinHash 对 Jaccard 的近似）
- 比较二元特征向量（存在/缺失数据）
- 评估分割模型（交并比 = Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离计算将一个字符串转换为另一个字符串所需的最少单字符操作次数。操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

使用动态规划计算。填充一个矩阵，其中条目 (i, j) 是字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查和纠正
- DNA 序列比对（带加权操作）
- 模糊字符串匹配
- 杂乱文本数据的去重

### KL 散度（不是距离，但被当作距离使用）

KL 散度度量一个概率分布与另一个概率分布的差异。在课程 09 中涵盖，但它属于本次讨论，因为人们将其作为"距离"使用，尽管它不是。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL 散度**不是**对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求。它也不满足三角不等式。它是散度，不是距离。

前向 KL（D_KL(P || Q)）是"均值寻找"的：Q 试图覆盖 P 的所有模态。
反向 KL（D_KL(Q || P)）是"模态寻找"的：Q 专注于 P 的单个模态。

当你看到 KL 散度时：
- VAE（ELBO 中的 KL 项将潜在分布推向先验）
- 知识蒸馏（学生试图匹配教师的分布）
- RLHF（KL 惩罚使微调模型接近基础模型）
- 策略梯度方法（约束策略更新）

### Wasserstein 距离（推土机距离）

Wasserstein 距离度量将一个概率分布转换为另一个概率分布所需的最小"工作量"。可以这样想：如果一个分布是一堆土，另一个是一个坑，你需要移动多少土、多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它简化为累积分布函数绝对差的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Wasserstein 距离为何重要：
- 它是真正的度量（对称的，满足三角不等式）
- 即使分布不重叠，它也能提供梯度（KL 散度会趋于无穷）
- 这一性质使其成为 Wasserstein GAN（WGAN）的核心，解决了原始 GAN 的训练不稳定性

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同距离

| 任务 | 最佳距离 | 原因 |
|------|----------|------|
| 文本相似度 | 余弦 | 大小是噪声，方向是意义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | 鲁棒，不会放大罕见的大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据天然是集合值，不是向量 |
| 字符串匹配 | 编辑距离 | 操作映射到人类编辑直觉 |
| 异常值检测 | 马氏距离 | 考虑特征相关性和尺度 |
| 比较分布 | KL 散度 | 度量使用 Q 代替 P 时丢失的信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| 嵌入（向量数据库） | 余弦或点积 | 嵌入被训练为在方向中编码意义 |
| 推荐系统 | 点积 | 模长可以编码流行度或置信度 |
| DNA 序列 | 加权编辑距离 | 替换成本因核苷酸对而异 |
| 制造质量控制 | L-无穷 | 任何维度中的最坏情况偏差重要 |

### 与损失函数的联系

损失函数是应用于预测值与目标值之间的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化向损失函数添加权重的范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 产生稀疏性而 L2 不产生：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最可能触及菱形的角，此时一个权重为零。它们触及圆的光滑点，此时两个权重都非零。

### 最近邻搜索

每个距离函数都意味着一个最近邻搜索问题：给定一个查询点，在数据集中找到最近的点。

精确最近邻搜索在 n 个点、d 维的数据集中每次查询的复杂度为 O(n * d)。对于大数据集，这太慢了。

近似最近邻（ANN）算法以少量精度换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（分层可导航小世界）是现代向量数据库中的主导算法。它构建一个多层图，其中每个节点连接到其近似最近邻。搜索从顶层（稀疏，长跳跃）开始，下降到最底层（密集，短跳跃）。

## 构建

### 步骤 1：所有范数和距离函数

参见 `code/distances.py` 获取完整实现。每个函数仅使用基本 Python 数学从零构建。

### 步骤 2：相同数据，不同距离，不同邻居

`distances.py` 中的演示创建一个数据集，选取一个查询点，并展示最近邻如何随距离度量变化。在 L1 下"最近"的点可能在 L2 或余弦下不是最近的。

### 步骤 3：嵌入相似度搜索

代码包含一个模拟嵌入相似度搜索，使用余弦相似度 vs L2 距离查找与查询最相似的"文档"，展示排名可能不同。

## 应用

最常见的实际用途：在向量数据库中查找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索向量数据库时，这就是底层发生的事情。嵌入模型将文本映射为向量。向量数据库计算查询向量与每个存储向量之间的余弦相似度（或点积），使用 ANN 算法避免检查所有向量。

## 练习

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-无穷距离。验证对于任意点对，L-无穷 <= L2 <= L1 始终成立。证明为什么这个顺序是保证的。

2. 创建两个向量，其中余弦相似度高（> 0.9）但 L2 距离大（> 10）。从几何上解释发生了什么。然后创建两个向量，其中余弦相似度低（< 0.3）但 L2 距离小（< 0.5）。

3. 实现一个函数，接收数据集和查询点，返回 L1、L2、余弦和马氏距离下的最近邻。找到一个四个度量对最近邻意见不一致的数据集。

4. 使用 CDF 方法手工计算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。然后计算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。哪个更大，为什么？

5. 实现 MinHash 进行近似 Jaccard 相似度计算。生成 100 个随机集合，计算所有对的精确 Jaccard，并与使用 50、100 和 200 个哈希函数的 MinHash 近似进行比较。绘制近似误差。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 范数 | "向量的大小" | 将向量映射到非负标量的函数，满足三角不等式、绝对齐次性，且仅对零向量为零 |
| L1 范数 | "曼哈顿距离" | 绝对分量值之和。在优化中产生稀疏性。对异常值鲁棒 |
| L2 范数 | "欧几里得距离" | 分量平方和的平方根。欧几里得空间中的直线距离 |
| Lp 范数 | "广义范数" | 绝对分量的 p 次幂之和的 p 次方根。L1 和 L2 是特例 |
| L-无穷范数 | "最大范数"或"切比雪夫距离" | 最大绝对分量值。Lp 当 p 趋近无穷大时的极限 |
| 余弦相似度 | "向量间的夹角" | 点积除以两个模长。范围从 -1 到 +1。忽略向量长度 |
| 余弦距离 | "1 减余弦相似度" | 将余弦相似度转换为距离。范围从 0 到 2 |
| 点积 | "未归一化的余弦" | 分量乘积之和。等于余弦相似度乘以两个模长 |
| 马氏距离 | "相关性感知距离" | 在使用数据协方差矩阵进行白化（去相关和归一化）的空间中的 L2 距离 |
| Jaccard 相似度 | "集合重叠" | 交集大小除以并集大小。用于集合，不是向量 |
| 编辑距离 | "Levenshtein 距离" | 将一个字符串转换为另一个字符串的最少插入、删除和替换次数 |
| KL 散度 | "分布之间的距离" | 不是真正的距离（不对称）。度量使用 Q 编码 P 时的额外比特数 |
| Wasserstein 距离 | "推土机距离" | 将一个分布的质量传输到另一个分布的最小工作量。真正的度量 |
| 近似最近邻 | "ANN 搜索" | 以少量精度换取巨大速度提升的算法（HNSW、LSH、IVF），比精确搜索快得多 |
| HNSW | "向量数据库算法" | 分层可导航小世界图。用于快速近似最近邻搜索的多层图 |
| L1 正则化 | "Lasso" | 向损失函数添加权重的 L1 范数。将权重驱至零（稀疏性） |
| L2 正则化 | "Ridge"或"权重衰减" | 向损失函数添加权重的平方 L2 范数。将权重向零收缩而不产生稀疏性 |
| Elastic Net | "L1 + L2" | 结合 L1 和 L2 正则化。比单独使用更好地处理相关特征组 |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的十亿级 ANN 搜索库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将推土机距离引入 GAN 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基础 ANN 算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，余弦相似度成为嵌入默认度量的地方
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量和邻居算法的实用指南
