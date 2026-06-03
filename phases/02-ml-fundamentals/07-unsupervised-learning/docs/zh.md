# 无监督学习

> 没有标签，没有老师。算法自己发现结构。

**类型：** Build
**语言：** Python
**前置知识：** Phase 1（范数与距离、概率与分布），Phase 2 Lessons 1-6
**时间：** ~90 分钟

## 学习目标

- 从零实现 K-Means、DBSCAN 和高斯混合模型（GMM），并比较它们的聚类行为
- 使用轮廓系数和肘部法评估聚类质量，选择最优 K 值
- 解释 DBSCAN 何时优于 K-Means，并识别哪种算法能处理非球形聚类和异常值
- 使用聚类方法构建异常检测流水线，标记偏离正常模式的点

## 问题

到目前为止，每一节 ML 课程都假设数据有标签："这是输入，这是正确输出。" 在现实世界中，标签很昂贵。一家医院有数百万份患者记录，但没有人手动为每一份标注疾病类别。一个电商网站有数百万次用户会话，但没有人手动标注客户细分。一个安全团队有网络日志，但没有人标记每一个异常。

无监督学习在没有被告知要找什么的情况下发现模式。它将相似的数据点分组，发现隐藏的结构，并浮现异常。如果说监督学习是从带答案的课本中学习，那么无监督学习就是盯着原始数据直到模式自己显现。

难点在于：没有标签，你无法直接衡量"对"或"错"。你需要不同的工具来评估算法发现的结构是否有意义。

## 概念

### 聚类：将相似的事物分组

聚类将每个数据点分配到一个组（簇），使得同一组内的点彼此比与其他组的点更相似。问题始终是："相似"是什么意思？

```mermaid
flowchart LR
    A[Raw Data] --> B{Choose Method}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[Hierarchical]
    B --> F[GMM]
    C --> G[Flat, spherical clusters]
    D --> H[Arbitrary shapes, noise detection]
    E --> I[Tree of nested clusters]
    F --> J[Soft assignments, elliptical clusters]
```

### K-Means：主力军

K-Means 将数据划分为恰好 K 个簇。每个簇有一个质心（其质量中心），每个点属于最近的质心。

Lloyd 算法：

1. 随机选择 K 个点作为初始质心
2. 将每个数据点分配到最近的质心
3. 将每个质心重新计算为其分配点的均值
4. 重复步骤 2-3，直到分配不再变化

目标函数（惯性）衡量每个点到其分配质心的总平方距离。K-Means 最小化这个值，但只能找到局部最小值。不同的初始化可能产生不同的结果。

### 选择 K

两种标准方法：

**肘部法：** 对 K = 1, 2, 3, ..., n 运行 K-Means。绘制惯性 vs K 的图。寻找"肘部"，即增加更多簇不再显著降低惯性的点。

**轮廓系数：** 对每个点，衡量它与其自身簇的相似度（a）与最近的其他簇的相似度（b）相比如何。轮廓系数为 (b - a) / max(a, b)，范围从 -1（错误的簇）到 +1（聚类良好）。对所有点取平均得到全局分数。

### DBSCAN：基于密度的聚类

K-Means 假设簇是球形的，并且要求你预先选择 K。DBSCAN 不做这两种假设。它将簇发现为被稀疏区域分隔的密集区域。

两个参数：
- **eps**：邻域的半径
- **min_samples**：形成密集区域所需的最少点数

三种点类型：
- **核心点**：在 eps 距离内至少有 min_samples 个点
- **边界点**：在核心点的 eps 范围内，但本身不是核心点
- **噪声点**：既不是核心点也不是边界点。这些是异常值。

DBSCAN 将彼此在 eps 范围内的核心点连接为同一个簇。边界点加入附近核心点的簇。噪声点不属于任何簇。

优点：发现任意形状的簇，自动确定簇的数量，识别异常值。缺点：难以处理密度变化的簇。

### 层次聚类

构建嵌套簇的树（树状图）。

凝聚式（自底向上）：
1. 从每个点作为独立的簇开始
2. 合并两个最近的簇
3. 重复直到只剩一个簇
4. 在期望的层次切割树状图以获得 K 个簇

簇之间的"接近度"可以衡量为：
- **单链接**：两个簇中任意两点之间的最小距离
- **全链接**：两个簇中任意两点之间的最大距离
- **平均链接**：所有点对之间的平均距离
- **Ward 法**：使总簇内方差增加最小的合并

### 高斯混合模型（GMM）

K-Means 给出硬分配：每个点恰好属于一个簇。GMM 给出软分配：每个点有属于每个簇的概率。

GMM 假设数据由 K 个高斯分布的混合生成，每个分布有自己的均值和协方差。期望最大化（EM）算法在以下两步之间交替：

- **E 步**：计算每个点属于每个高斯的概率
- **M 步**：更新每个高斯的均值、协方差和混合权重，以最大化数据的似然

GMM 可以建模椭圆形簇（不只是像 K-Means 那样的球形），并自然处理重叠的簇。

### 何时使用哪种方法

| 方法 | 最适合 | 避免使用 |
|------|--------|----------|
| K-Means | 大数据集、球形簇、已知 K | 不规则形状、存在异常值 |
| DBSCAN | 未知 K、任意形状、异常值检测 | 密度变化、维度很高 |
| 层次聚类 | 小数据集、需要树状图、未知 K | 大数据集（O(n²) 内存） |
| GMM | 重叠簇、需要软分配 | 非常大的数据集、维度太多 |

### 基于聚类的异常检测

聚类天然支持异常检测：
- **K-Means**：远离任何质心的点是异常
- **DBSCAN**：噪声点按定义就是异常
- **GMM**：在所有高斯下概率低的点是异常

## 动手实现

### 步骤 1：从零实现 K-Means

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 步骤 2：肘部法和轮廓系数

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### 步骤 3：从零实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 步骤 4：高斯混合模型（EM 算法）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 步骤 5：生成测试数据并运行所有内容

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

## 使用现成库

使用 scikit-learn，同样的算法只需一行代码：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

从零实现的版本让你确切了解这些库计算了什么。K-Means 在分配和重新计算之间迭代。DBSCAN 从密集种子生长簇。GMM 在期望和最大化之间交替。库版本增加了数值稳定性、更智能的初始化（K-Means++）和 GPU 加速，但核心逻辑相同。

## 交付成果

本节课产出了 K-Means、DBSCAN 和 GMM 的从零实现。聚类代码可以作为更高级无监督方法的基础重复使用。

## 练习

1. 实现 K-Means++ 初始化：不随机选择质心，而是随机选择第一个，后续每个质心以与其到最近现有质心平方距离成正比的概率选择。比较与随机初始化的收敛速度。
2. 在代码中添加层次凝聚聚类。实现 Ward 链接并生成树状图（作为嵌套的合并列表）。在不同层次切割并与 K-Means 结果比较。
3. 构建简单的异常检测流水线：在同一数据上运行 DBSCAN 和 GMM，标记两种方法都同意是异常值的点（DBSCAN 中的噪声，GMM 中的低概率）。衡量重叠并讨论方法何时不一致。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Clustering | "将相似的东西分组" | 将数据划分为子集，使得组内相似度超过组间相似度，由特定的距离度量来衡量 |
| Centroid | "簇的中心" | 分配给簇的所有点的均值；K-Means 用它作为簇的代表 |
| Inertia | "簇有多紧密" | 每个点到其分配质心的平方距离之和；越低越紧密 |
| Silhouette score | "簇分离得有多好" | 对每个点，(b - a) / max(a, b)，其中 a 是平均簇内距离，b 是平均最近簇距离 |
| Core point | "密集区域中的点" | 在 DBSCAN 中，在 eps 距离内有至少 min_samples 个邻居的点 |
| EM algorithm | "软 K-Means" | 期望最大化：迭代计算隶属概率（E 步）并更新分布参数（M 步） |
| Dendrogram | "簇的树" | 树状图，显示层次聚类中簇合并的顺序和距离 |
| Anomaly | "异常值" | 不符合预期模式的数据点，被 DBSCAN 识别为噪声或被 GMM 识别为低概率点 |

## 延伸阅读

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng 关于聚类和 EM 的讲义
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - 所有聚类算法的实用比较，附可视化示例
- [DBSCAN original paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 引入基于密度聚类的论文
