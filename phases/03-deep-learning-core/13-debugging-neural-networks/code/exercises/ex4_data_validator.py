"""
练习 4：数据管道验证器。

检查四类常见的数据管道问题:
1. 训练/测试集之间的重复样本 (数据泄漏最常见的来源之一)
2. 标签分布不平衡 (>10:1 视为需要警惕)
3. 输入归一化 (均值应接近0，标准差应接近1 —— 呼应第08课 Xavier/Kaiming 的
   单位方差假设)
4. 数据中的 NaN/Inf 值

在一个故意构造的、四类问题全部具备的损坏数据集上运行，验证每一类都能被抓出来。
"""

import torch


def check_duplicates(x_train, x_test, atol=1e-6):
    issues = []
    # 用四舍五入后的元组做 hash 集合比较，量级上比逐对比较快很多
    train_set = {tuple(round(v, 4) for v in row.tolist()) for row in x_train}
    test_set = {tuple(round(v, 4) for v in row.tolist()) for row in x_test}
    overlap = train_set & test_set
    if overlap:
        issues.append(
            f"DUPLICATE_SAMPLES: train/test 之间有 {len(overlap)} 个重复样本 "
            f"({len(overlap) / len(test_set):.1%} of test set) -- 数据泄漏风险"
        )
    return issues


def check_label_imbalance(y, ratio_threshold=10.0):
    issues = []
    values, counts = torch.unique(y, return_counts=True)
    counts = counts.tolist()
    values = values.tolist()
    max_c, min_c = max(counts), min(counts)
    ratio = max_c / min_c if min_c > 0 else float("inf")
    if ratio > ratio_threshold:
        dist = {int(v): c for v, c in zip(values, counts)}
        issues.append(
            f"LABEL_IMBALANCE: 类别分布比例 {ratio:.1f}:1 (>{ratio_threshold}:1 阈值), "
            f"分布={dist}"
        )
    return issues


def check_normalization(x, mean_tol=0.1, std_tol=0.15):
    issues = []
    mean = x.mean().item()
    std = x.std().item()
    if abs(mean) > mean_tol or abs(std - 1.0) > std_tol:
        issues.append(
            f"NOT_NORMALIZED: 整体 mean={mean:.4f} (期望~0), std={std:.4f} (期望~1) "
            f"-- 会破坏 Xavier/Kaiming 的单位方差假设 (第08课)"
        )
    return issues


def check_nan_inf(x, y, name="data"):
    issues = []
    n_nan_x = torch.isnan(x).sum().item()
    n_inf_x = torch.isinf(x).sum().item()
    n_nan_y = torch.isnan(y.float()).sum().item() if y.dtype.is_floating_point else 0
    if n_nan_x:
        issues.append(f"NAN_IN_INPUT: {name} 输入中有 {n_nan_x} 个 NaN 值")
    if n_inf_x:
        issues.append(f"INF_IN_INPUT: {name} 输入中有 {n_inf_x} 个 Inf 值")
    if n_nan_y:
        issues.append(f"NAN_IN_LABEL: {name} 标签中有 {n_nan_y} 个 NaN 值")
    return issues


def validate_data_pipeline(x_train, y_train, x_test, y_test):
    issues = []
    issues += check_nan_inf(x_train, y_train, "train")
    issues += check_nan_inf(x_test, y_test, "test")
    # NaN/Inf 存在时，其它统计量(mean/std/去重)会被污染，先单独跑、再跳过后续统计类检查
    if not issues:
        issues += check_duplicates(x_train, x_test)
        issues += check_label_imbalance(y_train)
        issues += check_normalization(x_train)
    else:
        issues.append("NOTE: 检测到 NaN/Inf，跳过依赖数值统计的检查(去重/不平衡/归一化)，先修 NaN/Inf")
    return issues if issues else ["HEALTHY"]


def make_corrupted_dataset():
    torch.manual_seed(0)

    n_train, n_test, n_features = 500, 100, 8

    # 未归一化: 均值50，标准差20 (远离 mean=0/std=1)
    x_train = torch.randn(n_train, n_features) * 20 + 50
    x_test = torch.randn(n_test, n_features) * 20 + 50

    # 标签严重不平衡: 95% 类别0, 5% 类别1
    y_train = torch.zeros(n_train, dtype=torch.long)
    n_pos = int(n_train * 0.05)
    y_train[:n_pos] = 1
    y_train = y_train[torch.randperm(n_train)]

    y_test = torch.zeros(n_test, dtype=torch.long)
    y_test[: int(n_test * 0.05)] = 1
    y_test = y_test[torch.randperm(n_test)]

    # 数据泄漏: 把 test 的前10行直接复制成 train 的最后10行
    x_train[-10:] = x_test[:10]

    # 撒几个 NaN / Inf 进去
    x_train[3, 2] = float("nan")
    x_train[7, 5] = float("inf")

    return x_train, y_train, x_test, y_test


def make_healthy_dataset():
    torch.manual_seed(1)
    n_train, n_test, n_features = 500, 100, 8
    x_train = torch.randn(n_train, n_features)
    x_test = torch.randn(n_test, n_features)
    y_train = torch.randint(0, 2, (n_train,))
    y_test = torch.randint(0, 2, (n_test,))
    return x_train, y_train, x_test, y_test


def main():
    print("=" * 60)
    print("在故意损坏的数据集上运行验证器 (四类问题全部具备)")
    print("=" * 60)
    x_train, y_train, x_test, y_test = make_corrupted_dataset()
    issues = validate_data_pipeline(x_train, y_train, x_test, y_test)
    for issue in issues:
        print(f"  {issue}")

    print("\n" + "=" * 60)
    print("单独重跑 NaN/Inf 修复后的检查 (验证其余三类检查独立生效)")
    print("=" * 60)
    x_train_clean_nan = x_train.clone()
    x_train_clean_nan[3, 2] = 0.0
    x_train_clean_nan[7, 5] = 0.0
    issues2 = validate_data_pipeline(x_train_clean_nan, y_train, x_test, y_test)
    for issue in issues2:
        print(f"  {issue}")

    print("\n" + "=" * 60)
    print("对照组: 在健康数据集上运行，验证不会误报")
    print("=" * 60)
    x_train_h, y_train_h, x_test_h, y_test_h = make_healthy_dataset()
    issues3 = validate_data_pipeline(x_train_h, y_train_h, x_test_h, y_test_h)
    for issue in issues3:
        print(f"  {issue}")


if __name__ == "__main__":
    main()
