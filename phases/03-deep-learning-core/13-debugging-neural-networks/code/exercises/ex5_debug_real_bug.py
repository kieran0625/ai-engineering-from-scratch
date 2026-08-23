"""
练习 5：调试真实故障。

从第10课的纯 Python mini framework 出发，注入一个"转置权重矩阵"的隐蔽 bug：
Linear.backward() 里 input_grad[j] += grad[i] * weights[i][j] 被写成
weights[j][i]（只在方阵情况下不会直接崩溃/越界，是最阴险的那种 bug——
形状对得上，跑起来不报错，但反传的梯度是错的，训练看起来"差不多能收敛"实际却学歪了）。

用纯 Python 有限差分实现梯度检查器（对接框架 parameters() 返回的
(container, i, j, grad_container) 位置引用元组），精确定位到底是哪一层、哪个参数
梯度算错了。
"""

import copy
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "10-mini-framework", "code"))
from main import Linear, ReLU, Sequential, MSELoss  # noqa: E402


class BuggyLinear(Linear):
    """故意注入的 bug: backward() 里把 weights[i][j] 写成了 weights[j][i]。

    只在 fan_in == fan_out (方阵) 时才不会越界崩溃——这正是这类 bug 最阴险的地方：
    形状检查完全通过，前向传播完全正常，训练能跑、loss 甚至可能下降，
    但反传给上一层的梯度方向是错的。
    """

    def backward(self, grad):
        assert self.fan_in == self.fan_out, "bug 演示要求方阵，否则会直接越界崩溃"
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[j][i]  # <-- BUG: 应该是 weights[i][j]
        return input_grad


def build_model(buggy=False):
    """2 -> 8 -> 8 -> 1，中间那层 8->8 是方阵，用来注入 bug。"""
    hidden = BuggyLinear(8, 8) if buggy else Linear(8, 8)
    return Sequential(
        Linear(2, 8),
        ReLU(),
        hidden,
        ReLU(),
        Linear(8, 1),
    )


def gradient_check_framework(model, x, t, criterion, eps=1e-4, rel_diff_threshold=1e-3):
    """对框架里每一个 (container, i, j, grad_container) 参数做有限差分梯度检查。

    数值梯度: (loss(w+eps) - loss(w-eps)) / (2*eps)
    解析梯度: 走一次真正的 forward+backward 拿到的 grad_container 里的值
    用相对误差 |num - ana| / max(|num|, |ana|, 1e-8) 比较，> threshold 判定为可疑。
    """

    def compute_loss():
        pred = model.forward(x)
        return criterion(pred, t)

    # 解析梯度: 正常走一遍 forward + backward
    for module in model.modules:
        if hasattr(module, "weight_grads"):
            for row in module.weight_grads:
                for k in range(len(row)):
                    row[k] = 0.0
            for k in range(len(module.bias_grads)):
                module.bias_grads[k] = 0.0

    pred = model.forward(x)
    loss = criterion(pred, t)
    grad = criterion.backward()
    model.backward(grad)

    results = []
    for layer_idx, module in enumerate(model.modules):
        if not hasattr(module, "weights"):
            continue
        layer_name = f"layer[{layer_idx}]({module.fan_in}->{module.fan_out})"

        for i in range(module.fan_out):
            for j in range(module.fan_in):
                analytical = module.weight_grads[i][j]

                orig = module.weights[i][j]
                module.weights[i][j] = orig + eps
                loss_plus = compute_loss()
                module.weights[i][j] = orig - eps
                loss_minus = compute_loss()
                module.weights[i][j] = orig

                numerical = (loss_plus - loss_minus) / (2 * eps)
                rel_diff = abs(numerical - analytical) / max(abs(numerical), abs(analytical), 1e-8)

                results.append({
                    "layer": layer_name, "param": f"weight[{i}][{j}]",
                    "analytical": analytical, "numerical": numerical, "rel_diff": rel_diff,
                })

        for i in range(module.fan_out):
            analytical = module.bias_grads[i]

            orig = module.biases[i]
            module.biases[i] = orig + eps
            loss_plus = compute_loss()
            module.biases[i] = orig - eps
            loss_minus = compute_loss()
            module.biases[i] = orig

            numerical = (loss_plus - loss_minus) / (2 * eps)
            rel_diff = abs(numerical - analytical) / max(abs(numerical), abs(analytical), 1e-8)

            results.append({
                "layer": layer_name, "param": f"bias[{i}]",
                "analytical": analytical, "numerical": numerical, "rel_diff": rel_diff,
            })

    # 检查完后模型的 grad state 是"最后一次数值扰动"留下的脏状态，重新跑一次干净的
    # forward+backward，确保调用方拿到的模型梯度是正确的解析梯度
    for module in model.modules:
        if hasattr(module, "weight_grads"):
            for row in module.weight_grads:
                for k in range(len(row)):
                    row[k] = 0.0
            for k in range(len(module.bias_grads)):
                module.bias_grads[k] = 0.0
    pred = model.forward(x)
    loss = criterion(pred, t)
    grad = criterion.backward()
    model.backward(grad)

    bad = [r for r in results if r["rel_diff"] > rel_diff_threshold]
    return results, bad, loss


def main():
    import random
    random.seed(0)

    x = [random.uniform(-1, 1) for _ in range(2)]
    t = [random.uniform(-1, 1) for _ in range(1)]
    criterion = MSELoss()

    print("=" * 70)
    print("第一步: 在'健康'模型上跑梯度检查，确认检查器本身是可信的")
    print("=" * 70)
    healthy_model = build_model(buggy=False)
    results, bad, loss = gradient_check_framework(healthy_model, x, t, criterion)
    max_rel = max(r["rel_diff"] for r in results)
    print(f"  loss={loss:.6f} | 参数总数={len(results)} | 最大相对误差={max_rel:.2e} | 可疑参数数={len(bad)}")
    assert not bad, "健康模型不应该有梯度检查失败的参数"
    print("  健康模型全部通过 (< 1e-3) -- 梯度检查器本身可信\n")

    print("=" * 70)
    print("第二步: 注入 bug (中间 Linear(8,8) 层 backward() 里 weights[i][j] 写成 weights[j][i])")
    print("跑同样的梯度检查，定位出问题的层/参数")
    print("=" * 70)
    buggy_model = build_model(buggy=True)
    results2, bad2, loss2 = gradient_check_framework(buggy_model, x, t, criterion)
    max_rel2 = max(r["rel_diff"] for r in results2)
    print(f"  loss={loss2:.6f} | 参数总数={len(results2)} | 最大相对误差={max_rel2:.2e} | 可疑参数数={len(bad2)}\n")

    by_layer = {}
    for r in results2:
        by_layer.setdefault(r["layer"], []).append(r)

    print("  按层汇总:")
    for layer_name, rs in by_layer.items():
        layer_bad = [r for r in rs if r["rel_diff"] > 1e-3]
        layer_max = max(r["rel_diff"] for r in rs)
        status = f"{len(layer_bad)}/{len(rs)} 个参数梯度错误" if layer_bad else "全部正常"
        print(f"    {layer_name:20s} | 最大相对误差={layer_max:10.2e} | {status}")

    print(f"\n  定位结果: {len(bad2)} 个参数梯度检查失败，全部集中在:")
    bad_layers = sorted({r["layer"] for r in bad2})
    for bl in bad_layers:
        print(f"    -> {bl}  <-- 存在 bug 的那一层")

    print("\n  抽样看几个具体参数的 解析梯度 vs 数值梯度 对比:")
    for r in bad2[:5]:
        print(f"    {r['layer']} {r['param']:12s} | 解析={r['analytical']:10.5f} | "
              f"数值={r['numerical']:10.5f} | 相对误差={r['rel_diff']:.2e}")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    assert bad_layers == [f"layer[2](8->8)"], f"预期只有 layer[2] 出问题，实际: {bad_layers}"
    print("  梯度检查精确定位到 layer[2] (中间的 Linear(8,8)) -- 与注入 bug 的层完全吻合。")
    print("  其余层 (layer[0], layer[4]) 全部通过检查，说明 bug 不会'扩散式'污染整个网络的")
    print("  检测结果——每一层的梯度是独立可验证的，这也是'逐层梯度检查'比'只看最终loss'")
    print("  更能精确定位 bug 的原因: loss 可能看起来'差不多'在下降，但某一层的梯度方向")
    print("  已经错了，只是还没错到让训练彻底失败的地步。")


if __name__ == "__main__":
    main()
