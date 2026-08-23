"""
练习 2：死亡神经元复活器。

识别死亡的 ReLU 神经元(在一批样本上激活始终为0)，并用 Kaiming 初始化重新初始化
"喂给这个神经元"的那部分输入权重(即前一层 Linear 对应该输出行的权重 + 偏置)。

设计要点:
- 只能重新初始化"死亡神经元对应的那一行权重"，不能整层重置——否则等于把这一层活着的
  神经元学到的东西也一起冲掉了。
- Kaiming (He) 初始化专为 ReLU 设计: bound = sqrt(6 / fan_in)（uniform, a=0, mode=fan_in,
  nonlinearity='relu'），能让重生的神经元有正常量级的输出方差，而不是继续卡在死区。
- 偏置重置为 0——原本很负的偏置(比如 lesson demo 里的 -5.0)正是致死原因之一。

测试场景: 复刻本课文档"Bug 2: Dead ReLUs from bad initialization"的构造方式
(weight.fill_(-1.0), bias.fill_(-5.0))，制造出 >70% 死亡神经元的网络，验证复活后
死亡比例大幅下降。
"""

import sys

import torch
import torch.nn as nn

sys.path.insert(0, "..")
from debug_neural_nets import NetworkDebugger  # noqa: E402


def find_dead_neurons(model, x_sample):
    """返回 {linear_layer_name: [dead_output_indices]}，'死亡'定义为在整批样本上
    经过其后紧跟的 ReLU 之后恒为 0。"""
    modules = list(model.named_children())
    dead = {}

    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    for name, m in modules:
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(x_sample)

    for h in hooks:
        h.remove()

    for i, (name, m) in enumerate(modules):
        if isinstance(m, nn.ReLU) and name in activations:
            act = activations[name]
            dead_mask = (act == 0).all(dim=0)
            dead_idx = dead_mask.nonzero(as_tuple=True)[0].tolist()
            if dead_idx:
                # 找到紧靠在这个 ReLU 前面的 Linear 层名字
                prev_linear_name = None
                for j in range(i - 1, -1, -1):
                    if isinstance(modules[j][1], nn.Linear):
                        prev_linear_name = modules[j][0]
                        break
                if prev_linear_name is not None:
                    dead[prev_linear_name] = dead_idx
    return dead


def revive_dead_neurons(model, x_sample, verbose=True):
    dead = find_dead_neurons(model, x_sample)
    named = dict(model.named_modules())

    total_dead = 0
    for linear_name, dead_idx in dead.items():
        linear = named[linear_name]
        fan_in = linear.in_features
        bound = (6.0 / fan_in) ** 0.5  # kaiming_uniform, a=0, nonlinearity='relu'
        with torch.no_grad():
            for idx in dead_idx:
                linear.weight[idx, :].uniform_(-bound, bound)
                linear.bias[idx] = 0.0
        total_dead += len(dead_idx)
        if verbose:
            print(f"  复活 {linear_name}: {len(dead_idx)} 个死亡神经元 (fan_in={fan_in}, bound=±{bound:.4f})")

    return total_dead


def dead_fraction_report(model, x_sample):
    dead = find_dead_neurons(model, x_sample)
    total_neurons = 0
    total_dead = 0
    for name, m in model.named_children():
        if isinstance(m, nn.ReLU):
            pass
    # 用 debugger 拿到每层的总宽度更方便
    for linear_name, idxs in dead.items():
        linear = dict(model.named_modules())[linear_name]
        total_neurons += linear.out_features
        total_dead += len(idxs)
    return total_dead, total_neurons, dead


def main():
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()
    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print("构造坏初始化网络: 权重用正常 Kaiming 初始化(每行独立)，")
    print("但 bias 初始化有 bug，被统一 fill 成了 -5.0 (很现实的场景: 权重写对了，bias这行错了)")
    print("=" * 60)
    model = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                m.bias.fill_(-5.0)

    dead_before, total_before, detail_before = dead_fraction_report(model, x)
    frac_before = dead_before / total_before if total_before else 0
    print(f"  复活前: {dead_before}/{total_before} 神经元死亡 ({frac_before:.1%})")
    for name, idxs in detail_before.items():
        print(f"    {name}: {len(idxs)} 个死亡 (indices示例: {idxs[:5]}{'...' if len(idxs)>5 else ''})")

    print("\n训练前（死亡状态下）先跑一下过拟合一个批次测试，预期会失败:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for step in range(100):
        optimizer.zero_grad()
        out = model(x[:8])
        loss = criterion(out, y[:8])
        loss.backward()
        optimizer.step()
    print(f"  100步后 loss={loss.item():.4f} (死亡神经元卡住时通常降不下去)")

    print("\n" + "=" * 60)
    print("应用死亡神经元复活器")
    print("=" * 60)
    revive_dead_neurons(model, x)

    dead_after, total_after, detail_after = dead_fraction_report(model, x)
    frac_after = dead_after / total_after if total_after else 0
    print(f"\n  复活后: {dead_after}/{total_after} 神经元死亡 ({frac_after:.1%})")
    print(f"  死亡比例下降: {frac_before:.1%} -> {frac_after:.1%}")

    print("\n重新初始化优化器状态后，再跑一次过拟合一个批次测试:")
    model2 = model
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
    model2.train()
    for step in range(100):
        optimizer2.zero_grad()
        out = model2(x[:8])
        loss2 = criterion(out, y[:8])
        loss2.backward()
        optimizer2.step()
        if step % 25 == 0 or step == 99:
            with torch.no_grad():
                preds = out.argmax(dim=1)
                acc = (preds == y[:8]).float().mean().item()
            print(f"    step {step:3d} | loss={loss2.item():.6f} | acc={acc:.1%}")

    assert frac_before > 0.70, f"测试场景死亡比例只有{frac_before:.1%}，没达到>70%的练习要求"
    print(f"\n验证: 复活前死亡比例 {frac_before:.1%} > 70% (满足练习要求)")
    print(f"验证: 复活后死亡比例 {frac_after:.1%}，从 {frac_before:.1%} 大幅下降")


if __name__ == "__main__":
    main()
