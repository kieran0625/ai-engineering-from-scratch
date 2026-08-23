"""
练习 1：给 NetworkDebugger 添加梯度爆炸检测器。

设计:
- 每次 backward 后，计算所有参数梯度的整体 L2 范数（跟 torch.nn.utils.clip_grad_norm_
  内部用的公式一样：sqrt(sum(grad_i.norm()^2 for grad_i in all grads))）。
- 维护一个"健康期"范数历史（explosion 判定之前的范数），用来估计一个合理的裁剪阈值。
- 当当前范数超过 explosion_threshold，或超过健康期中位数的 explosion_ratio 倍时，判定为
  梯度爆炸，并自动建议一个裁剪值：健康期范数中位数的 explosion_ratio/2 倍（经验规则——
  裁剪到"仍然明显大于正常水平但远小于爆炸点"的量级，给优化器一点缓冲而不是死死摁在中位数上）。

测试场景：20 层 Linear+Tanh 网络，无 BatchNorm/LayerNorm/残差连接，用偏大的初始化
（std=1.5，远超 Xavier 的量级）来制造真实的梯度爆炸。
"""

import math
import sys

import torch 
import torch.nn as nn
   
sys.path.insert(0, "..") 
from ..debug_neural_nets import NetworkDebugger  # noqa: E402

class GradientExplosionDetector:
    """挂在一个已有 NetworkDebugger 上，额外追踪整体梯度范数并给出裁剪建议。

    healthy_norms 可以预先用一个"已知正常"的校准跑（比如同架构但用合理初始化）填充，
    这样即使被监控的模型从第一步就爆炸，也依然有基线可以比较——不需要等它自己先"健康"
    几步。这更贴近真实工作流：你通常是拿一个跑得动的 baseline 校准出正常范围。
    """

    def __init__(self, model, explosion_threshold=50.0, explosion_ratio=10.0,
                 warmup_steps=5, healthy_baseline=None):
        self.model = model
        self.explosion_threshold = explosion_threshold
        self.explosion_ratio = explosion_ratio
        self.warmup_steps = warmup_steps
        self.norm_history = []
        self.healthy_norms = list(healthy_baseline) if healthy_baseline else []
        self.explosion_steps = []

    def record_step(self, step):
        """在 loss.backward() 之后、optimizer.step() 之前调用。"""
        total_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().float().pow(2).sum().item()
        norm = math.sqrt(total_sq)
        self.norm_history.append(norm)

        is_exploding = False
        reason = None
        if norm > self.explosion_threshold:
            is_exploding = True
            reason = f"norm={norm:.2f} > 绝对阈值 {self.explosion_threshold}"
        elif len(self.healthy_norms) >= self.warmup_steps:
            median = sorted(self.healthy_norms)[len(self.healthy_norms) // 2]
            if median > 0 and norm > median * self.explosion_ratio:
                is_exploding = True
                reason = f"norm={norm:.2f} > 健康期中位数({median:.4f}) x {self.explosion_ratio}"

        if is_exploding:
            self.explosion_steps.append((step, norm, reason))
        else:
            self.healthy_norms.append(norm)

        return is_exploding, norm

    def suggested_clip_value(self):
        if not self.healthy_norms:
            return None
        median = sorted(self.healthy_norms)[len(self.healthy_norms) // 2]
        return median * (self.explosion_ratio / 2)

    def report(self):
        print("\n=== 梯度爆炸检测报告 ===")
        print(f"  总步数: {len(self.norm_history)}")
        print(f"  健康期步数: {len(self.healthy_norms)}")
        print(f"  检测到爆炸的步数: {len(self.explosion_steps)}")
        if self.explosion_steps:
            first_step, first_norm, first_reason = self.explosion_steps[0]
            print(f"  首次爆炸: step {first_step}, norm={first_norm:.2f} ({first_reason})")
        suggestion = self.suggested_clip_value()
        if suggestion is not None:
            print(f"  建议的 clip_grad_norm_ 裁剪阈值: {suggestion:.4f}")
            print(f"  (用法: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm={suggestion:.4f}))")
        else:
            print("  健康期样本不足，无法给出裁剪建议")


def make_deep_unnormalized_net(depth=20, width=64, init_std=None):
    """init_std=None -> 用 Xavier（对 Tanh 合理的增益）; 否则用给定 std 的正态分布强行破坏初始化。"""
    layers = []
    layers.append(nn.Linear(10, width))
    layers.append(nn.Tanh())
    for _ in range(depth - 2):
        layers.append(nn.Linear(width, width))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(width, 2))
    model = nn.Sequential(*layers)

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if init_std is None:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=init_std)
                nn.init.zeros_(m.bias)
    return model


def calibrate_healthy_baseline(depth=20, width=64, steps=10):
    """用同架构 + Xavier 正常初始化跑几步，采集"健康"梯度范数分布，作为校准基线。"""
    torch.manual_seed(1)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    criterion = nn.CrossEntropyLoss()

    model = make_deep_unnormalized_net(depth=depth, width=width, init_std=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    norms = []
    for step in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        total_sq = sum(p.grad.detach().float().pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        norms.append(math.sqrt(total_sq))
        optimizer.step()
    return norms


def main():
    print("=" * 60)
    print("第0步: 用 Xavier 初始化(对Tanh合理) 跑 10 步，校准'健康'梯度范数基线")
    print("=" * 60)
    healthy_norms = calibrate_healthy_baseline(depth=20, width=64, steps=10)
    print(f"  健康基线梯度范数: {[f'{n:.4f}' for n in healthy_norms]}")
    print(f"  健康基线中位数: {sorted(healthy_norms)[len(healthy_norms)//2]:.4f}")

    torch.manual_seed(0)
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    criterion = nn.CrossEntropyLoss()

    model = make_deep_unnormalized_net(depth=20, width=64, init_std=1.5)
    debugger = NetworkDebugger(model)
    detector = GradientExplosionDetector(
        model, explosion_threshold=1000.0, explosion_ratio=10.0,
        healthy_baseline=healthy_norms,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("\n" + "=" * 60)
    print("20层 Linear+Tanh 网络，无归一化，init std=1.5（远超Xavier量级）")
    print("=" * 60)

    for step in range(15):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        debugger.record_loss(loss.item())

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print(f"  step {step}: loss={loss.item()} -- 已经 NaN/Inf，停止")
            break

        loss.backward()
        is_exploding, norm = detector.record_step(step)
        flag = " <-- EXPLODING" if is_exploding else ""
        print(f"  step {step:2d} | loss={loss.item():10.4f} | grad_norm={norm:14.2f}{flag}")

        optimizer.step()

    detector.report()

    print("\n" + "=" * 60)
    print("对照组: 应用建议的裁剪值重跑一遍，验证梯度范数被真正压住、loss不再是天文数字")
    print("=" * 60)
    torch.manual_seed(0)
    model2 = make_deep_unnormalized_net(depth=20, width=64, init_std=1.5)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    clip_value = detector.suggested_clip_value()

    for step in range(15):
        optimizer2.zero_grad()
        out = model2(x)
        loss2 = criterion(out, y)
        loss2.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=clip_value).item()
        if step % 3 == 0:
            print(f"  step {step:2d} | loss={loss2.item():10.4f} | 裁剪前原始norm={raw_norm:14.2f} | 裁剪到={clip_value:.4f}")
        optimizer2.step()
    print(f"  裁剪后最终 loss: {loss2.item():.4f}")


if __name__ == "__main__":
    main()
