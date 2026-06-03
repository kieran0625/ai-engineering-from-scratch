# 混合专家模型（MoE）

> 一个稠密的 70B transformer 为每个 token 激活所有参数。而一个 671B 的 MoE 每个 token 仅激活 37B，却在所有基准测试上击败前者。稀疏性是这个十年最重要的扩展思想。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer），Phase 7 · 07（GPT）
**时间：** ~45 分钟

## 问题所在

稠密 transformer 在推理时的 FLOPs 等于其参数数量（前向传播再乘以 2）。扩展稠密模型时，每个 token 都要支付完整代价。到 2024 年，前沿技术撞上了计算墙：要有意义地提升智能水平，每个 token 需要的 FLOPs 呈指数增长。

混合专家模型打破了这种绑定。将每个 FFN 替换为 `E` 个独立专家 + 一个为每个 token 挑选 `k` 个专家的路由器。总参数 = `E × FFN_size`。每个 token 的激活参数 = `k × FFN_size`。典型的 2026 年配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿几乎全是 MoE：DeepSeek-V3（671B 总参数 / 37B 激活参数）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 名开源模型都是 MoE。

## 核心概念

![MoE 层：路由器为每个 token 选择 k 个专家](../assets/moe.svg)

### FFN 替换

稠密 transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是独立的 FFN（通常是 SwiGLU）。路由器是一个单层线性层。每个 token 自行选择 `k` 个专家，并获得其输出的门控混合。

### 负载均衡问题

如果路由器把 90% 的 token 都送给专家 3，其他专家就会挨饿。人们尝试过三种修复方案：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加与专家使用方差成比例的惩罚项。有效，但增加了一个超参数和第二条梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 跳过该层。损害质量。
3. **无辅助损失均衡**（DeepSeek-V3）。添加一个可学习的逐专家偏置，调整路由器的 top-k 选择。偏置在训练损失之外更新。对主目标无惩罚。2024 年的重大突破。

DeepSeek-V3 的做法：每步训练后，对每个专家，检查其使用量高于还是低于目标。将偏置调整 `±γ`。选择使用 `scores + bias`。用于门控的专家概率是未经修改的原始 `scores`。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为*共享*和*路由*两类。每个 token 都经过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家负责专门化。V3 运行 1 个共享专家 + 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家宽度等于完整 FFN。`E` 较小（8–64），`k` 较小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（FFN 大小的 1/8）。`E` 很大（256+），`k` 更大（8+）。总参数相同，但组合空间急剧扩大。每个 token 可有 `C(256, 8) = 400 trillion` 种"专家"组合。质量提升，延迟不变。

### 成本特征

每层，每个 token：

| 配置 | 每个 token 的激活参数 | 总参数 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎所有基准测试上击败 Llama 3 70B（稠密），同时**每个 token 的激活 FLOPs 更少**。更多参数 = 更多知识。更多激活 FLOPs = 每个 token 更多计算。MoE 将两者解耦。

### 陷阱：内存

所有专家都驻留在 GPU 上，无论哪些被激活。671B 模型需要约 1.3 TB 的 fp16 权重显存。前沿 MoE 部署需要专家并行——将专家分片到多个 GPU，通过网络路由 token。延迟由 all-to-all 通信主导，而非矩阵乘法。

## 动手实现

参见 `code/main.py`。纯标准库实现的紧凑 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（每个一层线性层，用于示意）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过逐专家偏置实现无辅助损失均衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不均衡，而不影响模型的预测方向。

### 步骤 2：让 100 个 token 通过路由器

追踪每个专家的触发频率。没有偏置时，使用分布偏斜。加入偏置更新循环后（过度使用的专家 `-γ`，使用不足的 `+γ`），经过几次迭代，使用分布收敛到均匀。

### 步骤 3：参数数量对比

打印 MoE 配置的"稠密等效"参数。DeepSeek-V3 规格：256 个路由专家 + 1 个共享专家，8 个激活，d_model=7168。总参数数量惊人。激活参数仅为稠密 Llama 3 70B 的七分之一。

## 使用方式

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 拥有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 希望在更低的推理成本下获得前沿质量。
- 拥有 VRAM / 专家并行基础设施。
- 工作负载是 token 密集型（聊天、代码），而非上下文密集型（长文档）。

**何时不选 MoE：**
- 边缘部署——你为任何激活 FLOP 支付完整存储代价。
- 延迟关键的单用户服务——专家路由增加开销。
- 小模型（<7B）——MoE 的质量优势只在计算阈值以上显现（~6B 激活参数）。

## 实战应用

参见 `outputs/skill-moe-configurator.md`。该技能在给定参数预算、训练 token 和部署目标的情况下，为新 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失的偏置更新如何在 50 次迭代内使专家使用分布趋于均匀。
2. **中等。** 将学习式路由器替换为基于哈希的路由器（确定性，无需学习）。比较质量和均衡性。为什么学习式路由器更优？
3. **困难。** 实现类 GRPO 的" rollout 匹配路由"（DeepSeek-V3.2 技巧）：记录推理时哪些专家被激活，在梯度计算时强制使用相同路由。在简化的策略梯度设置上测量效果。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Expert（专家） | "众多 FFN 之一" | 一个独立的前馈网络；参数专用于 FFN 计算的稀疏切片。 |
| Router（路由器） | "门控" | 一个微型线性层，为每个 token 对每个专家打分；top-k 选择。 |
| Top-k routing（Top-k 路由） | "每个 token 激活 k 个专家" | 每个 token 的 FFN 计算恰好经过 k 个专家，按门控加权。 |
| Auxiliary loss（辅助损失） | "负载均衡惩罚" | 额外损失项，惩罚偏斜的专家使用分布。 |
| Auxiliary-loss-free（无辅助损失） | "DeepSeek-V3 的技巧" | 仅通过路由器选择上的逐专家偏置实现均衡；无额外梯度。 |
| Shared expert（共享专家） | "始终激活" | 每个 token 都经过的额外专家；捕获通用知识。 |
| Expert parallelism（专家并行） | "按专家分片" | 将不同专家分布到不同 GPU；通过网络路由 token。 |
| Sparsity（稀疏性） | "激活参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 原始思想。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 无辅助损失 MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于偏置的均衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器采用的细粒度 + 共享专家拆分。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始共享专家论文。
