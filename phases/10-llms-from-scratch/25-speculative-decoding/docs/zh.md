# 投机解码与 EAGLE

> 前沿大语言模型每生成一个 token 都需要对数十亿参数进行一次完整的前向传播。这种前向传播存在严重的资源过剩：在大多数情况下，一个小得多的模型就能正确预测接下来的 3-5 个 token，而大模型只需要*验证*这些猜测即可。如果猜对了，你就以一次计算的成本获得了 5 个 token。投机解码（Speculative Decoding，Leviathan 等，2023）实现了这一构想，而 EAGLE-3（2025）将接受率推高至每次验证约 4.5 个 token —— 在保持输出分布一致的前提下实现了 4-5 倍的加速。

**类型：** 构建
**语言：** Python（含 numpy）
**前置知识：** 第 10 阶段课程 12（推理优化）、第 10 阶段课程 04（预训练 Mini-GPT）
**耗时：** 约 75 分钟

## 问题背景

在 H100 上，70B 级别模型的解码吞吐量通常为每秒 40-80 个 token。每个 token 的生成都需要一次完整的前向传播，并从 HBM 读取所有模型权重。在不改变模型输出的前提下，你无法缩小模型体积；受限于显存，你也无法无限增大 batch size。你陷入了僵局——除非能让模型在一次前向传播中输出多个 token。

自回归生成看似天生是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里存在并发优化的空间。如果你有一个低成本预测器指出“接下来的 4 个 token 很可能是 [a, b, c, d]”，你就可以通过大模型的**单次前向传播**来验证这 5 个位置，并采纳最长的匹配前缀。

Leviathan、Kalai、Matias（2023，《Fast Inference from Transformers via Speculative Decoding》）通过一种巧妙的接受/拒绝规则实现了这一点，该规则能够保留目标模型的采样分布。输出分布完全相同，速度提升 2-4 倍。

## 核心概念

### 双模型架构

- **目标模型（Target model）** `M_p`：你真正想要采样的那个庞大、缓慢且高质量的大模型。分布：`p(x)`。
- **草稿模型（Draft model）** `M_q`：一个小型、快速但质量稍低的模型。分布：`q(x)`。参数量小 5-30 倍。

每一步流程如下：

1. 草稿模型自回归地提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型并行地对所有 `K+1` 个位置执行**一次**前向传播，为每个提议的 token 生成 `p(x_k)`。
3. 按照下方修改后的拒绝采样规则，从左到右依次接受或拒绝每个 token。采纳最长的匹配前缀。
4. 如果任何 token 被拒绝，则从修正后的分布中采样替换 token 并停止。否则，从 `p(· | x_1...x_K)` 中额外采样一个 token。

如果草稿与目标完全匹配，你每次目标模型前向传播可获得 K+1 个 token。如果草稿在第 1 个位置就错了，你只能得到 1 个 token。

### 精确性规则

投机解码在分布上**被严格证明等价于直接从 p 采样**。其拒绝规则如下：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差值的正部。当草稿与目标一致时（`p ≈ q`），接受率接近 1。当两者不一致时，残差分布的构造方式确保了整体采样结果仍然严格等于 `p`。

**贪婪解码情况。** 对于 temperature=0 的采样，只需检查 `argmax(p) == x_t`。若成立则接受；若不成立，则输出 `argmax(p)` 并停止。

### 预期加速比

如果草稿模型的 token 级接受率为 `α`，则每次目标模型前向传播的预期产出 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：每次前向传播产出 `(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token。单次目标模型前向传播的成本约为 `cost_q * K + cost_p`（包含 K 次草稿步骤和 1 次目标验证）。若 `cost_p >> cost_q * K`，则吞吐量的加速比为 `3.36× / 1 = 3.36×`。

唯一的关键参数是 `α`，它完全取决于草稿模型与目标模型的对齐程度。一个好的草稿模型至关重要。

### 训练草稿模型：蒸馏

随机的小模型无法作为优秀的草稿模型。标准做法是从目标模型进行蒸馏：

1. 选择一个小型架构（针对 70B 目标模型选约 1B，针对 7B 目标模型选约 500M）。
2. 让目标模型处理大规模文本语料库；保存其下一个 token 的分布。
3. 使用 KL 散度训练草稿模型以拟合目标模型的分布（而非直接拟合真实 token）。

结果：`α` 通常在代码任务上达到 0.6-0.8，在自然语言对话任务上达到 0.7-0.85。在生产环境中可实现 2-3 倍的加速。

### EAGLE：树状草稿 + 特征复用

Li、Wei、Zhang、Zhang（2024，《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》）指出了标准投机解码中的两个低效之处：

1. 草稿模型需要执行 K 步串行计算，每一步都是完整的堆栈操作。但草稿模型本可以复用最近一次验证中目标模型的特征（隐藏状态）——目标模型已经计算出了丰富的表征，而草稿模型却在从头重新推导。
2. 草稿模型输出的是线性链。如果草稿模型能输出候选项的*树*结构（每个节点包含多个猜测），目标模型的单次前向传播就可以通过树注意力掩码并行验证多条候选路径，并选择最长的那条已接受分支。

EAGLE-1 的改进：
- 草稿输入 = 目标模型在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 个 Transformer 解码层（而非独立的小型模型）。
- 输出 = 深度为 4-6 的树，每层包含 K = 4-8 个候选项。

EAGLE-2（2024）引入了动态树拓扑结构：在草稿模型不确定时树结构变宽，在确定时保持狭窄。在不增加验证成本的前提下提升了 `α_effective`。

EAGLE-3（Li 等，2025，《EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test》）移除了对固定顶层特征的依赖，并使用新的“测试时仿真”损失函数训练草稿模型——草稿模型在匹配目标模型测试时分布的输出上进行训练，而非传统的教师强制训练分布。接受率从 0.75（EAGLE-2）提升至 0.82（EAGLE-3），平均每次验证的 token 数从 3.0 提升至 4.5。

### 树注意力验证

当草稿模型输出树结构时，目标模型会使用**树注意力掩码**在一次前向传播中完成验证——这是一种因果掩码，编码了树拓扑结构而非单纯的线性序列。每个 token 仅关注其在树中的祖先节点。验证过程依然是一次前向传播、一次矩阵乘法；拓扑掩码仅需消耗少量额外的 KV 缓存条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争性的首 token 候选项，而 `c, d, e, f` 是第二 token 候选项，那么所有六个位置将在一次前向传播中得到验证。最终输出是任意已接受路径上的最长前缀。

### 适用场景与局限性

**优势场景：**
- 文本可预测的对话/补全任务（如代码、常见英语、结构化输出）。此时 `α` 较高。
- 解码阶段存在未充分利用的 GPU 算力的设置（通常处于显存瓶颈阶段）。树状草稿能有效利用可用的 FLOPs。

**劣势/无效场景：**
- 高度随机的输出（如高温设置下的创意写作）。此时 `α` 会下降至接近 `1/|vocab|`。
- 极高并发批处理服务——批处理本身已占满 FLOPs，几乎没有空间用于树验证。
- 目标模型本身非常小，导致草稿模型并未显著更小。

生产环境通常报告：对话任务实现 2-3 倍的实际运行时间加速，代码生成任务实现 3-5 倍加速，创意写作任务加速效果接近于零。

## 动手实现

`code/main.py`：

- 一个参考实现的 `speculative_decode(target, draft, prompt, K, temperature)`，用于实现精确的拒绝规则，并验证其是否保留了目标模型的分布（与纯目标采样相比，经验 KL 散度 < 0.01）。
- 一个类 EAGLE 的树状草稿生成器，用于构建具有 top-p 分支的深度 K 树。
- 一个树注意力掩码构建器，用于为验证器生成正确的因果模式。
- 一个接受率测试框架，在微型语言模型上同时运行两者（从一个 GPT-2-medium 目标模型中蒸馏出一个 GPT-2-small 草稿模型）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 实际应用

- **vLLM** 和 **SGLang** 原生支持一流的投机解码。相关参数：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 参数支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树结构。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（适用于 Qwen3-32B 的草稿模型）、`meta-llama/Llama-3.2-1B-Instruct-spec`（适用于 70B 的草稿模型）。
- **Medusa heads**（Cai 等，2024，《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》）：无需独立的草稿模型，直接在目标模型上添加 K 个并行预测头。部署更简单，但接受率略低于 EAGLE。

## 交付成果

本课程将产出 `outputs/skill-speculative-tuning.md` —— 一项能够对目标模型的工作负载进行分析画像，并据此选择：草稿模型、K（草稿长度）、树宽度、temperature 以及何时回退到普通解码的技能。

## 练习

1. 实现精确的拒绝规则并进行经验验证。分别通过 `speculative_decode` 和纯目标采样运行 1 万次样本；计算两种输出分布之间的全变差（TV）距离。结果应小于 0.01。
2. 推导加速比公式。给定固定的 `α` 和 `K`，绘制每次目标前向传播的预期 token 数曲线。找出 α ∈ {0.5, 0.7, 0.9} 时的最优 K 值。
3. 训练微型草稿模型。选取 124M 参数的 GPT-2 作为目标模型，在 1 亿 token 上使用 KL 损失蒸馏出 30M 参数的 GPT-2 草稿模型。在未参与训练的文本上测量 `α`。预期结果：0.6-0.7。
4. 实现类 EAGLE 的树状草稿生成。不再输出线性链，而是让草稿模型在每个深度输出 top-3 分支。构建树注意力掩码。验证目标模型是否能接受最长的正确分支。
5. 测量失败模式。在 temperature=1.5（高随机性）下运行投机解码。展示 α 如何崩溃，并证明由于草稿开销，算法此时比普通解码更慢。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Target model（目标模型） | “大模型” | 你真正想要采样的缓慢、高质量模型（p 分布） |
| Draft model（草稿模型） | “投机者” | 小型、快速的预测模型（q 分布）；体积小 5-30 倍 |
| K / draft length（K / 草稿长度） | “前瞻步数” | 每次验证过程中推测的 token 数量 |
| α / acceptance rate（α / 接受率） | “命中率” | 草稿模型提议的 token 被接受的单 token 概率 |
| Exact rejection rule（精确拒绝规则） | “接受测试” | 比较 r < p/q 以保留目标模型分布的规则 |
| Residual distribution（残差分布） | “修正后的 p-q” | (p - q)+ / ||(p - q)+||_1，拒绝时需从中采样的分布 |
| Tree drafting（树状草稿） | “分支推测” | 草稿模型输出候选项树，通过树结构注意力掩码在一次传递中完成验证 |
| Tree attention mask（树注意力掩码） | “拓扑掩码” | 编码树拓扑结构的因果掩码，使每个节点仅关注其祖先 |
| Medusa heads（Medusa 头） | “并行头” | 直接在目标模型上添加 K 个额外预测头；无需独立草稿模型 |
| EAGLE feature reuse（EAGLE 特征复用） | “隐藏状态草稿” | 草稿输入为目标模型的最后一层隐藏状态而非原始 token，从而缩小草稿规模 |
| Test-time simulation loss（测试时仿真损失） | “EAGLE-3 训练” | 在匹配目标模型测试时分布的输出上训练草稿模型，而非使用教师强制 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — 《Fast Inference from Transformers via Speculative Decoding》](https://arxiv.org/abs/2211.17192) — 精确拒绝规则与理论加速比分析
- [Chen, Borgeaud, Irving 等, 2023 — 《Accelerating Large Language Model Decoding with Speculative Sampling》](https://arxiv.org/abs/2302.01318) — DeepMind 同期发表的投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — 《Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads》](https://arxiv.org/abs/2401.10774) — 替代草稿模型的并行头方案
- [Li, Wei, Zhang, Zhang, 2024 — 《EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty》](https://arxiv.org/abs/2401.15077) — 特征复用与树状草稿
- [Li 等, 2024 — 《EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees》](https://arxiv.org/abs/2406.16858) — 动态树拓扑结构
- [Li 等, 2025 — 《EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test》](https://arxiv.org/abs/2503.01840) — 训练时与测试时分布对齐
- [Fu, Haotian, Peng 等, 2024 — 《Break the Sequential Dependency of LLM Inference Using Lookahead Decoding》](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，一种无需投机者的替代方案
