# Jamba — 混合 SSM-Transformer

> 状态空间模型（SSM）和 Transformer 的需求截然不同。Transformer 通过注意力机制以二次方代价换取质量。SSM 则通过递归实现线性时间推理和恒定内存，但在质量上稍逊一筹。AI21 的 Jamba（2024年3月）和 Jamba 1.5（2024年8月）将两者融合于同一模型中：每 7 个 Mamba 层搭配 1 个 Transformer 层，每隔一个块使用 MoE，并支持在单张 80GB GPU 上容纳 256k 上下文窗口。Mamba-3（ICLR 2026）通过复数状态空间和 MIMO 投影进一步增强了 SSM 侧的性能。本课程将端到端地阅读这两种架构，并解释为何这种混合配方在纯 SSM 和纯 Transformer 长上下文尝试均未成功的三年扩展期后依然屹立不倒。

**类型：** 学习
**语言：** Python (stdlib, layer-mix calculator)
**前置要求：** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**耗时：** 约 60 分钟

## 学习目标

- 阐述 Jamba 块中的三个基础组件——Transformer 层、Mamba 层、MoE，以及 1:7:even 的交错配方。
- 从宏观层面描述 SSM 递归的形式，并解释其为何能实现恒定内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV cache 占用，并与纯 Transformer 模型所需的占用进行对比。
- 列举 Mamba-3 的三项创新（指数-梯形离散化、复数状态更新、MIMO），并说明每项创新针对的问题。

## 问题所在

注意力机制的计算复杂度随序列长度呈二次方增长。状态空间模型则是线性的。这种差异会不断累积：在 256k token 时，Transformer 的注意力图每个头包含 650 亿个条目；而无论序列多长，SSM 的递归状态大小都是固定的。

纯 SSM 模型（Mamba、Mamba-2）在小规模下能达到与 Transformer 相当的 perplexity，但在状态追踪任务上表现落后，且在部分类别的 in-context retrieval 任务中失败。直观理解是：SSM 将历史压缩为固定大小的状态，当历史过长时，信息会发生泄漏。Attention 能精确记住所有内容，但需支付二次方代价。

显而易见的解决方案：两者兼用。在需要精确 recall 的地方使用 Transformer 层，在其他地方使用 SSM 层。调整比例。Jamba 是首款以生产级标准大规模部署此混合配方的模型（总计 52B，激活 12B，256k 上下文，单张 80GB GPU）。Jamba 1.5 将该系列扩展至总计 398B / 激活 94B。Mamba-3（ICLR 2026）是目前最佳的纯 SSM 基线，未来的混合模型可围绕它重新构建。

本课程将阅读上述三篇论文，并建立“如何选择合适比例”的心智模型。

## 核心概念

### 一页纸看懂 SSM

状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

在每一步中，状态通过线性动力学 `A` 演化，接收输入 `B x_t`，并输出结果 `C h_t`。`A, B, C` 可被学习。注意关键特性：计算 `y_t` 仅需 `h_{t-1}` 和 `x_t`，无需任何早期的 `x`。内存占用恒定。每个 token 的推理复杂度为 O(1)。

建模质量的关键在于 `A` 的结构。S4（Gu 2021）使用了高度结构化的矩阵，可在训练期间作为长卷积高效求值。Mamba（Gu, Dao 2023）将固定的 `A, B, C` 替换为数据依赖的变体（即“selective”部分）。Mamba-2（2024）进一步简化了该结构。Mamba-3（2026）则在特定位置重新引入了复杂性。

关键特性：对于 decoder LLM，SSM 层可作为 attention 层的无缝替代品，以每层固定大小的状态取代不断增长的 KV cache。

### Jamba 块

Jamba 块根据两个数值交错排列层：

- `l`：attention-to-Mamba ratio。Jamba 使用 `l = 8`，即每 7 个 Mamba 层搭配 1 个 Transformer 层（7 个 Mamba + 1 个 Attention = 每组 8 层）。
- `e`：MoE frequency。Jamba 使用 `e = 2`，即每隔一层应用 MoE。

块内的层序列如下：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba 块包含 8 层。在深度为 4 个块（共 32 层）的情况下，你将得到 28 个 Mamba 层和 4 个 Attention 层。其中 16 层使用 MoE。

### 为何采用 1:7 比例

AI21 进行了 ablations：何种 attention-to-Mamba 比例能在 long-context evals 中同时实现最佳的 perplexity-per-parameter 与 in-context recall？

- Attention 过多（1:1）：quality 提升，但 memory 和 speed 下降。
- Attention 过少（1:15）：memory 表现极佳，但 in-context retrieval 失败。
- 最佳平衡点：1:7 或 1:8。

直观理解：Transformer 层负责 exact recall 和 state tracking。Mamba 层负责处理大部分低成本计算。

### Positional encoding

Mamba 层本身具备 position-aware 能力（通过 recurrence 实现）。原始基于 Mamba 的 hybrid 模型中的 attention 层并未使用 RoPE——SSM 层已提供 position info。Jamba 1.5 为增强 long-context generalization 能力，在 attention 层中加入了 RoPE，这是基于实证 long-context evaluation 的事后优化。

### 内存预算

对于 Jamba-1 shape（32 层：28 个 Mamba + 4 个 Attention，hidden 4096，32 个 attention heads）：

- KV cache（仅 attention 层）：在 256k 上下文、BF16 精度下为 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。仅由 4 个 attention 层贡献。
- SSM state：每个 token prefix 占用 `28 * hidden * state_size`，但这是每层固定大小的 state，不随 sequence length 扩展。典型 Mamba state 为每个 feature 16，hidden 4096：总计 `28 * 4096 * 16 * 2 = 3.7 MB`。

对比 32 层纯 Transformer（相同 hidden，32 头 full MHA）：在 256k 上下文、BF16 精度下为 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV cache 减少了 8 倍。即使与大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`）相比，Jamba 的 1:7 hybrid 仅占 16 GB，仍小一半。

这就是 AI21 所谓“在单张 80GB GPU 上运行 256k 上下文”的含义。full-MHA 纯 Transformer 的 KV cache 无法容纳；即使是 GQA 基线也会耗尽空间，导致 weights 和 activations 无处安放；而 Jamba 可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 侧引入了三项创新：

1. **Exponential-trapezoidal discretization。** 将 Mamba-2 中的 Euler-method discretization 替换为表达能力更强的 recurrence。Convolution-like operation 应用于核心 recurrence 内的 state-input 之间，而非作用于 `x_t` 的外部卷积。
2. **Complex-valued state update。** 之前的 Mamba 版本将 state matrix 从 complex（S4）简化为 real diagonal（Mamba），再简化为 scaled identity（Mamba-2）。Mamba-3 重新引入 complex values——等效于对 state 施加 data-dependent rotary embedding。这恢复了先前 real-valued simplifications 所丧失的 state-tracking 能力。
3. **Multi-input multi-output (MIMO) projections。** 摒弃 per-feature scalar projections，改用 matrix-valued projections。在不增加 decode latency 的前提下，提升了 modeling power 和 inference-time hardware utilization。

在 1.5B parameters 下，Mamba-3 的平均 downstream accuracy 较 Gated DeltaNet 提升 0.6 个百分点；MIMO variant 额外提升 1.2 个百分点，累计提升 1.8 个百分点。在相同 state size 下，Mamba-3 仅用一半的 state 即可达到 Mamba-2 的性能。

Mamba-3 尚未在生产级 hybrid 模型中大规模部署——但它无疑是下一代 Jamba-class 模型 SSM 侧的明显候选者。

### 何时选择混合架构

Hybrid 架构在以下场景占优：

- Context 足够长，使得纯 Transformer KV cache 成为瓶颈（64k+）。
- 任务混合了 short-range structure（适合 SSM）与 long-range recall（需要 Transformer）。
- 希望在 single-GPU memory budgets 下部署，且仅 Transformer 的 KV cache 就无法容纳的场景。

Hybrid 架构在以下场景劣势：

- Context 较短（低于 16k）。SSM 的 overhead 会被浪费；纯 Transformer 即可胜任。
- 任务需要 everywhere-to-everywhere attention（deep reasoning、multi-document cross-reference）。hybrid 中 sparse 的 attention 层会成为短板。
- 正在向 trillion-parameter frontier models 扩展。Pure-Transformer + MLA + MoE（DeepSeek-V3 style）目前在 capability race 中领先。

### 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 年格局：pure-Transformer MoE 主导 frontier 领域，但 hybrid 牢牢占据 256k-plus context 细分市场。Mamba-3 的 state-tracking wins 可能会推动下一代 hybrid 降低比例（更多 SSM，更少 attention）。

## 实践应用

`code/main.py` 是一款面向 hybrid architectures 的 memory calculator。给定 SSM-Transformer ratio 及 hidden-size / layer-count config，它将计算：

- Target context 下的 KV cache 占用。
- SSM state memory。
- 一系列 model shapes 在 context N 时的总 memory 占用。

该 calculator 支持：

- Pure-Transformer baseline（KV cache 随 N 增长）。
- Jamba-style 1:7 hybrid。
- Pure-SSM（完全不使用 KV cache）。

数据直接来源于 Jamba-1 和 Jamba-1.5 papers 中公布的 shapes，hypothetical variants 的数据则为外推得出。

实际部署的 integration considerations：

- 大多数 production inference servers（vLLM、SGLang）均支持 Jamba 和 Mamba。请核对 specific version。
- 在 256k context 下，Jamba 的 memory advantage 体现在 concurrent-request throughput 上。在相同的 VRAM 容量下，你能运行的 Jamba sequences 多于 Transformer sequences。
- Mamba-3 作为 standalone model 尚未投入 production environment——目前仅提供 1.5B 参数的 research preview。

## 交付成果

本课程将产出 `outputs/skill-hybrid-picker.md`。给定 workload specification（context length profile、task mix、memory budget），它将在 pure Transformer、Jamba-style hybrid 和 pure SSM 之间给出推荐，并明确阐述 memory 与 quality tradeoffs 的逻辑。

## 练习

1. 运行 `code/main.py`，计算 32 层 pure Transformer（hidden 4096，32 heads）与同 shape Jamba-1 hybrid 在 256k context 下的 KV cache。验证 AI21 paper 中声称的约 8x memory reduction 效果。
2. 修改 calculator 以模拟 1:3 hybrid（4 Mamba : 1 Attention）和 1:15 hybrid（14 Mamba : 1 Attention）。绘制 KV cache vs ratio 曲线。在何种 ratio 下 KV cache 等于 SSM state memory？
3. 阅读 Jamba paper 第 3 节（arXiv:2403.19887）。解释为何尽管 Mamba-2 更快，AI21 仍选用 Mamba-1。Hint：hybrid ablation section 对此有记录。
4. 计算 Jamba 1.5 Large（398B total，94B active）中 moe-every-other-layer 的 parameter overhead。将其 active ratio 与 DeepSeek-V3（37B/671B）对比，并解释为何 Jamba 的 architecture 能推高 active ratio。
5. 阅读 Mamba-3 paper 第 3 节（arXiv:2603.15569）。用三句话解释为何 complex-valued state update 等效于 data-dependent rotary embedding。结合 Phase 7 · Lesson 04 的 RoPE derivation 进行说明。

## 核心术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State space model (SSM) | "Recurrence with a fixed state" | A layer with a learned recurrence `h_t = A h_{t-1} + B x_t`; constant memory per token |
| Selective SSM | "Mamba's trick" | Data-dependent A, B, C parameters that give the model gating-like selectivity at linear time |
| Attention-to-Mamba ratio | "How many attention layers" | In Jamba, `l = 8` means 1 attention layer per 7 Mamba layers |
| Jamba block | "The 8-layer group" | One attention + seven Mamba + MoE on alternate positions |
| SSM state | "The hidden buffer" | Fixed-size per-layer state that replaces the KV cache for Mamba layers |
| 256k context | "Jamba's flagship number" | The sequence length Jamba-1 fits on a single 80GB GPU; pure Transformer cannot at that size |
| Mamba-3 | "2026 pure SSM" | Current-best pure-SSM architecture with complex state + MIMO; the baseline hybrids rebuild around |
| MIMO | "Multi-input multi-output" | Mamba-3 innovation using matrix-valued projections instead of scalar per-feature |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | More expressive recurrence that subsumes Mamba-2's Euler-method discretization |
| Hybrid architecture | "Mix attention and SSM" | Any model that interleaves Transformer and SSM layers; Jamba is the production archetype |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — the original Jamba paper, ratio ablations, 256k context claim
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — the scaled-up family, 398B/94B and 12B/52B public releases
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — the selective SSM paper Jamba builds on
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — the simplified structured-state-space successor
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — complex-valued state, MIMO, the 2026 pure-SSM frontier
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — the S4 paper, the SSM genealogy's starting point for LLMs
