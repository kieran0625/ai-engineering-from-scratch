# 原生稀疏注意力 (DeepSeek NSA)

> 在 64k token 长度下，注意力机制会消耗 70-80% 的解码延迟。每个开源模型实验室都有修复它的方案。DeepSeek 的 NSA（ACL 2025 最佳论文）是真正落地的那一个：三个并行的注意力分支——压缩的粗粒度 token、选择性保留的细粒度 token，以及用于局部上下文的滑动窗口——通过一个学习到的门控机制进行组合。它与硬件对齐（对 kernel 友好），原生可训练（适用于预训练阶段，而非仅在推理时外挂），在 64k 解码时比 FlashAttention 更快，同时达到或超越全注意力的质量。本课程将端到端地实现这三个分支，并解释为何这种稀疏性是端到端可微的。

**类型：** 构建
**语言：** Python (stdlib)
**前置知识：** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**耗时：** 约 60 分钟

## 学习目标

- 阐述 NSA 的三个注意力分支及其各自捕获的信息。
- 解释为何 NSA 是“原生可训练”的，而此前的稀疏注意力方法仅适用于推理阶段。
- 计算在 64k 上下文长度下，NSA 相较于全注意力的注意力计算节省量，该值作为压缩块大小和选择 top-k 的函数。
- 使用 Python 标准库在短合成序列上实现三分支组合，并验证门控权重的行为是否符合预期。

## 问题背景

序列长度为 N 时的全注意力计算耗时为 `O(N^2)`，每层需要 `O(N)` 的 KV cache。在 64k token 长度下，计算量和内存带宽需求极其庞大。根据 NSA 论文中的实测理论估算：在 64k 长度下，注意力机制占总解码延迟的 70-80%。所有下游指标——TTFT、tokens/sec、每百万 token 成本——均被注意力成本所主导。

稀疏注意力是显而易见的解决方案。以往的尝试主要分为两类。固定模式稀疏（如滑动窗口、步长采样、块局部）会丢弃信息，在长程召回任务中表现不佳。推理时稀疏（如 KV cache pruning、H2O、StreamingLLM）应用于基于稠密注意力预训练的模型，且只能恢复部分潜在加速比，因为模型从未被要求通过稀疏模式来路由信息。

Native Sparse Attention（Yuan 等人，DeepSeek + PKU + UW，ACL 2025 最佳论文，arXiv:2502.11089）两者兼得：一种模型在预训练期间学习的稀疏模式，并以与 kernel 对齐的算法形式实现，从而在推理时真正交付计算节省。两年后，NSA 或其直接衍生架构将成为所有前沿长上下文模型的默认注意力机制。

## 核心概念

### 三个并行分支

对于每个 query，NSA 运行三次注意力计算，分别针对 KV cache 的三种不同视图：

1. **压缩分支。** Token 被分组为大小为 `l` 的块（通常为 32 或 64）。每个块通过一个小型学习到的 MLP 压缩为单个摘要 token。query 对这些压缩后的 token 执行注意力计算，获得整个序列的粗粒度视图。

2. **选择分支。** 利用压缩分支的注意力分数，识别出与当前 query 最相关的 top-k 个块。读取这些块中的细粒度（未压缩）token，query 对所有这些 token 执行注意力计算。可将压缩分支的注意力视为选择的信号路由。

3. **滑动窗口分支。** query 关注最近的 `W` 个 token（通常为 512）以获取局部上下文。该分支捕获其他两个分支可能遗漏的高结构依赖的短程模式（句法、局部指代）。

三个分支的输出通过一个按位置学习到的门控机制进行组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是来自 query 的小型 MLP 的门控权重。它们无需总和为 1 —— 可以独立地为各分支分配权重。

### 为何这是“原生可训练”的

选择步骤（top-k 块）是离散的。离散操作会阻断梯度流。先前的稀疏注意力研究要么跳过选择步骤的反向传播（限制训练效果），要么使用连续松弛方法，但无法在推理时产生真正的稀疏性。

NSA 巧妙地规避了这一问题：压缩分支的注意力本身就是对整个序列的可微粗粒度注意力。top-k 操作只是复用了压缩分支的最高注意力分数来选择加载哪些细粒度块。梯度可以通过压缩分支的分数流动（这些分数同时影响压缩输出和选择逻辑），且所选块对最终输出的贡献也是可微的。不可微的 `top_k` 操作在前向计算图中是一个无实际计算的操作（no-op）——它仅控制从内存中加载哪些块。

这就是为什么 NSA 可以端到端地用于预训练。模型学会联合通过这三个分支路由信息，生成一种稀疏模式，在推理时真正兑现承诺的加速比。

### 与硬件对齐的 kernel

NSA 的 kernel 专为现代 GPU 内存层次结构设计。kernel 按 GQA 组加载 query（外层循环），为每组获取对应的稀疏 KV 块（内层循环），并在 SRAM 上执行注意力计算。由于每个 query 组看到相同的选中块（选择是按 query 组而非按 query head 进行的），KV 加载开销可在组内分摊。算术强度得以保持高位。

论文报告称，Triton kernel 在 64k 解码时比 FlashAttention 快 9 倍，且加速比随序列长度增加而提升。前向和反向 kernel 均已提供。

### 计算预算

设 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择数量，`w` 为滑动窗口大小，`b` 为选中块大小（通常等于 `l`）。

- 压缩分支：每个 query 对应 `O(N/l)` 个 key，总计 `O(N * N / l)`。
- 选择分支：每个 query 对应 `O(k * b)` 个 key，总计 `O(N * k * b)`。
- 滑动分支：每个 query 对应 `O(w)` 个 key，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512` 时：单 query 成本为 `1000 + 1024 + 512 = 2536 keys`。全注意力为 `64000 keys`。计算量减少 25 倍。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512` 时：单 query 成本为 `2000 + 1024 + 512 = 3536 keys`。全注意力为 `128000 keys`。减少 36 倍。收益随序列长度增长，这正是核心所在。

### 横向对比

| 方法 | 可微性 | 真实推理加速比 | 长程召回能力 |
|--------|---------------|----------------------|-------------------|
| 仅滑动窗口 | 是 | 是 | 失败 |
| 步长/块稀疏 | 是 | 是 | 部分 |
| KV 剪枝 (H2O, StreamingLLM) | 不适用（推理时） | 是 | 部分 |
| MoBA (Moonshot) | 部分 | 是 | 良好 |
| NSA | 是（原生） | 是（64k 时 9 倍） | 媲美全注意力 |

MoBA（Moonshot，arXiv:2502.13189）同期发表，采用了类似的“三优于一”思路，将 MoE 原则应用于注意力块。NSA 和 MoBA 是 2026 年长上下文预训练中必须掌握的两类架构。

## 动手实现

`code/main.py` 在短合成序列上实现了三个分支，并展示：

- 压缩 MLP（出于教学清晰度的考虑，此处使用简单的均值池化基线；实际 NSA 使用学习到的 MLP）。
- 由压缩分支分数驱动的选择分支。
- 对最后 `w` 个 token 的滑动窗口注意力。
- 门控组合机制。
- 与全注意力对比的计算量打印输出。

### 步骤 1：将 token 压缩为块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 步骤 2：压缩分支注意力

对 query 与压缩后的 keys 执行 softmax 注意力计算。压缩分支的分数同时作为 top-k 选择的信号。

### 步骤 3：top-k 块选择

选取得分最高的 `k` 个压缩块的索引。从这些块中加载原始未压缩 token，并对其执行注意力计算。

### 步骤 4：滑动窗口注意力

取最后 `w` 个 token，对其执行标准注意力计算。

### 步骤 5：门控 + 组合

对 query 应用一个小型 MLP 生成三个门控权重。最终输出为三个分支输出的加权和。

### 步骤 6：计算量统计

打印每个分支及总计的每 query 注意力 key 数量。与 `N`（全注意力）进行对比。在包含 1024 个 token 的合成数据且 `l = 32, k = 4, w = 128` 的情况下，NSA 每 query 仅需处理 `32 + 128 + 128 = 288` 个 key，而全注意力需 1024 个——减少了 3.5 倍。

## 实际应用

NSA 已部署于 DeepSeek 自身的长上下文预训练流水线中。截至 2026 年 4 月，其在公开推理框架中的集成状态如下：

- **DeepSeek 内部**：原生支持，已发布权重使用 NSA 或其继任者 DSA（DeepSeek Sparse Attention）。
- **vLLM**：正在为 DeepSeek-V3.x 权重开发实验性 NSA 支持。
- **SGLang**：已发布 NSA 基准测试；生产环境路径将跟随 vLLM。
- **llama.cpp / CPU**：不支持；在 CPU 吞吐量下，kernel 分解的开销得不偿失。

何时使用 NSA：

- 面向 64k 以上上下文且拥有充足计算预算的预训练或持续训练任务。
- 推理 DeepSeek 自身的长上下文 checkpoint。其权重为 NSA 原生设计。

何时不使用：

- 服务现有的基于稠密注意力预训练的模型。若不进行持续训练，无法后期适配 NSA。
- 上下文长度低于 16k。三分支的开销将抵消节省的收益。
- Batch-1 交互式对话。虽然延迟敏感的解码能受益，但这仅限于长上下文场景。

## 交付成果

本课程将产出 `outputs/skill-nsa-integrator.md`。给定长上下文预训练任务的配置，它将生成一份 NSA 集成计划：包括压缩块大小、top-k、滑动窗口、门控 MLP 宽度、kernel 选择，以及证明该架构变更合理的具体长上下文评估指标。

## 练习

1. 在 1024-token 合成数据上运行 `code/main.py`。遍历 `(l, k, w)` 的三种预设并打印计算量。找出在大海捞针测试中相对于全注意力保持 95% 召回率的同时，每 query key 数量最低的预设。

2. 用微型学习 MLP（2 层，隐藏层维度 32）替换均值池化压缩器。在一个信号为块平均值的合成任务上对其进行训练。在预留数据上测量其与均值池化基线的困惑度（perplexity）差距。

3. 实现门控 MLP。它以 query 为输入并输出三个标量。验证门控行为是否合理：对随机 query 呈现近似均匀的权重，当 query 命中远端块时，选择分支获得较高权重。

4. 计算启用 NSA 的 70B 模型在 128k 上下文下的 KV cache 显存预算。KV heads 为 8，head dim 为 128，精度为 BF16。将其与全注意力及 MLA 进行对比（Phase 10 · 14 展示了 MLA 的数据）。找出 NSA 细粒度分支的 KV cache 占用等于全注意力时的序列长度。

5. 阅读 NSA 论文第 4 节（arXiv:2502.11089），用三句话解释为何复用压缩分支的注意力分数进行 top-k 选择，而不是单独计算路由分数。请将答案与梯度流联系起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Compressed branch | “粗粒度视图” | 对块均值 key 执行注意力计算，以 O(N/l) 的 key 数量提供全局上下文 |
| Selected branch | “Top-k 块” | 对得分最高的 `k` 个压缩块执行细粒度注意力 |
| Sliding window | “局部上下文” | 对最后 `W` 个 token 执行注意力以捕捉短程模式 |
| Native trainability | “带稀疏性预训练” | 稀疏模式在预训练期间学习得到，而非推理时外挂 |
| Compression block size l | “粗粒度分组大小” | 合并为一个摘要的 token 数量；通常为 32-64 |
| Top-k | “保留的块数” | 读取未压缩 token 的压缩块数量；通常为 16 |
| Sliding window W | “局部注意力半径” | 通常为 512；过短损害局部连贯性，过长浪费计算 |
| Branch gate | “如何混合三者” | 按位置输出的 MLP 结果，用于加权三个分支的贡献 |
| Hardware alignment | “对 Kernel 友好的稀疏” | 选择的稀疏模式能使实际 GPU kernel 达到理论加速比 |
| DSA | “NSA 的继任者” | DeepSeek Sparse Attention，DeepSeek 谱系中紧随 NSA 的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 论文原文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 目标应用的架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，基于 MoE 风格的块级注意力
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — 滑动窗口的起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA 改进的推理时稀疏基线
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA kernel 在 64k 长度下超越的全注意力基线
