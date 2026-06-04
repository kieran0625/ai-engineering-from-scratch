# 梯度检查点与激活值重计算

> 反向传播会保留每一个中间激活值。在 70B 参数和 128K 上下文的情况下，每个 rank 需要存储 3 TB 的激活值。检查点技术用 FLOPs 换取内存：通过重计算代替保存。关键问题在于要丢弃哪些片段，而答案绝不是“全部丢弃”。

**类型：** Build
**语言：** Python（含 numpy，可选 torch）
**前置知识：** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**时间：** ~70 分钟

## 问题所在

训练 Transformer 时，会为每一层存储反向传播中需要求导的每个操作的输入：包括注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出以及残差流。对于隐藏维度为 `d`、序列长度为 `L`、批次大小为 `B` 的层来说，每层大约需要存储 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，在 BF16 精度下每层占用 800 MB。一个 64 层的模型将产生 51 GB 的激活值——这还没乘以微批次大小，没加上注意力 softmax 中间结果（每个头 `L^2`），也没考虑张量并行中的部分副本。

双面账单：BF16 权重加上优化器状态可能还能塞进 80GB，但激活值会让你直接超标。梯度检查点（又称激活值重计算）是标准解决方案。丢弃大部分激活值；在反向传播期间重新执行前向传播以恢复它们。代价：额外的 FLOPs。收益：内存占用按检查点片段数与总层数的比例下降。

如果简单粗暴地实现，每次迭代的检查点会带来约 33% 的前向传播 FLOPs 开销。如果做得好——遵循 Korthikanti 等人的“智能选择”进行选择性检查点——你可以用不到 5% 的 FLOPs 开销节省 5 倍内存。结合 FP8 矩阵乘法、FSDP 卸载和专家并行 MoE，这一点至关重要：你既负担不起超额的显存，也浪费不起算力。

## 核心概念

### 反向传播实际需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。为了计算它们，它需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 某些激活值导数的中间结果（ReLU/GELU/softmax 的导数依赖于激活值本身）

前向传播会自动将这些内容存储在 autograd 图中。每个 `tensor.retain_grad()` 以及任何需要其输入的操作都会保留引用。

### 朴素的全量检查点

将网络划分为 `N` 个片段。在前向传播期间，仅保存每个片段的*输入*。当反向传播需要中间结果时，重新运行该片段的前向传播以生成它们，然后再进行求导。

示例：将 32 层的 Transformer 拆分为 32 个各含 1 层的片段。

- 内存：32 个层输入（较小）对比 32 * (每层激活值体积)（巨大）。
- 额外计算：每个片段多一次前向传播，即总共增加约 33% 的前向 FLOPs（因为反向传播的计算量通常是前向的 2 倍，完整步骤从 1 + 2 = 3 变为 1 + 1 + 2 = 4 个单位）。

这是 Chen 等人 2016 年提出的原始方案：每隔 `sqrt(L)` 层设置一个检查点，以平衡内存和计算。对于 L=64，就是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活值的存储成本都相同。注意力 softmax 输出为 `B*L*L*heads`，且随序列长度*二次方*增长。FFN 隐藏层激活值为 `B*L*4d`，呈线性增长。对于长序列，softmax 占主导地位。

选择性检查点保留易于存储的激活值（线性投影、残差），仅重计算昂贵的部分（注意力机制）。你只需支付极少的 FLOPs 进行重计算，即可节省 O(L^2) 的内存。

Megatron-Core 将其实现为“选择性”激活值重计算。它是 2024 年及以后前沿模型训练的标准配置。

### 卸载（Offload）

重计算的替代方案：在前向和反向传播之间将激活值传输到 CPU 内存。这需要 PCIe 带宽；当空闲带宽超过重新生成激活值的成本时，此方法有益。混合策略很常见：对部分层使用检查点，对其他层进行卸载。

FSDP2 将卸载作为一等公民选项提供。当 GPU 受限于内存瓶颈，但 CPU-GPU 传输仍有空间时，卸载效果最佳。

### 重计算成本模型

朴素检查点（每 `k` 层设置一次，共 `L` 层）的每步 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

采用选择性检查点时，你仅重计算注意力内核，而非整个层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活值体积：`A`。对于 `L` 层，总激活值内存：`L * A`。

全量检查点（片段大小为 1）：仅存储 `L * input_volume`（标准 Transformer 约为 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层设置检查点：存储 `L/k * A` 加上活跃片段内 `k-1` 层的激活值。

在 `k = sqrt(L)` 时，内存和重计算成本均随 `sqrt(L)` 缩放——这是均匀成本层的最优权衡。

### 何时不应使用检查点

- 流水线阶段中已经处于飞行状态的内部最深层。它们无论如何都必须完成。
- 如果首尾层主导了阶段的计算量（在 Transformer 中很少见）。
- 已使用 FlashAttention 的注意力内核——Flash 已经快速重计算了 softmax，因此额外的层级别检查点带来的收益微乎其微。

### 实现模式

1. **函数包装器：** 将片段包裹在 `torch.utils.checkpoint.checkpoint(fn, input)` 中。PyTorch 仅存储 `input`，并在反向传播时重计算其余所有内容。

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些片段被包装。

3. **手动显式重计算：** 自行编写反向传播，调用自定义的 `recompute_forward`，利用存储的输入复制前向过程。

这三种方式产生的功能结果相同。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行（TP）：** 重计算时必须收集或重新散列检查点输入；需处理通信开销。
- **流水线并行（PP）：** 典型模式是对每个流水线阶段的前向传播设置检查点，以便逆序微批次可以复用激活值内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史记录必须与原始前向传播一致，否则 FP8 缩放因子会发生漂移。大多数框架会对缩放因子进行快照。

## 动手构建

### 步骤 1：带有片段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：需要所有激活值的朴素反向传播

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 层检查点的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优片段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 实际应用

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 中的标准包装器。包装一个函数；仅存储输入，在反向传播时重计算。
- **Megatron-Core 激活值重计算**：支持 `selective`、`full` 和 `block` 模式。是 2024 年及以后前沿训练的标准配置。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 `offload_policy`，在 FSDP2 中将激活值分片卸载到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**：CPU 卸载优化器状态和激活值，作为检查点的补充。

## 交付成果

本课将生成 `outputs/prompt-activation-recompute-policy.md` —— 一个提示词工具，接收你的模型配置（层数、隐藏维度、序列长度、批次大小）和可用 GPU 内存，并输出逐层重计算策略（无 / 选择性 / 全量 / 卸载）。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（全量激活值）与 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）的对比。参数梯度必须在机器精度范围内完全一致。

2. 遍历片段大小 `k`，范围从 1 到 `L`。绘制 FLOPs 开销和内存曲线。找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块的输入但不存储其中间结果。测量在 seq=8192 的 32 层模型上，相对于全层检查点的 FLOPs 开销。

4. 添加卸载功能。将片段输入保存到模拟的“CPU 缓冲区”（一个独立的列表）。将“PCIe 带宽”度量为单位时间的字节数，并找出卸载与重计算之间的盈亏平衡点。

5. 对真实 PyTorch Transformer 进行基准测试，分别开启和关闭 `torch.utils.checkpoint`。通过 `torch.cuda.max_memory_allocated` 测量内存和单步耗时。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 梯度检查点 | “通过重做前向传播来节省内存” | 仅保存片段输入；在反向传播期间重计算中间结果以获取支持梯度的张量 |
| 激活值重计算 | “与检查点相同” | 高性能计算领域对该技术的称呼 |
| 片段大小 (k) | “每个检查点包含多少层” | 一起丢弃并重新生成的中间结果所对应的层数 |
| 选择性检查点 | “Korthikanti 的技巧” | 仅重计算存储成本高的激活值（注意力 softmax）；保留成本低的那些 |
| 全量检查点 | “朴素版本” | 在每个片段中重计算每一层的中间结果 |
| 块级检查点 | “粗粒度” | 对整个 Transformer 块设置检查点；粒度最大 |
| FLOPs 开销 | “计算税” | 每步额外 FLOPs = (重计算 FLOPs) / (前向 + 反向 FLOPs)；朴素版 33%，选择性版 5% |
| 激活值卸载 | “传输到 CPU” | 在前向到反向传播期间将激活值移至 CPU 内存；重计算的替代方案 |
| sqrt-L 规则 | “经典最优解” | 对于均匀成本层，最优检查点间隔为 sqrt(L) 层 |
| 注意力 softmax 体积 | “O(L^2) 问题” | L^2 * 头数 * 批次大小的浮点数；在长上下文中主导激活值内存 |

## 延伸阅读

- [Chen 等人, 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的原始论文
- [Korthikanti 等人, 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活值重计算及正式的成本分析
- [Pudipeddi 等人, 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重新生成实现恒定内存的替代方案
- [Ren 等人, 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模激活值卸载
- [PyTorch torch.utils.checkpoint 文档](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core 激活值重计算文档](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 选择性、全量和块级模式
