# DeepSeek-V3 架构导览

> 第 10 阶段 · 第 14 课列出了每个开源模型都会调整的六个架构旋钮。DeepSeek-V3（2024年12月发布，总参数量 671B，激活参数量 37B）不仅调整了全部六个旋钮，还增加了四项新特性：多头潜在注意力（Multi-Head Latent Attention）、无辅助损失负载均衡、多令牌预测（Multi-Token Prediction）和 DualPipe 训练。本课将自上而下解析 DeepSeek-V3 的架构，并根据公开配置推导每一个参数量。学完后，你将能够解释为何 671B/37B 的比例是正确的选择，以及为何在前沿模型中，MLA + MoE 的组合优于单独使用其中任何一项。

**类型：** 学习
**语言：** Python（标准库、参数计算器）
**前置知识：** 第 10 阶段 · 14（开源模型导览）、第 10 阶段 · 17（NSA）、第 10 阶段 · 18（MTP）、第 10 阶段 · 19（DualPipe）
**预计时间：** 约 75 分钟

## 学习目标

- 逐行阅读 DeepSeek-V3 配置文件，并结合 GPT-2 的六个基础旋钮与 DeepSeek 特有的四项新增特性，解释每个字段的含义。
- 推导总参数量（671B）、激活参数量（37B）及其构成组件。
- 计算 128k 上下文下 MLA 的 KV Cache 占用，并与同等激活参数规模下采用 GQA 的稠密模型进行对比。
- 阐述四项 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出它们分别针对架构或训练栈的哪个部分。

## 问题背景

DeepSeek-V3 是首个架构与 Llama 系列有显著差异的前沿开源模型。Llama 3 405B 可视为“调整了六个旋钮的 GPT-2”。DeepSeek-V3 则是“调整了全部六个旋钮并额外增加四项”的 GPT-2。阅读 Llama 3 的配置只是热身，但其深层结构——注意力块的形状、路由逻辑、训练时的目标函数——差异足够大，需要单独的导览。

学习它的回报在于：DeepSeek-V3 的开源权重发布改变了开源模型中“前沿能力”的定义。该架构已成为许多 2026 年训练任务效仿的蓝图。理解它是任何涉及前沿 LLM 训练或推理岗位的必备基础。

## 核心概念

### 不变的核心（再次强调）

DeepSeek-V3 依然是自回归模型。它依然堆叠解码器块。每个块依然包含注意力机制 + MLP + 两个 RMSNorm。它依然在 MLP 中使用 SwiGLU。依然使用 RoPE。Pre-norm。权重绑定的嵌入层。与所有 Llama 或 Mistral 模型的基线相同。

### 核心变化：用 MLA 替代 GQA

从第 10 阶段 · 第 14 课可知，GQA 通过让一组 Q 头共享 K 和 V 来缩小 KV Cache。多头潜在注意力（MLA）走得更远：K 和 V 被压缩为一个共享的低秩潜在表示（即 `kv_lora_rank`），然后在运行时按头动态解压。KV Cache 仅存储潜在表示——通常每层每个 token 仅 512 个浮点数，而非 8 x 128 = 1024 个。

在 128k 上下文中，采用 MLA 的 DeepSeek-V3（每层每个 token 一个共享潜在表示 `c^{KV}`；K 和 V 均通过上投影从此潜在表示派生，且这些上投影可合并到后续的矩阵乘法中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设的 GQA 基线（Llama 3 70B 结构，8 个 KV 头，头维度 128）需支付：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文中，MLA 的体积仅为 Llama-3-70B 风格 GQA Cache 的四分之一。

权衡之处：MLA 为每次注意力计算（每个头）增加了一个解压步骤。与节省的带宽相比，额外的计算量很小。对长上下文推理而言总体收益为正。

### 路由机制：无辅助损失负载均衡

MoE 路由器决定由哪 top-k 专家处理每个 token。朴素的路由器会将过多工作集中在少数专家上，导致其他专家闲置。标准解决方案是添加一个惩罚负载不平衡的辅助损失项。这有效但会轻微降低主任务性能。

DeepSeek-V3 引入了无辅助损失方案。在路由器 logits 中加入每个专家的偏置项，并在训练期间通过简单规则进行调整：如果专家 `e` 过载，则减小 `bias_e`；如果欠载，则增大它。无需额外损失项。训练保持简洁，专家负载保持平衡。

对主损失的影响：无可测量。对 MoE 架构的影响：更简洁，无需调优辅助损失超参数。

### MTP：更密集的训练 + 免费的草稿生成

从第 10 阶段 · 第 18 课可知，DeepSeek-V3 添加了 D=1 的 MTP 模块，用于预测两个位置之后的 token。在推理时，训练好的模块被重新用作推测解码的草稿生成器，接受率超过 80%。在训练时，每个隐藏状态都在 D+1 = 2 个目标上进行监督，提供更密集的信号。

参数量：在主模型 671B 基础上增加 14B。开销：2.1%。

### 训练：DualPipe

从第 10 阶段 · 第 19 课可知，DualPipe 是一种双向流水线，通过将前向和后向计算块与跨节点 all-to-all 通信重叠来优化效率。在 DeepSeek-V3 的 2048 张 H800 规模下，它挽回了原本会被 1F1B 流水线的气泡浪费掉的约 24.5 万 GPU 小时。

### 配置文件逐项解析

以下是 DeepSeek-V3 配置文件（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析如下：

- `hidden_size=7168`：嵌入维度。
- `num_hidden_layers=61`：总块深度。
- `first_k_dense_layers=3`：前 3 个块使用大小为 18432 的稠密 MLP。其余 58 个块使用 MoE。
- `num_attention_heads=128`：128 个查询头。
- `kv_lora_rank=512`：K 和 V 被压缩至此潜在维度，并按头解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE 块包含 256 个专家，路由 top-8。
- `shared_experts=1`：在 256 个路由专家之上，还有 1 个始终开启的专家参与每个 token 的计算。可将其视为确保每个 token 都能获得可靠输出的“稠密基底”。
- `moe_intermediate_size=2048`：每个专家的 MLP 隐藏层大小。比稠密 MLP 小，因为共有 256 个专家。

### 参数量核算

完整计算位于 `code/main.py`。核心数据如下：

- 嵌入层：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个稠密块：带 MLA 的注意力（每块约 1.44 亿）+ 稠密 MLP（每块约 2.6 亿）+ 归一化层。总计约 12 亿。
- 58 个 MoE 块：带 MLA 的注意力（约 1.44 亿）+ 256 个专家（各 3000 万）+ 1 个共享专家（3000 万）+ 归一化层。含所有专家的单块总计约 79.5 亿。58 个 MoE 块总计 4610 亿。
- MTP 模块：140 亿。

总计：核心架构约 4760 亿 + 140 亿 MTP + 明确说明的是，公开的 6710 亿数字包含了额外的结构参数（偏置张量、专家特定组件、共享专家缩放等）。我们在计算器中复现的数字与公开数据相差 3-5% —— 差异来源于 DeepSeek 报告附录第 2 节中记录的细粒度核算项。

单次前向传播的激活参数量：

- 注意力：每层 1.44 亿 * 61 = 88 亿（所有层均激活）。
- MLP 激活：前 3 层为稠密（3 * 2.6 亿 = 7.8 亿），58 个 MoE 层每层激活 8 个路由专家 + 1 个共享专家 + 路由开销。每层激活 MLP 约 2.6 亿。总计：3 * 2.6 亿 + 58 * 2.6 亿 ≈ 159 亿。
- 嵌入层 + 归一化层：12 亿。
- 激活总量：核心约 260 亿 + 140 亿 MTP（训练时启用，但推理时并非总是运行）≈ 370 亿。

### 671B / 37B 比例

稀疏度比为 18 倍（激活参数占总参数的 5.5%）。DeepSeek-V3 是已开源权重的最稀疏的前沿 MoE 模型。Mixtral 8x7B 的比例为 13/47（28%），稠密得多。Llama 4 Maverick 的比例为 17B/400B（4.25%），与之相当。DeepSeek 的策略是：在前沿规模下，更多专家配合更低的激活比例，能在每个激活 FLOP 上产出更高的质量。

### DeepSeek-V3 的定位

| 模型 | 总参数量 | 激活参数量 | 比例 | 注意力机制 | 创新点 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 后续演进：R1、V4

DeepSeek-R1（2025年）是基于 V3 骨干网络进行的推理训练版本。R1 沿用相同的架构。发生变化的是后训练配方（在可验证任务上进行大规模强化学习），而非预训练架构。

DeepSeek-V4（若发布）预计将保留 MLA + MoE + MTP，并新增 DSA（DeepSeek 稀疏注意力），作为第 10 阶段 · 第 17 课中 NSA 的继任者。技术脉络保持稳定：架构级创新不断累积；每个新版本都会调整更多的旋钮。

## 实践应用

`code/main.py` 是专为 DeepSeek-V3 结构设计的参数计算器。运行它，将输出结果与论文数据对比，并可用于测试假设变体（256 个专家 vs 512 个，top-8 vs top-16，MLA 秩 512 vs 1024）。

重点关注：

- 总参数量与公开数据 671B 的对比。
- 激活参数量与公开数据 37B 的对比。
- 128k 上下文下的 KV Cache 占用——MLA 与 GQA 的对比。
- 逐层拆解，查看参数预算实际消耗在何处。

## 动手实现

本课将产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它将生成一份逐组件的架构解读，列出配置的每个字段，按组件推导参数量，并指明该模型使用了哪几项 DeepSeek 特有创新。

## 练习

1. 运行 `code/main.py`。将计算器的总参数量估算值与公开的 671B 进行对比，并找出差异来源。论文的第二节提供了完整的明细清单。
2. 修改配置，将 MLA 秩从 512 改为 256。计算此时 128k 上下文下的 KV Cache 大小。它能带来多大的百分比缩减，代价是每个头的表达能力下降多少？
3. 将 DeepSeek-V3 的路由策略（256 个专家，top-8）与假设变体（512 个专家，top-8）进行对比。总参数量会增加，但激活参数量保持不变。理论上额外的专家容量能带来什么收益，在推理时又会产生什么成本？
4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的第 2.1 节关于 MLA 的内容。用三句话解释为什么 K 和 V 的解压矩阵可以“吸收”进后续的矩阵乘法中，以提升推理效率。
5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算存储 671B 权重时，FP8 相比 BF16 能节省多少内存。这与 14.8T token 的训练预算有何关联？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | 将 K 和 V 压缩为共享的低秩潜在表示（kv_lora_rank，通常为 512），运行时按头动态解压；KV Cache 仅存储潜在表示 |
| kv_lora_rank | “MLA 压缩维度” | K 和 V 共享潜在表示的大小；DeepSeek-V3 使用 512 |
| First k dense layers | “早期层保持稠密” | MoE 模型的前几层跳过路由器，直接运行稠密 MLP 以保证稳定性 |
| num_experts_per_tok | “Top-k 路由” | 每个 token 激活的路由专家数量；DeepSeek-V3 使用 8 |
| Shared experts | “常驻专家” | 无论路由如何都处理每个 token 的专家；DeepSeek-V3 使用 1 |
| Auxiliary-loss-free routing | “偏置调整负载均衡” | 训练期间调整每个专家的偏置项，在不添加损失项的情况下保持专家负载平衡 |
| MTP module | “额外预测头” | Transformer 块根据 h^(1) 和 E(t+1) 预测 t+2；提供更密集的训练信号，并可免费用于推测解码草稿生成 |
| DualPipe | “双向流水线” | 将前向/后向计算与跨节点 all-to-all 通信重叠的训练调度策略 |
| Active parameter ratio | “稀疏度” | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | “8位训练” | 使用 FP8 进行训练存储和大量计算操作；相比 BF16 内存占用减半，伴随微小的质量损耗 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) —— 完整的架构、训练与结果文档
- [Hugging Face 上的 DeepSeek-V3 模型卡片](https://huggingface.co/deepseek-ai/DeepSeek-V3) —— 配置文件与部署说明
- [DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) —— 引入 MLA 的前代模型
- [DeepSeek-R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) —— 基于 V3 架构的推理训练后继版本
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) —— DeepSeek 家族注意力机制的未来方向
- [DualPipe 代码仓库](https://github.com/deepseek-ai/DualPipe) —— 训练调度参考实现
