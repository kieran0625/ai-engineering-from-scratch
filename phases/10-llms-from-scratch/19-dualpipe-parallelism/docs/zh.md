# DualPipe 并行

> DeepSeek-V3 在 2,048 张 H800 GPU 上进行了训练，MoE 专家分散在各个节点上。跨节点的专家 all-to-all 通信成本为：每 1 GPU 小时的计算对应 1 GPU 小时的通信。GPU 有一半时间是空闲的。DualPipe（DeepSeek，2024年12月）是一种双向流水线，它将前向和后向计算与它们触发的 all-to-all 通信重叠。气泡减少，吞吐量提升，而在专家并行（Expert Parallelism）已经将专家分散到各个 rank 的情况下，维护两份模型参数副本（即名称中“双”的含义）的成本是低廉的。本课是一篇 Learn 类型的导览，详解 DualPipe 的实际运作机制，以及为什么海智实验室（Sea AI Lab）的 DualPipeV 改进版以略微增加气泡为代价，消除了 2 倍参数复制成本。

**类型：** Learn
**语言：** Python（标准库、调度模拟器）
**前置知识：** Phase 10 · 05（分布式训练、FSDP、DeepSpeed），Phase 10 · 14（开源模型架构与 MoE）
**预计时间：** ~60 分钟

## 学习目标

- 说出 DualPipe 前向-后向 chunk 的四个组成部分，并解释为何每个部分都有独立的重叠窗口。
- 解释大规模下的流水线气泡问题，以及实践中“无气泡”与营销术语中的区别。
- 手动推导一个包含 8 个 PP rank 和 16 个 micro-batch 的 DualPipe 调度表，并确认前向流和后向流如何互相填补空闲时隙。
- 说明 DualPipeV（海智实验室，2025）所做的权衡：在专家并行不活跃时，以略微增大的气泡为代价，消除 2 倍参数复制开销。

## 问题所在

在 2k 张 H800 GPU 上训练 671B MoE 模型会面临三个叠加的瓶颈：

1. **显存压力。** 每张 GPU 持有模型的一个切片。在序列长度 8k、61 层、128 个注意力头的配置下，激活值显存占用极其庞大。
2. **流水线气泡。** 传统的流水线并行（GPipe、1F1B）会导致 GPU 在等待本阶段输入或梯度时空闲。在 8 个阶段下，即使采用 1F1B 调度，GPU 时间也有约 12% 浪费在气泡上。
3. **跨节点 all-to-all 通信。** 结合专家并行的 MoE 将专家分散到各个节点。每次前向传播都会触发一次 all-to-all 将 token 分发到对应专家，另一次用于聚合结果。在 2k 张 GPU 规模下，这很容易导致计算与通信的时间比达到 1:1。

针对这些问题已有各自的解决方案：用梯度检查点（gradient checkpointing）解决显存，用 Zero Bubble（海智实验室，2023）解决流水线气泡，用专家并行通信内核优化 all-to-all。DualPipe 的做法是让它们协同工作。它在一个完整的前向-后向 chunk 内重叠计算与通信，同时从流水线两端注入 micro-batch，并利用生成的调度表将 all-to-all 通信隐藏到计算窗口中。

官方报告结果：几乎消除了流水线气泡，在 DeepSeek-V3 的 14.8T token 训练中实现了超过 95% 的 GPU 利用率。

## 核心概念

### 流水线并行回顾

将 N 层模型拆分到 P 个设备上。设备 `i` 持有第 `i * N/P .. (i+1) * N/P - 1` 层。一个 micro-batch 从前向后流经设备 0 到 P-1，然后从后向前返回。每个设备必须等到上游设备发送输出后才能开始前向阶段，也必须等到下游设备发送上游梯度后才能开始后向阶段。

GPipe（Huang 等，2019）一次只调度一个 micro-batch，浪费了大部分 GPU 时间。1F1B（Narayanan 等，2021）交错调度多个 micro-batch 的前向和后向传播。Zero Bubble（Qi 等，2023）将后向传播拆分为两部分——基于输入的梯度（B）和基于权重的梯度（W）——并通过调度将它们填入气泡中。经过 Zero Bubble 优化后，流水线已近乎紧凑。

DualPipe 是下一步演进。它在上述基础上引入了两个核心思想：

### 思想 1：Chunk 分解

每个前向 chunk 被拆分为四个组成部分：

- **Attention（注意力）。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all dispatch（分发）。** 跨节点通信，将 token 发送到对应的专家。
- **MLP（多层感知机）。** MoE 专家计算。
- **All-to-all combine（聚合）。** 跨节点通信，将专家输出结果聚合回来。

后向 chunk 则包含上述各部分的梯度版本。DualPipe 的调度策略使得 all-to-all dispatch 与下一个 chunk 的 attention 计算并行执行，而 all-to-all combine 与后续 chunk 的 MLP 计算并行执行。

### 思想 2：双向调度

大多数流水线调度仅从 stage 0 注入 micro-batch 并向 stage P-1 流动。DualPipe 从**两端**同时注入 micro-batch。Stage 0 看到从自身起始的前向 micro-batch；stage P-1 也看到从自身起始的前向 micro-batch。两条流在中间汇合。

为了实现这一点，设备 `i` 必须同时持有早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 中“双”的含义：每个设备保留两份所需服务模型层的副本（分别对应两个方向）。在 DeepSeek-V3 的规模下，这意味着 2 倍的参数复制开销。但由于 Expert Parallelism 已经将 MoE 专家切分得极细，因此重复两份非专家层参数的成本九牛一毛。

关键在于，一个方向的前向流与另一个方向的后向流，恰好会在单向调度产生气泡的位置重叠。气泡由此消失。

### 手动推导的调度表

假设 P = 4 个 rank，8 个 micro-batch，分为 4 个前向 / 4 个反向。时间从左向右流逝；行代表设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读 “F4/F5R” 标记：rank 1 在同一时隙内，既运行 micro-batch 4 的前向（在流水线中从左向右），又运行 micro-batch 5 的前向（在流水线中从右向左）。这就是操作层面上“双向”的含义。

在 rank 2 处交叉流较早重叠，在 rank 0 和 P-1 处重叠最晚。在调度的稳定中间阶段，每个 rank 都在运行 X 方向的前向与 Y 方向的后向重叠任务。计算单元持续忙碌。前向传播的 all-to-all dispatch 隐藏在后向计算中，all-to-all combine 隐藏在前向计算中。气泡被彻底挤出。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 优化将其降低但未归零。DualPipe 在稳定阶段，只要 micro-batch 数量能被流水线深度的 2 倍整除，即可实现零气泡。在稳定阶段之外（预热和冷却期），仍存在少量气泡，但它不会随 micro-batch 数量的增加而增长——这是论文强调的关键特性。

营销语境下称为“无气泡”。技术语境下指：气泡大小不随 micro-batch 数量增长。海智实验室的后续分析（DualPipeV / Cut-in-half）表明，仅在 Expert Parallelism 不是瓶颈时才能实现完全零气泡；当由 EP 驱动 all-to-all 通信时，必然存在某种调度妥协。

### DualPipeV —— 改进版

海智实验室（2025）指出，当 EP 通信重叠并非核心诉求时，2 倍参数复制是浪费。他们的 DualPipeV 调度将双向注入折叠为一种“V 形”调度，仅需单份参数副本即可运行。其气泡略大于 DualPipe，但内存节省显著。DeepSeek 在其开源 DualPipe 实现中将 DualPipeV 作为 EP-off 模式采纳。

权衡对比：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备参数副本数 | 2 | 1 | 1 | 1 |
| 气泡随 micro-batch 变化 | 恒定 | 微小增长 | 增长 | 增长 |
| 计算-通信重叠度 | 完全 | 部分 | 最小 | 部分 |
| 适用场景 | EP 重度 MoE | 稠密模型或 EP 轻量 | 基线 | 任意流水线 |

### 对 14.8T token 训练的意义

DeepSeek-V3 的预训练在 2,048 张 H800 GPU 上消耗了约 280 万 GPU 小时，处理了 14.8T token。若使用朴素的 1F1B，将有 12-15% 的时间浪费在流水线气泡上——相当于 34 万至 42 万 GPU 小时，足以完整训练一个 70B 模型。DualPipe 回收了其中绝大部分。由于缺乏内部日志难以直接量化具体贡献，但论文声称其在整个训练期间平均 GPU 利用率超过 95%。

对于较小规模的训练（少于 1k GPU），DualPipe 属于杀鸡用牛刀——相对于总成本，流水线气泡占比更小，且稠密模型训练很少触及 all-to-all 瓶颈。但在数千 GPU 规模的前沿 MoE 训练中，它几乎是必选项。

### 在技术栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 负责在 rank 间分片模型参数；DualPipe 负责在 rank 间调度计算。两者可结合使用。
- 兼容 **ZeRO-3** 梯度分片。双副本管理的记账逻辑需与 ZeRO 的分片梯度协同工作。
- 需要针对特定集群拓扑优化的**自定义 all-to-all 内核**。DeepSeek 的开源内核即为参考实现。

## 实践应用

`code/main.py` 是一个流水线调度模拟器。它接收 `(P, n_micro_batches, schedule)` 作为输入，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 在稳定阶段的利用率。这是一个教学工具——数据与论文中的定性结论一致，并非对生产环境实测加速比的承诺。

模拟器的价值：使用不同的 P 值和 micro-batch 数量运行它，观察 1F1B 的气泡比例如何增长，而 DualPipe 则不会。

实际训练集成的注意事项：

- 选择能整除你的 micro-batch 数量的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考实现。
- 首次使用时，预计要花费一周时间调试调度逻辑本身。账本管理非常繁琐。
- 监控每个 rank 的 GPU 利用率，而非仅看聚合指标。DualPipe 的收益正来自于消除短板（stragglers）。

## 交付成果

本课将产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑结构、互联方式、模型形状），它将推荐一套流水线并行策略、应使用的调度算法，以及在目标规模下的预期气泡比例。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率的差异，并将其表示为每百万 token 训练中恢复的 GPU 小时数。
2. 手绘 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。在每个时隙中标注 micro-batch ID 和方向。找出第一个没有气泡的时隙。
3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）中的图 5。指出 DualPipe 前向 chunk 内 all-to-all dispatch 的重叠窗口。解释计算调度是如何将其隐藏的。
4. 计算 DualPipe 在 P=8 流水线阶段的 70B 稠密模型，以及 P=16 流水线阶段的 671B MoE 模型上的 2 倍参数开销。说明为何 MoE 案例的开销按比例更小（大部分参数是专家，分布在大型 EP 组中进行分片）。
5. 将 DualPipe 与 Chimera（2021 年的另一种双向调度器）进行对比。以论文第 3.4 节为参考，指出 DualPipe 新增而 Chimera 不具备的两个具体特性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Pipeline bubble（流水线气泡） | “每个 rank 的空闲时间” | 因流水线阶段等待输入或梯度而浪费的 GPU 周期 |
| 1F1B | “默认流水线调度” | 一前一后的交错调度；DualPipe 超越的基线 |
| Zero Bubble | “海智实验室 2023” | 将后向拆分为 B（输入梯度）和 W（权重梯度）；使流水线近乎紧凑 |
| DualPipe | “DeepSeek-V3 调度” | 双向流水线 + 计算-通信重叠；气泡不随 micro-batch 数量增长 |
| DualPipeV | “减半方案（Cut-in-half）” | V 形改进版，以略微增大的气泡为代价消除 2 倍参数复制 |
| Chunk | “流水线工作单位” | 单个 micro-batch 穿过单个流水线阶段的一次前向或后向传播 |
| All-to-all dispatch | “将 token 发送给专家” | 跨节点通信，将 token 路由到分配的 MoE 专家 |
| All-to-all combine | “将专家输出聚合回来” | 跨节点通信，在 MLP 之后收集专家输出 |
| Expert Parallelism (EP)（专家并行） | “专家分布在不同 GPU 上” | 将 MoE 专家跨 rank 分片，使不同 GPU 持有不同专家 |
| Pipeline Parallelism (PP)（流水线并行） | “层分布在不同 GPU 上” | 将模型层跨 rank 分片；DualPipe 进行调度的维度 |
| Bubble fraction（气泡比例） | “浪费的 GPU 时间” | (bubble_time / total_time)；DualPipe 致力于将其趋近于零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 技术报告 (arXiv:2412.19437)，第 3.3.2 节与图 5](https://arxiv.org/abs/2412.19437) —— DualPipe 的核心参考文献
- [DeepSeek — DualPipe GitHub 仓库](https://github.com/deepseek-ai/DualPipe) —— 开源参考实现，包含 DualPipeV（Cut-in-half）模式
- [Qi 等 — Zero Bubble 流水线并行 (arXiv:2401.10241, 海智实验室 2023)](https://arxiv.org/abs/2401.10241) —— Zero Bubble 的前置工作
- [海智实验室 — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) —— 指导 DeepSeek 采用 EP-off 模式的 DualPipeV 分析
- [Narayanan 等 — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) —— DualPipe 对比的 1F1B 调度基准
- [Huang 等 — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) —— 原始流水线并行论文及气泡问题起源
