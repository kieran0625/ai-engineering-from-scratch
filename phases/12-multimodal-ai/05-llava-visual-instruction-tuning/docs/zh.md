# LLaVA 与视觉指令微调

> LLaVA（2023年4月）是地球上被复制最多的多模态架构。它用两层 MLP 替换了 BLIP-2 的 Q-Former，用朴素的分词拼接替换了 Flamingo 的门控交叉注意力机制，并使用由 GPT-4 从纯文本描述中生成的 158k 个视觉指令轮次进行训练。在 2023 年至 2026 年间构建任何 VLM 的实践者都实现了某种 LLaVA 变体。LLaVA-1.5 引入了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 将单图、多图和视频统一到了同一个配方中。本课将解析该配方，实现 projector，并解释为何“更简单反而赢了”。

**类型：** 构建
**语言：** Python（标准库，用于实现 projector 和指令模板构建器）
**前置知识：** 第 12 阶段 · 02（CLIP），第 11 阶段（LLM 工程 —— 指令微调）
**耗时：** 约 180 分钟

## 学习目标

- 构建一个两层 MLP projector，将 ViT patch embeddings（维度 1024）映射到 LLM 的 embedding 维度（维度 4096）。
- 掌握 LLaVA 的两阶段训练配方：(1) 在 558k 图文对上进行 projector 对齐，(2) 在 158k 个 GPT-4 生成的指令轮次上进行视觉指令微调。
- 构建符合 LLaVA 格式的 prompt，包含图像 token 占位符、system prompt 以及 user/assistant 对话轮次。
- 解释尽管 Q-Former 在 token 预算上占优，但社区为何转向 MLP。

## 问题所在

BLIP-2 的 Q-Former（课程 12.03）将图像压缩为 32 个 token。干净、高效，适合基准测试。但它有两个问题。

首先，Q-Former 是可训练的，但其损失函数并非最终任务。第一阶段训练 ITC+ITM+ITG。第二阶段训练 LM loss。Query 学习到的是一种中间表示，随后 LLM 必须对其进行解码。信息在瓶颈处丢失。

其次，Q-Former 需要 188M 参数，且在 LLaVA 2023 年的规模下，你必须将其与目标 LLM 协同设计。更换 LLM，需重新训练 Q-Former。更换视觉编码器，也需重新训练。每种组合都是一个独立的研发项目。

LLaVA 的答案简单得令人尴尬：取出 ViT 的 576 个 patch tokens，让每个 token 通过一个两层 MLP（`1024 → 4096 → 4096`），然后将全部 576 个直接拼接到 LLM 的输入序列中。没有瓶颈。不需要针对奇怪目标的阶段一预训练。只需直接在 LM loss 上训练 MLP。

数据从何而来？LLaVA 的第二个洞察：使用 GPT-4（仅文本）来生成指令数据。将图像的 COCO 描述和边界框数据喂给 GPT-4，让它生成对话、描述和复杂推理问题。免费获得 158k 个指令-回复轮次。无需人工标注。

结果：一个 VLM 仅在 8 张 A100 上运行了一天，就在 MMMU 上击败了 Flamingo，并发布了可供社区扩展的开源 checkpoint。到 2023 年底，它已衍生出 50 多个 fork。

## 核心概念

### 架构

LLaVA-1.5 (13B 版本)：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段一冻结，阶段二可选解冻）。
- Projector：带 GELU 激活函数的两层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后续版本使用 Llama-3.1-8B）。

图像 + 文本 prompt 的前向传播过程：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM 上下文中的 576 个 token。在 2048 的上下文长度下，剩余 1472 个 token 供文本使用。在 32k 的上下文长度下，这几乎可以忽略不计。

### 阶段一：Projector 对齐

冻结 ViT。冻结 LLM。仅训练两层 MLP。数据集：558k 图文对（LAION-CC-SBU）。损失函数：基于投影后的图像 token 条件，对 caption 进行语言建模。

在 batch size 为 128 的情况下，单个 epoch 仅需数小时即可完成。Projector 学会了将 ViT 空间映射到 LLM 空间。无需特定任务的监督信号。

### 阶段二：视觉指令微调

解冻 projector（保持可训练）。解冻 LLM（通常完全解冻，有时使用 LoRA）。在 158k 个视觉指令轮次上进行训练。

指令数据是关键。Liu 等人通过以下步骤生成：
1. 选取一张 COCO 图像。
2. 提取文本描述（5 条人工 caption + 边界框列表）。
3. 使用三个 prompt 模板发送给 GPT-4：
   - 对话：“生成一段用户与助手关于此图像的来回对话。”
   - 详细描述：“给出丰富且详细的图像描述。”
   - 复杂推理：“提出一个需要结合图像进行推理的问题，然后回答它。”
4. 将 GPT-4 的输出解析为 (instruction, response) 对。

整个过程并未直接处理图像——仅依赖文本描述。GPT-4 会幻觉出合理的图像内容。存在一定噪声，但确实有效：158k 个轮次足以解锁对话能力。

### 社区为何广泛采用此方案

- 无需调优阶段一特有的损失函数。全程使用 LM loss。
- Projector 训练仅需数小时，而非数天。
- 可通过仅重新训练 projector 来切换 LLM（如 LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3）。
- 视觉指令数据流水线依赖 GPT-4，针对新领域重新生成成本极低。

### LLaVA-1.5 与 LLaVA-NeXT

LLaVA-1.5（2023年10月）增加了：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调中。
- 更完善的 system prompt。
- 上下文长度从 2048 提升至 32k。

LLaVA-NeXT（2024年1月）增加了：
- AnyRes：将高分辨率图像分割为 2x2 或 1x3 网格的 336x336 裁剪块，外加一个全局低分辨率缩略图。每个裁剪块变为 576 个 token；每幅图像总计约 2880 个视觉 token。OCR 和图表任务性能显著提升。
- 更好的指令数据混合策略，引入 ShareGPT4V（高质量 GPT-4V caption）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

课程 12.08 深入讲解了 OneVision。简而言之：使用相同的 projector，但采用课程学习策略，在一个模型中统一覆盖单图、多图和视频，并共享视觉 token 预算。

### 与 Q-Former 的对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每幅图像的视觉 token 数 | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数量 | 188M + LM | 40M + LM |
| 阶段一损失函数 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需重新训练 | 替换时仅需极少量重训 |
| 多图支持 | 别扭 | 自然（拼接） |
| 视频支持 | 别扭 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 胜在简单性和 token 灵活性。Q-Former 胜在 token 预算。到 2023 年底，token 预算已不再是硬性约束（LLM 上下文增长至 32k-128k+），简单性成为主导因素。

### Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位符 token。在分词前，它会被替换为 576 个视觉 token（若使用 AnyRes 则为 2880 个）。Tokenizer 看到的序列比其训练时的序列稍长，但 LLM 能够处理这种新颖输入，因为阶段一已经教会了它。

### 参数经济性

LLaVA-1.5-7B 参数拆解：
- CLIP ViT-L/14 @ 336：303M（阶段一冻结，阶段二常解冻）。
- Projector（2 层线性层）：~22M 可训练参数。
- Llama-7B：7B。
- 总计：7.3B 参数。阶段二可训练参数：完整的 7B + 22M projector。

阶段二的训练成本：在 8 张 A100 上约 20 小时。这是关键数字——一天、一个节点、可复现。这正是 LLaVA 得以广泛传播的原因。

## 实践应用

`code/main.py` 实现了：

1. 纯 Python 实现的两层 MLP projector（玩具规模下维度为 16 → 32 → 32）。
2. Prompt 构建流水线：system prompt + `<image>` 被替换为 N 个投影 token + user 轮次 + assistant 生成占位符。
3. 可视化器，展示 576 个 token 的视觉块在 LLM 上下文中的占比（消耗 2k / 32k / 128k 上下文的百分比）。

## 交付成果

本课将产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列 checkpoint，它会运行一套包含 10 个 prompt 的 vibes-eval 评估套件（3 个 captioning、3 个 VQA、2 个推理、2 个拒绝场景），并输出人类可读的评分卡。这不是基准测试，而是一个冒烟测试，用于确认 projector 和 LLM 连接良好。

## 练习

1. 计算在 `1024 → 4096 → 4096` 配置下两层 MLP projector 的可训练参数量。考虑 GELU 和 bias，它占 LLaVA-13B 的比例是多少？

2. 为“拒绝”场景构建一个 LLaVA prompt —— 图像中包含私人个体。写出预期的 assistant 回复。为什么 LLaVA 应在零样本情况下拒绝此类请求？需要什么样的训练数据来强化这种拒绝行为？

3. 阅读 LLaVA-NeXT 博客中的 AnyRes 部分。计算 AnyRes 下 1344x672 图像的视觉 token 数量。与 336x336 下的基础 576 个 token 进行比较。

4. LLaVA 阶段一的 projector 使用 caption 的 LM loss 进行训练。如果跳过阶段一直接进入阶段二（视觉指令微调）会发生什么？请引用 Prismatic VLMs 的消融实验（arXiv:2402.07865）给出答案。

5. LLaVA-Instruct-150k 使用 GPT-4 配合 COCO caption 生成指令。针对一个新领域（医学 X 光片、卫星影像），描述生成领域指令的四步数据流水线。每一步可能出现什么问题？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Projector | “MLP 桥梁” | 带 GELU 的两层 MLP，将 ViT 维度映射到 LLM 维度 |
| Image token | “<image> 占位符” | 推理前被 N 个投影视觉 token 替换的 prompt 标记 |
| Visual instruction tuning | “LLaVA 阶段二” | 在 GPT-4 生成的 (image, instruction, response) 三元组上进行训练 |
| Stage 1 alignment | “Projector 预训练” | 冻结 ViT 和 LLM，使用 caption 的 LM loss 训练 projector |
| AnyRes | “多裁剪平铺” | 将高分辨率图像分割为平铺网格，并拼接每个平铺块的视觉 token |
| LLaVA-Instruct | “GPT-4 生成” | 基于 COCO caption + GPT-4 合成的 158k 指令-回复对 |
| Vision encoder freeze | “骨干网络锁定” | 阶段一中 CLIP 权重不更新，阶段二有时也不更新 |
| ShareGPT4V | “更优质的 caption” | 由 GPT-4V 生成的 1M 密集 caption，用于更高质量的对齐 |
| VQA | “视觉问答” | 回答关于图像的开放式问题的任务 |
| Prismatic VLMs | “设计空间论文” | Karamcheti 2024 的消融实验，系统测试 projector 和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) —— LLaVA 原始论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) —— LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) —— 密集 caption 数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) —— 设计空间消融实验。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) —— 统一单图、多图与视频。
