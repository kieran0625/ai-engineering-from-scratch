# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，拆分为块（chunks），嵌入这些块并存储向量。每一步都会丢失信号：OCR 会遗漏图表数据，分块会破坏表格行，文本嵌入会忽略图像。ColPali（Faysse 等人，2024年7月）提出了一个更简单的问题：为什么一定要提取文本？直接通过 PaliGemma 嵌入页面图像，使用类似 ColBERT 的晚期交互（late interaction）进行检索，并保留文档携带的所有布局、图像、字体和格式信号。已发布的基准测试显示：在视觉丰富型文档上，端到端准确率比文本 RAG 高出 20-40%。ColQwen2、ColSmol 和 VisRAG 进一步扩展了这一模式。本课程将深入阅读该视觉原生 RAG 的核心论文，并构建一个小型的类 ColPali 索引器。

**类型：** 动手实践
**语言：** Python（标准库、多向量索引器 + MaxSim 评分器）
**前置知识：** 第 11 阶段（LLM 工程 — RAG 基础）、第 12 阶段 · 05（LLaVA）
**预计时间：** ~180 分钟

## 学习目标

- 解释双编码器检索（每个文档一个向量）与晚期交互检索（每个文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作，以及 ColPali 如何将其从文本 token 推广到图像 patch。
- 构建一个小型类 ColPali 索引器：页面 → patch 嵌入 → 对查询词嵌入执行 MaxSim → 返回 top-k 页面。
- 对比 ColPali + Qwen2.5-VL 生成器与文本 RAG + GPT-4 在发票/财务报告用例中的表现。

## 问题所在

基于 PDF 的文本 RAG 丢弃了文档的大部分信息。财务报告中 Q3 的收入增长通常位于图表中；医疗报告的结论包含在带标注的图像中；法律合同的签名区块是一个布局事实，而非文本事实。

文本 RAG 流水线：

1. PDF → 通过 OCR / pdftotext 转为文本。
2. 文本 → 拆分为 300-500 token 的块。
3. 块 → 双编码器嵌入（单个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。未捕获图表。表格被跨块切断。多栏布局被展平。图像标注消失。

ColPali 的解决方案：跳过 OCR，直接嵌入页面图像。使用类似 ColBERT 的晚期交互进行检索，使模型能够在查询时关注细粒度的 patch。

## 核心概念

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) 是一种文本检索方法。它不为每个文档生成一个向量，而是为每个 token 生成一个向量。在查询时：

- 查询 token 获得各自的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常已缓存）。
- 得分 = 对所有查询 token 求和，取每个查询 token 与所有文档 token 的余弦相似度的最大值：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token “挑选”出与其最匹配的文档 token。最终得分为总和。

优点：召回率高，能处理词级语义。缺点：每个文档需要 N_d 个向量，存储成本高。

### ColPali

ColPali (Faysse et al., arXiv:2407.01449) 将 ColBERT 模式应用于图像。

- 每个页面由 PaliGemma（ViT + 语言模型）编码为 patch 嵌入：每页 N_p 个向量。
- 每个用户查询（文本）被编码为查询 token 嵌入：N_q 个向量。
- 得分 = Σ_i max_j cos(q_i, p_j)，即查询文本 token 与页面图像 patch 之间的 MaxSim。
- 按总分检索 top-k 页面。

文档摄入时：用 PaliGemma 编码每一页，存储所有 patch 嵌入。查询时：编码查询 token，与所有已存储的页面嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富型文档上，端到端性能比文本 RAG 提升 20-40%。每个 patch 向量捕捉局部布局和内容的细节。

缺点：每页 N_p 个 patch × 4 字节浮点数 × D 维向量 = 存储量增长迅速。可通过 PQ / OPQ 量化缓解。

### ColQwen2 与 ColSmol

ColQwen2 (illuin-tech, 2024-2025) 将 PaliGemma 替换为 Qwen2-VL。基座编码器更强，检索效果更好。

ColSmol 是面向本地/边缘设备的小规模变体。约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) 是另一种变体：不在 patch 上使用 MaxSim，而是通过 VLM 将每页池化为单个向量，然后进行双编码器检索。索引更快、存储更小，但召回率较弱。

质量与成本的权衡：追求质量选 ColPali，追求规模选 VisRAG。

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) 将多模态检索扩展到多页多文档推理。跨文档检索页面，为 VLM 组合多页上下文。

### ViDoRe — 基准测试

ColPali 的配套基准测试。Visual Document Retrieval Evaluation（视觉文档检索评估）。任务包括财务报告、科学论文、行政文档、医疗记录、手册等。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上的得分约为 80% nDCG@5；同一批文档上的文本 RAG 得分约为 50-60%。

### 端到端 RAG 流水线

对于视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。
2. 查询：用户文本 → 查询 token 嵌入 → 与所有索引页面计算 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 生成答案。

全程无需 OCR。图像、图表、字体、布局全部流入最终答案。

### 存储计算

一份 50 页的财务报告，每页 729 个 patch，128 维嵌入，4 字节浮点数：

- ColPali：50 * 729 * 128 * 4 字节 ≈ 18 MB 原始大小，PQ 压缩后约 4 MB。
- 文本 RAG：50 个块 * 768 维 * 4 字节 ≈ 150 kB。

ColPali 每个文档的存储量约为文本 RAG 的 30 倍。在大规模场景下，OPQ / PQ 可将其降至约 5-10 倍，通常可接受。

### 何时文本 RAG 仍占优

- 纯文本且无布局信号的文档（维基百科文章、聊天记录）。文本 RAG 更简单且存储成本更低。
- 数百万页的档案库，其中存储成本占主导。
- 严格的监管要求，要求在检索的同时提供可提取的 OCR 文本。

对于 2026 年的其他所有场景——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

## 动手实现

`code/main.py`：

- 简易 patch 编码器：将“页面”（特征向量的小网格）映射为 patch 嵌入数组。
- MaxSim 评分器：计算查询 token 嵌入集与页面 patch 集之间的类 ColBERT 得分。
- 索引 5 个简易页面，运行 3 个查询，返回带得分的 top-k 结果。

## 交付成果

本课程将产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，能够选择 ColPali / ColQwen2 / VisRAG / 文本 RAG 并估算存储规模。

## 练习

1. 一份 200 页的年度报告，每页 729 个 patch，128 维嵌入，4 字节浮点数。计算原始存储大小和 PQ 压缩（8 倍）后的存储大小。

2. MaxSim 公式为 Σ_i max_j cos(q_i, p_j)。这种求和方式捕捉到了简单平均相似度无法捕捉的什么信息？

3. ColPali 以 patch 集的形式索引页面。如果我们改为在词级别进行索引（如 ColBERT 所做的那样），会有什么变化？权衡是什么？

4. 为一个 100 万页的语料库设计端到端流水线，查询延迟预算为 500ms。选择 ColQwen2 或 VisRAG 并说明理由。

5. 阅读 M3DocRAG (arXiv:2411.04952)。描述其多页注意力模式，并说明其与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| Late interaction | “类 ColBERT 风格” | 使用逐 token 或逐 patch 嵌入 + MaxSim 进行检索，而非单个文档向量 |
| MaxSim | “Patch 上的最大值” | 对每个查询 token，选取相似度最高的文档 token；对所有查询 token 求和 |
| Bi-encoder | “单向量” | 每个文档一个向量；速度更快但粒度损失较大 |
| Multi-vector | “每文档多向量” | 每个文档/页面存储 N_p 个向量；存储成本增加但召回率提升 |
| Patch embedding | “页面特征” | 来自 VLM 编码器的每个图像 patch 对应一个向量，按页面缓存 |
| ViDoRe | “视觉文档基准” | ColPali 的视觉文档检索基准测试套件 |
| PQ quantization | “乘积量化” | 一种压缩技术，在缩小存储约 8 倍的同时保持向量相似度 |

## 延伸阅读

- [Faysse 等人 — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu 等人 — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho 等人 — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
