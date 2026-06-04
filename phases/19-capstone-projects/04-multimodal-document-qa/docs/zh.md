# 综合项目 04 — 多模态文档问答（视觉优先的 PDF、表格与图表）

> 2026 年的文档问答前沿已从“先 OCR 后文本”转向视觉优先的晚期交互。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，使用多向量晚期交互进行嵌入，并让查询直接关注图像块（patches）。在财务 10-K 文件、科学论文和手写笔记上，这种模式大幅优于先 OCR 后文本的方法。在 1 万页数据上端到端构建该流水线，并发布与“先 OCR 后文本”方法的对比结果。

**类型：** 综合项目
**语言：** Python（流水线）、TypeScript（查看器 UI）
**前置要求：** Phase 4（计算机视觉）、Phase 5（NLP）、Phase 7（transformers）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**耗时：** 30 小时

## 问题描述

企业拥有大量被传统 OCR 流水线破坏的 PDF：包含旋转表格的扫描版 10-K 文件、公式密集的学术论文、仅作为图像才有意义的图表以及手写批注。将其视为纯文本会导致丢失一半的关键信息。2026 年的解决方案是在原始页面图像上执行晚期交互多向量检索。ColPali（Illuin Tech）率先引入了该技术；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确率。在 ViDoRe v3 基准测试中，视觉优先的检索得分显著高于“先 OCR 后文本”方法——且在图表、表格和手写内容上的优势更为明显。

权衡在于存储与延迟。一个 ColQwen 嵌入包含约 2048 个图像块向量，而非单个 1024 维向量。原始存储需求会急剧膨胀。DocPruner（2026版）可在不造成可测量准确率损失的情况下实现 50% 的剪枝压缩。你需要索引 1 万页文档，测量 ViDoRe v3 的 nDCG@5 指标，确保答案生成延迟低于 2 秒，并与“先 OCR 后文本”的基线进行直接对比。

## 核心概念

晚期交互意味着每个查询词元（token）都会与每个图像块词元进行打分，并将每个查询词元的最高分相加。这样无需依赖单一的全局池化向量即可实现细粒度匹配。多向量索引（如 Vespa、Qdrant 多向量或 AstraDB）存储每个图像块的嵌入向量，并在检索时运行 MaxSim 算法。

答案生成器是一个视觉语言模型（VLM），它接收查询和检索到的 top-k 页面图像，并输出带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年前沿的选择。针对公式和科学符号，可接入 OCR 回退方案（Nougat、dots.ocr）作为可选的文本通道。

评估采用二维矩阵。一个维度是内容类型（纯文本段落、密集表格、柱状图/折线图、手写笔记、公式）。另一个维度是检索方法（视觉优先晚期交互 vs 先 OCR 后文本 vs 混合方法）。每个单元格记录 nDCG@5 和答案准确率。最终报告即为交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF (fitz)，180 DPI，纵向标准化
- 晚期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（vidore team 在 Hugging Face 发布）
- 索引：带有多向量字段的 Vespa，或 Qdrant 多向量，或支持 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差图像块，50% 压缩率下准确率损失 < 0.5%）
- OCR 回退方案（公式/密集表格）：dots.ocr 或 Nougat
- VLM 答案生成器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为备用
- 评估：ViDoRe v3 基准测试，M3DocVQA（多页推理）
- 查看器 UI：Next.js 15，支持证据区域 Canvas 叠加

## 构建步骤

1. **数据摄入。** 遍历包含 10-K 文件、科学论文和扫描文档的 1 万页 PDF 语料库。将每页渲染为 1536x2048 的 PNG 图片。持久化存储至 `{doc_id, page_num, image_path}`。

2. **嵌入处理。** 对每张页面图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个图像块嵌入，维度为 128。应用 DocPruner 保留信号最强的前 50%。写入 Vespa 多向量字段或 Qdrant 多向量索引。

3. **查询检索。** 针对每个传入查询，使用查询塔（query tower）进行嵌入（词元级嵌入）。与索引运行 MaxSim：对每个查询词元，取其与页面图像块嵌入的最大点积，然后求和。返回 top-k 页面。

4. **答案合成。** 调用 Qwen3-VL-30B，输入查询和 top-5 页面图像。提示词：“仅使用提供的页面回答问题。通过 (doc_id, page) 引用每个主张，并指明具体区域（图表、表格或段落）。”

5. **证据区域。** 后处理答案以提取引用的区域。如果 VLM 输出了边界框（Qwen3-VL 支持此功能），则在查看器中将其渲染为叠加层。

6. **OCR 回退。** 对于被识别为公式密集的页面（基于图像方差的启发式规则），运行 Nougat 或 dots.ocr，并将 OCR 文本作为额外通道与图像一同传入。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。同时在相同语料库上运行“先 OCR 后文本”流水线并使用相同的合成器。生成内容类型 × 检索方法的对比矩阵。

8. **UI 开发。** 先构建 Streamlit 原型；随后开发基于 Next.js 15 的生产级查看器，支持逐页证据区域叠加显示。

## 使用指南

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付说明

`outputs/skill-doc-qa.md` 描述了交付物：一个针对特定语料库优化的视觉优先多模态文档问答系统，并在 ViDoRe v3 上与“先 OCR 后文本”基线进行了对比评估。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准测试数值与 OCR-文本基线及已发布排行榜的对比 |
| 20 | 证据区域定位 | 实际包含答案片段的引用区域占比 |
| 20 | 存储与延迟工程 | DocPruner 压缩比、索引 p95 延迟、答案生成 p95 延迟 |
| 20 | 多页推理能力 | 人工标注的 100 题多页问答集上的准确率 |
| 15 | 溯源检查体验 | 查看器清晰度、叠加层保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料库上对比测试 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面前者正确而后者遗漏？为索引添加“内容类别”标签以实现按类型路由。
2. 激进剪枝嵌入向量（75%、90%）。寻找压缩拐点：ViDoRe nDCG@5 降至 OCR 基线以下的临界点。
3. 构建混合架构：并行运行“先 OCR 后文本”与 ColQwen，使用 RRF 融合，再用交叉编码器重排序。混合方案是否优于单一方案？它在哪些方面提升最显著？
4. 将 Qwen3-VL-30B 替换为更小的 VLM（如 Qwen2.5-VL-7B）。绘制“每美元准确率”曲线。
5. 增加手写笔记支持。渲染手写语料库，使用 ColQwen 嵌入并测量检索效果。与手写 OCR 流水线进行对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 晚期交互 (Late interaction) | “ColPali 风格检索” | 查询词元独立对页面图像块打分；MaxSim 进行聚合 |
| 多向量 (Multi-vector) | “逐图像块嵌入” | 每个文档包含多个向量，而非单一池化向量 |
| MaxSim | “晚期交互打分” | 对每个查询词元，取文档向量的最大相似度并求和 |
| DocPruner | “图像块压缩” | 2026 年剪枝技术，保留 50% 图像块且准确率损失可忽略不计 |
| ViDoRe v3 | “文档检索基准” | 2026 年衡量视觉文档检索的标准基准 |
| 证据区域 (Evidence region) | “引用边界框” | 源页面上用于定位答案片段位置的边界框 |
| OCR 回退 (OCR fallback) | “公式通道” | 针对公式或表格密集型页面，与视觉管道并行的文本处理管线 |

## 延伸阅读

- [ColPali (Illuin Tech) 仓库](https://github.com/illuin-tech/colpali) —— 晚期交互文档检索参考
- [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) —— 奠基性方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) —— 生产就绪的检查点
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) —— 多模态多页 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) —— 参考服务架构
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) —— 备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) —— 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) —— 支持公式的 OCR 回退方案
