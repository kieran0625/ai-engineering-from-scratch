# 综合项目 02 —— 基于代码库的 RAG（跨仓库语义搜索）

> 到 2026 年，所有严谨的工程团队都会运行一种理解语义而非仅匹配字符串的内部代码搜索工具。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图谱、Aider 的 repo-map、Pinterest 的内部 MCP——形态大同小异。流程均为：摄入多个仓库，使用 tree-sitter 解析，对函数和类级别的代码块进行向量化，执行混合搜索与重排序，最后附带引用来源生成答案。本综合项目要求你构建一个能够处理 10 个仓库共 200 万行代码，并能在每次 git push 时安全进行增量重索引的系统。

**类型：** 综合项目
**语言：** Python（数据摄入），TypeScript（API + UI）
**前置要求：** 第 5 阶段（NLP 基础）、第 7 阶段（transformers）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**耗时：** 30 小时

## 问题描述

到 2026 年，每个前沿编程代理都会配备代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 100 万 token 上下文确实有帮助，但并不能消除对排序检索的需求。在原始代码块上直接进行余弦相似度搜索会导致结果被污染，尤其是在生成式代码、单体仓库重复代码以及长尾罕见符号上。生产环境的标准方案是：对感知 AST 的代码块进行混合（稠密 + BM25）搜索，配合重排序器，并由符号引用图谱提供支持。

你将通过索引真实的代码舰队（而非单个教程仓库）来掌握这一点，并测量 MRR@10、引用忠实度以及增量新鲜度。失败模式通常来自基础设施层面：一个包含 10 万个文件的单体仓库、一次修改了半数文件的推送、一个需要跨越四个仓库才能正确回答的查询。

## 核心概念

感知 AST 的数据摄入流水线使用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界处进行分块，而不是使用固定的 token 窗口。每个代码块获得三种表示形式：稠密向量嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 术语，以及简短的自然语言摘要。摘要引入了第三种可检索模态——用户询问“X 是如何进行权限验证的”，摘要中会提到“authz”，即使代码本身只包含 `check_permission`。

检索采用混合模式。查询同时触发稠密搜索和 BM25 搜索，合并 top-k 结果，并将并集交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表连同“按文件和行号范围引用每一项声明”的指令，一起发送给长上下文合成器（Claude Sonnet 4.7 配合提示词缓存，或自托管的 Llama 3.3 70B）。没有引用的答案会被后置过滤器拒绝。

增量新鲜度是基础设施层面的挑战。Git push 触发差异计算：哪些文件变了，哪些符号变了。仅重新嵌入受影响的代码块。受影响的跨文件符号边（导入、方法调用）会被重新计算。索引在无需每次提交都重新处理 200 万行代码的情况下保持数据一致。

## 架构设计

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为备选
- 稀疏索引：Tantivy（Rust），使用 BM25F，对符号名称和代码主体进行字段加权
- 向量数据库：Qdrant 1.12（支持混合搜索），或面向 5000 万向量以下团队的 pgvector + pgvectorscale
- 代码块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，启用提示词缓存
- 重排序器：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex Workflows 用于数据摄入，LangGraph 用于查询代理
- 合成器：Claude Sonnet 4.7（100 万上下文），启用提示词缓存
- 符号图谱：Neo4j（托管）或 kuzu（嵌入式），用于存储导入和方法调用边
- 可观测性：Langfuse，为每次检索和合成步骤记录 spans

## 构建指南

1. **数据摄入遍历器。** 在每个 push hook 上迭代 git 历史。收集变更文件。对每个文件使用 tree-sitter 解析，提取带有完整源码范围的函数和类节点。输出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要器。** 将代码块批量送入 Haiku 4.5，在系统提示词前缀上启用提示词缓存。提示词：“用一句话总结此函数，明确其公开契约和副作用。”将摘要与代码块一同存储。

3. **嵌入池。** 两个并行队列：稠密向量（Voyage-code-3，批次大小 128）和摘要向量（同一模型，但作用于摘要文本）。将向量写入 Qdrant，负载字段为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名称权重 4，符号主体权重 1，摘要权重 2。支持“查找名为 X 的函数”和“查找实现 X 功能的函数”两类查询。

5. **符号图谱。** 对每个代码块记录边关系：导入（此文件从仓库 Z 使用了符号 Y）、调用（此函数调用了类 C 的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索范围。

6. **查询代理。** LangGraph 包含三个节点。`retrieve` 并行触发稠密搜索和 BM25 搜索，按（仓库、路径、符号）去重。`rerank` 对 top-50 结果运行交叉编码器，保留 top-10。`synth` 将重排序后的代码块放入上下文，调用 Claude Sonnet 4.7，缓存系统提示词，并要求提供 file:line 引用。

7. **引用强制校验。** 解析模型输出；任何缺少 `(repo/path:start-end)` 锚点的声明都会被标记为重试或直接丢弃。仅返回带引用的答案给用户。

8. **增量重索引。** 每次收到 webhook 时，计算符号级差异。仅重新嵌入文本发生变化的代码块。重新计算导入发生变化的代码块的符号边。性能指标：对于 200 万 LOC 的代码舰队，50 个文件的推送应在 60 秒内完成重索引。

9. **评估。** 标注 100 个跨仓库问题及其标准 file:line 答案。测量 MRR@10、nDCG@10、引用忠实度（可验证锚点的声明比例）以及 p50/p99 延迟。

## 使用方式

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付标准

交付技能 `outputs/skill-codebase-rag.md`。给定一组仓库语料库，它能启动数据摄入流水线、混合索引和查询代理，并为任意跨仓库问题返回带引用的答案。评分标准如下：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 题的预留测试集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 答案声明中包含可验证 file:line 锚点的比例 |
| 20 | 延迟与规模 | 在索引语料库规模下，10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 50 文件提交的 git push 到可搜索状态的耗时 |
| 15 | 用户体验与答案格式 | 引用可点击性、代码片段预览、后续交互引导 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的变化量。报告在启用重排序后差距是否缩小。

2. 向语料库中注入 20% 的生成式代码（LLM 生成的样板代码）并重新评估。观察检索污染现象。在负载中添加“generated”标志并降低这些结果的权重。

3. 在你的语料库规模下，对比基准测试 Qdrant 混合搜索与 pgvector + pgvectorscale。报告批次大小为 1 时的 p99 延迟。

4. 添加基于采样的漂移检查：每周重新运行 100 题评估。若 MRR@10 下降超过 5%，则发出告警。

5. 扩展至跨语言符号解析：例如一个通过 gRPC 调用 Go 服务的 Python 函数。利用符号图谱将它们关联起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| AST-aware chunking | “函数级切分” | 在 tree-sitter 节点边界处切割代码，而非固定 token 窗口 |
| Hybrid search | “稠密 + 稀疏” | 并行运行 BM25 和向量搜索，合并 top-k，再进行重排序 |
| Cross-encoder rerank | “第二阶段排序” | 联合打分每个（查询，候选）对的模型，比余弦相似度更准确 |
| Prompt caching | “缓存系统提示词” | 2026 版 Claude / OpenAI 的特性，可将重复的前缀 token 折扣高达 90% |
| Symbol graph | “代码图谱” | 跨文件和仓库的导入、调用、继承关系边 |
| Citation faithfulness | “接地答案率” | 用户可通过点击锚点并阅读引用片段来验证的声明比例 |
| Incremental re-index | “推送到搜索时间” | 从 git push 到变更符号可被查询的墙钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本综合项目的参考深度解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — 基于 tree-sitter 的排名仓库视图
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商业级符号图谱 RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现文档
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详细说明
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 内部平台参考案例
