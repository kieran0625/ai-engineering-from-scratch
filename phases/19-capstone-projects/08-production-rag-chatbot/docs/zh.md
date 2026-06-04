# 综合项目 08 — 面向受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都采用相同的生产级架构。使用 docling 或 Unstructured 进行文档解析，ColPali 处理视觉内容。混合检索。使用 bge-reranker-v2-gemma 进行重排序。使用 Claude Sonnet 4.7 结合提示词缓存（命中率 60-80%）进行内容生成。通过 Llama Guard 4 和 NeMo Guardrails 进行安全守卫。使用 Langfuse 和 Phoenix 进行监控。基于包含 200 个问题的黄金测试集使用 RAGAS 进行评估。在一个受监管的领域（法律、临床、保险）中构建一个，只要通过黄金测试集评估、红队测试以及漂移仪表盘监控，即视为完成综合项目。

**Type:** 综合项目
**Languages:** Python（管道与 API）、TypeScript（聊天界面）
**Prerequisites:** Phase 5（NLP）、Phase 7（transformers）、Phase 11（LLM engineering）、Phase 12（multimodal）、Phase 17（infrastructure）、Phase 18（safety）
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 hours

## Problem

面向受监管领域的 RAG（法律合同、临床试验方案、保险条款）是 2026 年最主流的生产级架构，因为其投资回报率显而易见，且业务风险具体可衡量。Harvey（为 Allen & Overy 律所构建）将其用于法律场景。Mendable 提供开发者文档版本。Glean 覆盖企业搜索。标准模式为：高保真度数据摄入、混合检索配合重排序、通过强制引用和提示词缓存进行内容生成、多层安全守卫，以及持续监控数据漂移。

难点不在于模型本身，而在于：符合管辖权要求的合规性（HIPAA、GDPR、SOC2）、可追溯至引用级别的审计能力、成本控制（高命中率下提示词缓存可带来 60-90% 的成本折扣）、通过 RAGAS faithfulness 指标检测幻觉，以及在源文档更新而索引未同步时检测数据漂移。本综合项目要求你交付一套完整系统，并在附带红队测试套件的情况下，通过包含 200 个问题的黄金测试集验证。

## Concept

该管道分为两个部分。**Ingestion**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉密集型文档；文本块会生成摘要、标签和基于角色的访问权限标签。向量存入 pgvector + pgvectorscale（5000 万向量以内）或 Qdrant Cloud；同时运行稀疏 BM25 检索。**Conversation**：LangGraph 管理记忆和多轮对话；每次查询执行混合检索，使用 bge-reranker-v2-gemma-2b 进行重排序，通过 Claude Sonnet 4.7（启用提示词缓存）进行内容生成，输出结果经过 Llama Guard 4 和 NeMo Guardrails 过滤，最终返回带有引用锚点的回答。

评估栈包含四个层级。**Golden set**（200 条带引用的标注问答对）用于评估正确性。**Red team**（越狱攻击、PII 信息提取尝试、非领域问题）用于评估安全性。**RAGAS** 用于自动逐轮评估 faithfulness、answer relevance 和 context precision。**Drift dashboard**（Arize Phoenix）每周监控检索质量和幻觉得分。

提示词缓存是控制成本的关键杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存 system prompts 与检索到的上下文。在 60-80% 的命中率下，单次查询成本可降低 3-5 倍。管道设计必须确保前缀稳定（system prompt + 重排序后的上下文优先），以实现高缓存命中率。

## Architecture

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Stack

- Ingestion: Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉密集型 PDF
- Vector DB: pgvector + pgvectorscale（5000 万向量以内）；否则使用 Qdrant Cloud
- Sparse: Tantivy BM25（支持字段权重）
- Orchestration: LlamaIndex Workflows（数据摄入）+ LangGraph（对话交互）
- Re-ranker: 自托管 bge-reranker-v2-gemma-2b 或托管 Voyage rerank-2
- LLM: Claude Sonnet 4.7（启用提示词缓存）；备用方案为自托管 Llama 3.3 70B
- Eval: 在线 RAGAS 0.2，DeepEval 用于幻觉和红队测试套件
- Observability: 自托管 Langfuse（含标注队列）；Arize Phoenix 用于漂移监控
- Guardrails: Llama Guard 4 输入/输出分类器、NeMo Guardrails v0.12 策略引擎、Presidio PII 信息清洗
- Compliance: 文本块附加基于角色的访问标签；GDPR/HIPAA 管辖权标签

## Build It

1. **Ingestion.** 使用 Unstructured 或 docling 解析你的语料库（正式构建建议 1000-10000 份文档）。对于扫描件或视觉密集型页面，路由至 ColPali 处理。生成带有摘要、role-labels 和 jurisdiction tags 的文本块。
2. **Index.** 将稠密向量（Voyage-3 或 Nomic-embed-v2）存入 pgvector + pgvectorscale。通过 Tantivy 建立 BM25 旁路索引。将角色和管辖权过滤器作为 payload。
3. **Hybrid retrieve.** 首先按 role+jurisdiction 过滤；然后并行执行稠密检索与 BM25 检索；使用 reciprocal rank fusion 合并结果；取前 20 名送入 re-ranker；取前 5 名送入 synth。
4. **Synthesize with prompt caching.** System prompt 与静态策略放入 cache header；重排序后的上下文作为 cache extension；用户问题作为未缓存的 suffix。在稳态下目标缓存命中率为 60-80%。
5. **Guardrails.** 输入端使用 Llama Guard 4；NeMo Guardrails rails 拦截非领域问题或 policy-forbidden 话题；Presidio 清洗输出中意外出现的 PII 信息；最后通过 citation enforcement 后置过滤。
6. **Golden set.** 由领域专家标注 200 组问答对，包含 (answer, citations)。从 exact-citation match、answer correctness、faithfulness (RAGAS) 三个维度对 agent 进行评分。
7. **Red team.** 准备 50 个 adversarial prompts：越狱攻击（PAIR、TAP）、PII exfiltration 尝试、非领域提问、跨管辖权泄露。以 pass/fail 及 severity 进行评分。
8. **Drift dashboard.** Arize Phoenix 每周跟踪检索质量（nDCG、citation faithfulness）。下降超过 5% 时触发告警。
9. **Cost report.** Langfuse 统计：prompt-caching hit rate、tokens per query、各阶段 $/query 成本明细。

## Use It

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Ship It

`outputs/skill-production-rag.md` 描述了交付物要求。需部署一个带有合规标签的受监管领域聊天机器人，通过评分标准考核，并接入实时漂移监控进行观察。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | 黄金测试集（200 组问答）上的在线得分 |
| 20 | Citation correctness | 包含可验证来源锚点的回答比例 |
| 20 | Guardrail coverage | Llama Guard 4 通过率 + jailbreak suite 结果 |
| 20 | Cost / latency engineering | Prompt-cache hit rate、p95 latency、$/query |
| 15 | Drift monitoring dashboard | Phoenix live dashboard 及每周检索质量趋势 |
| **100** | | |

## Exercises

1. 构建另一个不同管辖权下的语料子集（例如在 GDPR 之外额外加入 HIPAA 数据）。通过 20 题的跨管辖权探测，演示 role+jurisdiction 过滤如何防止数据交叉泄露。
2. 在生产流量运行一周后，测量 prompt-cache hit rate。识别哪些查询破坏了 cache prefix。重新调整结构。
3. 添加 multi-turn memory 功能，使用 10k-token 的 summary buffer。测量随着对话轮次增加，faithfulness 是否下降。
4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 $/query 变化及 faithfulness delta。
5. 添加 "unsure" 模式：如果最高 re-ranked scores 低于阈值，agent 回复 "I do not have confident citations" 而不是强行回答。测量 false-confidence reduction。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Prompt caching | “缓存系统与上下文” | Claude/OpenAI 特性：命中时缓存的前缀 token 享受 60-90% 折扣 |
| RAGAS | “RAG evaluator” | 自动评估 faithfulness、answer relevance、context precision |
| Golden set | “Labeled eval” | 200+ 条专家标注的带引用问答对；作为 ground truth |
| Jurisdiction tag | “Compliance label” | 附加到文本块的 GDPR/HIPAA/SOC2 范围标识；由 retrieval filter 强制执行 |
| Citation faithfulness | “Grounded answer rate” | 可由检索到的原文片段支撑的主张比例 |
| Drift | “Retrieval quality decay” | nDCG 或 citation score 的周度变化；告警阈值为 5% |
| Red team | “Adversarial eval” | 发布前的 jailbreak、PII extraction、off-domain probes |

## Further Reading

- [Harvey AI](https://www.harvey.ai) — 法律领域生产级架构参考
- [Glean enterprise search](https://www.glean.com) — 企业级 RAG 规模参考
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管式数据摄入
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本控制杠杆参考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 权威的 RAG eval 框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏框架
