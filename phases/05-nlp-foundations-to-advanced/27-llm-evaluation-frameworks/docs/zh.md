# LLM 评估 — RAGAS、DeepEval、G-Eval

> 精确匹配和 F1 无法捕捉语义等价性。人工审核无法扩展。LLM-as-judge 是生产环境的答案——经过充分校准后，你可以信任这个数字。

**类型：** Build
**语言：** Python
**前置知识：** Phase 5 · 13（问答系统），Phase 5 · 14（信息检索）
**时间：** ~75 分钟

## 问题所在

你的 RAG 系统回答："June 29th, 2007."
黄金参考是："June 29, 2007."
精确匹配得分 0。F1 得分约 75%。人类会打 100%。

现在乘以 10,000 个测试用例。再乘以检索器、分块、提示词或模型的每次变更。你需要一个能理解含义、能低成本大规模运行、不会谎报回归、并能暴露正确失败模式的评估器。

2026 年有三个框架主导了这个领域。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四项 RAG 指标（忠实度、答案相关性、上下文精确率、上下文召回率），采用 NLI + LLM-judge 后端。有研究支撑，轻量级。
- **DeepEval。** 面向 LLM 的 Pytest。G-Eval、任务完成度、幻觉、偏见指标。原生支持 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：基于思维链的 LLM-as-judge，自定义标准，0-1 分。

三者都依赖 LLM-as-judge。本节课建立对该方法及其信任层机制的直觉。

## 核心概念

![四个评估维度，LLM-as-judge 架构](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用 LLM 替代静态指标，根据评分标准对输出打分。给定 `(query, context, answer)`，向 judge LLM 提问："按忠实度 0-1 分评分。"返回分数。

为什么有效：LLM 以极低成本近似人类判断。GPT-4o-mini 每个评分案例约 $0.003，1000 样本的回归评估运行成本低于 $5。

为什么静默失效：

1. **Judge 偏见。** Judge 偏好更长的答案、偏好自己模型家族生成的答案、偏好与提示词风格匹配的回答。
2. **JSON 解析失败。** 错误 JSON → NaN 分数 → 静默从聚合中排除。RAGAS 用户深知此痛。用 try/except 加显式失败模式来防护。
3. **模型版本漂移。** 升级 judge 会改变所有指标。冻结 judge 模型 + 版本。

**RAG 四项指标。**

| 指标 | 问题 | 后端 |
|------|------|------|
| Faithfulness（忠实度） | 答案中的每个 claim 是否都来自检索到的上下文？ | 基于 NLI 的蕴含关系 |
| Answer relevance（答案相关性） | 答案是否回答了问题？ | 从答案生成假设问题；与真实问题比较 |
| Context precision（上下文精确率） | 检索到的 chunk 中，有多少比例是相关的？ | LLM-judge |
| Context recall（上下文召回率） | 检索是否返回了所有需要的内容？ | LLM-judge 对比黄金答案 |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？"框架自动扩展为思维链评估步骤，然后 0-1 分评分。适用于 RAGAS 未覆盖的特定领域质量维度。

**校准。** 在与人工标签建立相关性之前，永远不要信任原始 judge 分数。运行 100 个手工标注样本。绘制 judge vs 人类散点图。计算 Spearman rho。若 rho < 0.7，你的 judge 评分标准需要改进。

## 动手实现

### 步骤 1：基于 NLI 的忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子 claim。对每个 claim 与检索到的上下文进行 NLI 检验。忠实度 = 被支持 claim 的比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与提问不同，相关性下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤即评分标准。显式步骤比隐式的"0-1 分评分"提示更稳定。

### 步骤 4：CI 门禁

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件提交。每个 PR 运行。阻断回归。

### 步骤 5：从零构建简易评估

参见 `code/main.py`。仅使用标准库的忠实度近似（答案 claim 与上下文的重叠度）和相关性近似（答案 token 与问题 token 的重叠度）。非生产级。展示原理。

## 常见陷阱

- **缺乏校准。** 与人工标签相关性仅 0.3 的 judge 只是噪声。上线前必须进行校准运行。
- **自评估。** 用同一 LLM 生成和评判会虚高 10-20% 分数。Judge 使用不同模型家族。
- **成对评判中的位置偏见。** Judge 偏好第一个呈现的选项。始终随机化顺序并双向运行。
- **原始聚合掩盖失败。** 平均分 0.85 往往掩盖 5% 的灾难性失败。始终检查底部分位数。
- **黄金数据集腐化。** 未版本化的评估集随时间漂移，破坏纵向比较。每次变更都给数据集打标签。
- **LLM 成本。** 大规模时，judge 调用占主导成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 实际应用

2026 年技术栈：

| 使用场景 | 框架 |
|---------|------|
| RAG 质量监控 | RAGAS（4 项指标） |
| CI/CD 回归门禁 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 内的 G-Eval |
| 在线实时流量监控 | RAGAS 参考无关模式 |
| 人机协同抽查 | LangSmith 或 Phoenix 带标注 UI |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型组合：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新颖维度。三者都运行；它们的有用分歧本身就是信号。

## 交付物

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在  个已知存在幻觉的 RAG 示例上使用 RAGAS。验证忠实度指标能捕获每一个。
2. **中等。** 对 50 个 QA 答案按正确性手工标注 0-1 分。用 G-Eval 评分。测量 judge 与人类之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门禁。故意让检索器回归。验证门禁失败。对最低 10% 添加底部分位数阈值告警。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| LLM-as-judge | 用 LLM 评分 | 向 judge 模型提问，按评分标准给输出打 0-1 分。 |
| RAGAS | RAG 指标库 | 开源评估框架，含 4 项参考无关的 RAG 指标。 |
| Faithfulness（忠实度） | 答案有依据吗？ | 答案 claim 中被检索上下文蕴含的比例。 |
| Context precision（上下文精确率） | 检索的 chunk 相关吗？ | top-K chunk 中真正起作用的占比。 |
| Context recall（上下文召回率） | 检索找全了吗？ | 黄金答案 claim 中被检索 chunk 支持的比例。 |
| G-Eval | 自定义 LLM judge | 评分标准 + 思维链评估步骤 + 0-1 分。 |
| Calibration（校准） | 信任但验证 | judge 分数与人类分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — 偏见、校准、局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 统一框架，集成 RAGAS、DeepEval、Phoenix。
