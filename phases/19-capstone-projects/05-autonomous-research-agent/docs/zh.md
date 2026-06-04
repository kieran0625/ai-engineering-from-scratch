# Capstone 05 — Autonomous Research Agent (AI-Scientist Class)

> Sakana 的 AI-Scientist-v2 发表了完整论文。Agent Laboratory 运行了相关实验。Allen AI 分享了追踪日志。2026 年的形态是对实验进行计划-执行-验证树搜索、成本约束、沙箱代码执行、视觉反馈 LaTeX 撰写器，以及自动化的 NeurIPS 风格评审委员会集成。本结业项目的目标是构建一个这样的系统，以每篇论文不超过 30 美元的成本端到端运行它，并通过 Sakana 所记录的沙箱逃逸红队测试。

**类型：** 结业项目
**语言：** Python（智能体 + 沙箱）、LaTeX（输出）
**前置要求：** 第 2 阶段（机器学习）、第 3 阶段（深度学习）、第 7 阶段（transformers）、第 10 阶段（从零实现大语言模型）、第 14 阶段（智能体）、第 15 阶段（自主智能体）、第 16 阶段（多智能体）、第 18 阶段（安全）
**涉及阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**耗时：** 40 小时

## Problem

2026 年，自主研究智能体跨过了一个临界点。Sakana AI 的 AI-Scientist-v2 在《自然》杂志上发表，其生成的论文通过了研讨会同行评审。ShinkaEvolve（ICLR 2026）将该方向扩展至假设演化。AMD 的 Agent Laboratory 提供了可复现的追踪日志。这些智能体并非魔法——它们是在候选实验树之上运行的计划-执行-验证循环，配合成本上限、种子绑定的沙箱以及自动化评审。其中的精髓在于循环设计、预算控制与安全机制。

你将在一个狭窄领域内针对一个初始想法（例如，对参数量为 1 亿的 Transformer 进行注意力稀疏性消融实验）实现该循环来学习它。其价值不在于首次运行就发现新事物，而在于基础设施：树搜索、实验沙箱、撰写-评审循环、红队报告。Sakana 团队记录了沙箱逃逸的失败案例；你的智能体必须通过同样的红队测试。

## Concept

该智能体采用最佳优先树搜索。节点为实验规格说明：(假设、配置、代码、预期结果)。扩展步骤会提出带有小幅修改的子节点（更换优化器、调整批次大小、消融某个组件）。每个子节点都在具有严格资源上限的全新沙箱中运行。结果反馈给评分函数，按（新颖性 × 质量 × 剩余预算）对节点进行排序。树持续生长直至预算耗尽，随后对最优分支进行论文撰写。

撰写器是多模态的。它生成 LaTeX 草稿，编译它，渲染图表，并将渲染后的 PDF 反馈给 Claude Opus 4.7 的视觉模式，以评估排版、图表可读性以及论点与证据的一致性。由五个 LLM 裁判组成的评审委员会会给出 NeurIPS 风格的评分（新颖性、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文将带着批评意见返回给撰写器。

安全性是核心支撑。每项实验均在无网络出站流量、限定墙钟时间且固定资源限制的 E2B 或 Daytona 沙箱中运行。智能体的代码生成步骤需经过策略层，该层会拦截试图逃逸沙箱的系统调用。红队报告重现了 Sakana 记录的攻击面（fork 炸弹、文件系统逃逸、LLM 生成的网络调用）。

## Architecture

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Stack

- 编排：LangGraph（含检查点与人工审批门控）
- 树搜索：基于实验节点自定义的最佳优先搜索（源自 Sakana v2 的 AB-MCTS 风格）
- 沙箱：每项实验使用 E2B，Docker-in-Docker 作为回退方案；通过 cgroups 实施资源上限
- 文献检索：Semantic Scholar Graph API + OpenAlex + 摘要本地 FAISS 缓存
- 撰写器：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表评估与排版
- 评审员：5 名裁判集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max）及加权聚合
- 实验框架：PyTorch 2.5 用于实际实验，W&B 用于日志记录
- 可观测性：Langfuse 用于智能体追踪，每篇论文硬性预算 30 美元

## Build It

1. **初始想法与领域范围界定。** 选取一个初始想法（例如，“探究参数量低于 10 亿的 Transformer 注意力图中的稀疏模式”）。定义搜索空间：模型、数据集、计算预算。

2. **文献梳理。** 查询 Semantic Scholar + OpenAlex 获取 50 篇最高引用的相关论文；将摘要缓存至本地；生成一份一页纸的领域综述。

3. **树结构搭建。** 用初始假设初始化根节点。实现 `expand(node) -> children`，包含小幅编辑提议（每个子节点仅更改一项配置）。实现 `score(node)` 作为加权的新颖性 × 质量 × 预算项。

4. **沙箱封装。** 每项实验均运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子文件写入沙箱；输出结果以只读方式挂载回宿主机。

5. **计划-执行-验证循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志与指标。`verify` 对指标运行单元测试（损失是否下降？消融实验是否隔离了效应？）。失败的节点会将失败原因记录在树上。

6. **撰写。** 预算耗尽后，选择最优分支。使用 matplotlib 渲染图表。将分支追踪日志作为上下文，通过 Claude Opus 4.7 生成 LaTeX 草稿。编译。将编译后的 PDF 反馈给 Opus 4.7 视觉模式进行评估。迭代优化。

7. **评审委员会集成。** 五名裁判依据 NeurIPS 风格评分标准对草稿进行打分（新颖性、严谨性、清晰度、可复现性、影响力）。若平均分 < 4.0/5，则带着批评意见返回撰写器。重写超过 3 次则强制停止。

8. **红队测试。** 构建或集成一组针对沙箱的对抗性任务：fork 炸弹、网络数据外泄尝试、文件系统逃逸、LLM 生成的 Shell 元字符。确认全部被拦截。撰写测试报告。

9. **可复现性。** 每篇论文均附带树搜索追踪 JSON、种子文件、W&B 运行链接、沙箱配置，以及一份端到端复现指南（README）。

## Use It

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Ship It

`outputs/skill-ai-scientist.md` 是交付物。给定初始想法、领域和 30 美元预算，它将运行完整流水线并输出一篇可供评审的论文及可复现性包。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 论文质量 | 对照已发表的研讨会论文进行盲评 |
| 20 | 实验严谨性 | 基线、种子、消融实验；每项主张均由结果表中的单元格数据支撑 |
| 20 | 成本与算力纪律 | 严格执行每篇 30 美元上限，通过 Langfuse 追踪 |
| 20 | 安全性 | 通过沙箱红队测试；网络策略与紧急停止开关已验证 |
| 15 | 可复现性 | 使用相同种子一键重跑即可复现论文 |
| **100** | | |

## Exercises

1. 在同一领域内针对三个不同的初始想法运行该流水线。比较树搜索中哪些部分存在重叠。识别重复浪费的计算资源。

2. 为预估成本超过 5 美元的节点，在实验执行前添加一个人工介入门控。测量总成本下降了多少。

3. 将评审委员会集成替换为单一裁判。在预留的已知劣质论文集上测量误接受率。

4. 引入网络外泄红队测试：智能体编写尝试向外部地址 `curl` 的代码。确认 `--network=none` 策略成功拦截。记录该尝试。

5. 将你的树搜索与扁平随机基线（相同预算，无扩展策略）进行比较。报告新颖性 × 质量的增益。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 树搜索 | “AB-MCTS 风格扩展” | 基于新颖性×质量×预算评分的实验节点最佳优先探索 |
| 沙箱 | “实验隔离” | 无网络、CPU/内存受限、种子固定、输入只读的容器 |
| 视觉评估 | “先渲染后阅读” | 将论文编译为 PDF，将 PDF 反馈给 VLM 以评估排版与论点一致性 |
| 评审委员会集成 | “自动化同行评审” | 多名 LLM 裁判依据 NeurIPS 标准对论文打分；加权聚合结果控制流水线流转 |
| 新颖性评分 | “这是否新颖？” | 惩罚与 50 篇文献缓存距离过近的启发式算法 |
| 成本上限 | “$ 预算” | 单篇论文总支出的硬性上限；结合 Langfuse 计数器与预运行估算 |
| 红队测试 | “沙箱逃逸审计” | 若策略配置错误即可逃逸沙箱的对抗性任务 |

## Further Reading

- [Sakana AI-Scientist-v2 仓库](https://github.com/SakanaAI/AI-Scientist-v2) —— 参考级生产环境研究智能体
- [Sakana AI-Scientist-v1 论文 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) —— 原始方法论
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) —— 演化扩展
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) —— 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) —— 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) —— 文献检索
- [E2B 沙箱](https://e2b.dev) —— 参考实验隔离方案
- [NeurIPS 审稿指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) —— 评审委员会集成所采用的评分标准
