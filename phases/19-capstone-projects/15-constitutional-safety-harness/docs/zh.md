# 综合项目 15 —— 宪法安全护栏 + 红队测试范围

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety 以及用于多语言覆盖的 X-Guard，共同定义了 2026 年的安全分类器技术栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 已成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们整合到生产流水线中。本综合项目将这一切串联起来：围绕目标应用构建分层安全护栏，运行包含 6 种以上攻击家族的自主红队智能体，并执行宪法式自我批判流程，以产出可衡量的无害性提升幅度（delta）。

**类型：** 综合项目
**语言：** Python（安全流水线、红队测试）、YAML（策略配置）
**前置要求：** 第 10 阶段（从零构建 LLM）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 18 阶段（伦理、安全与对齐）
**涉及阶段：** P10 · P11 · P13 · P14 · P18
**预计耗时：** 25 小时

## 问题描述

2026 年 LLM 安全的前沿问题不在于分类器是否有效（它们大致是有效的），而在于如何围绕生产级应用正确地组合它们，同时避免过度拒绝或留下明显的漏洞。Llama Guard 4 处理英文政策违规。X-Guard（支持 132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的提示注入。NVIDIA Nemotron 3 Content Safety 覆盖企业级类别。Anthropic 的 Constitutional Classifiers 是一种不同的方法，主要用于训练阶段而非推理服务阶段。

攻击技术的演进同样重要。PAIR 和 TAP 自动化越狱发现过程。GCG 运行基于梯度的后缀攻击。多轮对话和代码切换攻击利用智能体的记忆能力。任何部署的 LLM 都需要一套红队测试范围——garak 和 PyRIT 是其中的标准驱动工具——以及文档化的缓解措施和基于 CVSS 评分的发现报告。

你将加固一个目标应用（可以是 8B 指令微调模型，也可以是其他综合项目中开发的 RAG 聊天机器人之一），对其运行 6 种以上的攻击家族，并生成攻击前后的无害性对比测量结果。

## 核心概念

安全流水线包含五个层级。**输入清洗**：剥离零宽字符、解码 base64/rot13、规范化 Unicode。**策略层**：NeMo Guardrails v0.12 规则（偏离主题、毒性、PII 提取）。**分类器网关**：输入端使用 Llama Guard 4，非英语内容使用 X-Guard，图像输入使用 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：输出端使用 Llama Guard 4、Presidio 进行 PII 清理、在适用时强制执行引用规范。**人工介入层（HITL）**：标记为高风险的输出将进入 Slack 队列。

红队测试范围由调度器驱动运行。PAIR 和 TAP 自主发现越狱攻击。GCG 运行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（角色伪装、记忆利用）。代码切换攻击（混合英语与斯瓦希里语或泰语）。每次运行都会生成一份结构化的发现报告文件，包含 CVSS 评分和披露时间线。

宪法式自我批判流程是一种训练阶段的干预手段。收集 1000 条有害尝试提示词，让模型起草回复，对照书面宪法（不伤害规则）对其进行批判性评估，并在批判循环中进行重新训练。在预留的评估集上测量攻击前后的无害性提升幅度。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动工具：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao 等，2023）、Tree-of-Attacks（TAP）、GCG 后缀
- 宪法训练：Anthropic 风格的自我批判循环 + 基于批判数据的 SFT
- PII 清理：Presidio
- 目标应用：8B 指令微调模型或其他综合项目中的 RAG 聊天机器人之一

## 构建步骤

1. **目标环境搭建**。在 vLLM 上部署一个 8B 指令微调模型（或复用其他综合项目中的 RAG 聊天机器人）。这是待测应用。

2. **安全流水线封装**。将五层流水线集成到目标应用周围。验证每一层均可独立观测（在 Langfuse 中为每层分配独立的 span）。

3. **分类器覆盖**。加载 Llama Guard 4、X-Guard（多语言版）、ShieldGemma-2（图像版）。在小型标注数据集上分别运行以建立基线。

4. **红队调度器**。调度 garak、PyRIT、PAIR 智能体、TAP 智能体、GCG 运行器、多轮攻击器和代码切换攻击器。每个任务在独立的队列中运行。

5. **攻击套件**。六种攻击家族：(1) PAIR 自动化越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色伪装，(6) 多语言代码切换。按家族汇报成功率。

6. **宪法式自我批判**。整理 1000 条有害尝试提示词。针对每条提示词，目标模型起草回复。批评者 LLM 依据书面宪法（“不造成伤害”、“引用证据”、“拒绝非法请求”）进行打分。被批评者提出异议的提示词将被重写；目标模型基于经批判改进后的配对数据进行微调。在预留评估集上测量攻击前后的无害性变化。

7. **过度拒绝测量**。在良性提示词套件（如 XSTest）上跟踪误报率。目标模型必须在面对良性问题时保持有用性。

8. **CVSS 评分**。对每一次成功的越狱攻击，按照 CVSS 4.0 标准（攻击向量、复杂度、影响程度）进行评分。生成披露时间线和缓解计划。

9. **测试范围自动化**。上述所有流程均通过 cron 定时任务运行；发现报告写入队列；过度拒绝回归告警将发送至 Slack。

## 使用方式

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付说明

`outputs/skill-safety-harness.md` 是最终交付物。包含生产级分层安全流水线、可复现的红队测试范围，以及攻击前后的无害性对比数据。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 运行 6+ 种攻击家族，覆盖 2+ 种语言 |
| 20 | 真阳性/假阳性权衡 | 攻击拦截率 vs XSTest 良性通过率 |
| 20 | 自我批判提升幅度 | 预留评估集上的前后无害性对比 |
| 20 | 文档与披露 | 附带时间线的 CVSS 评分发现报告 |
| 15 | 自动化与可重复性 | 全部流程通过 cron 运行并触发告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的提示注入插件，对比开启与关闭输出过滤层时的攻击成功率。

2. 增加第七种攻击家族：通过检索文档进行的间接提示注入。测量所需额外防御措施的强度。

3. 实现“拒绝但提供协助”模式：当护栏拦截请求时，目标模型提供相关的安全替代答案，而非直接拒绝。测量其对 XSTest 指标的影响。

4. 多语言覆盖缺口：找出 X-Guard 表现不佳的语言。提出针对性的微调数据集方案。

5. 在 30B 模型上运行宪法式自我批判流程，并测量该提升幅度是否具有可扩展性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 分层安全 | “纵深防御” | 输入、网关、输出及 HITL 环节的多重护栏 |
| Llama Guard 4 | “Meta 的安全分类器” | 2026 年参考级的输入/输出内容分类器 |
| PAIR | “越狱智能体” | 关于 LLM 驱动越狱发现的论文（Chao 等） |
| TAP | “攻击树” | PAIR 的树搜索变体 |
| GCG | “贪婪坐标梯度” | 基于梯度的对抗性后缀攻击 |
| 宪法式自我批判 | “Anthropic 风格训练” | 目标起草 -> 批评者打分 -> 重写 -> 重新训练 |
| XSTest | “良性探测集” | 用于检测过度拒绝回归的基准测试 |
| CVSS 4.0 | “严重性评分” | 安全发现的标准漏洞评分体系 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练阶段参考文档
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像与多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业级参考实现
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 支持 132 种语言的多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队测试工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队测试框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 规则护栏框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
