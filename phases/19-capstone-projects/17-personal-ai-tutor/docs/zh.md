# 毕业设计 17 —— 个人 AI 辅导系统（自适应、多模态、带记忆）

> 2026年，Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 均已大规模推出自适应多模态辅导功能。其共同架构包括：苏格拉底式教学策略（绝不直接抛出答案）、每次交互后更新的学员模型（类似贝叶斯知识追踪风格）、语音+文本+拍照数学输入、课程图谱检索、间隔重复调度，以及针对适龄内容的严格安全过滤。本毕业设计的任务是交付一个特定学科的辅导系统（K-12代数或入门级Python），开展为期两周、包含10名学习者的有效性研究，并通过内容安全审计。

**类型：** 毕业设计
**语言：** Python（后端、学员模型）、TypeScript（Web 应用）、SQL（通过 Postgres + Neo4j 构建的课程图谱）
**前置要求：** 第5阶段（NLP）、第6阶段（语音）、第11阶段（大语言模型工程）、第12阶段（多模态）、第14阶段（智能体）、第17阶段（基础设施）、第18阶段（安全）
**涉及阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**耗时：** 30小时

## 问题背景

自适应辅导曾是教育科技领域的研究细分方向。到了2026年，它已成为面向消费者的成熟产品。Khanmigo 已部署至美国大多数学区。Duolingo Max 的月活跃用户数达到数千万。Google 的 LearnLM / Gemini for Education 为 Google Classroom 提供辅导支持。Quizlet Q-Chat 与闪卡功能并列。Synthesis Tutor 凭借“好奇儿童专属辅导”功能实现病毒式传播。这些产品的共同要素包括：多模态输入（打字、语音、拍摄公式）、苏格拉底教学法（先提问，后解释）、每次交互后更新的学员模型，以及严格的适龄安全机制。

你将针对特定学习者群体构建其中一种系统。衡量标准是一项真实的有效性研究：在10名学习者中，进行为期两周的前测与后测。语音交互必须自然流畅（复用毕业设计03的子栈）。记忆模块必须尊重隐私。安全过滤器必须通过针对 K-12 阶段的 COPPA 合规红队测试。

## 核心概念

包含四个核心组件。**辅导策略**采用苏格拉底循环：当学习者询问答案时，策略会提出引导性问题；答对时，进入下一个概念；卡壳时，提供分步提示。**学员模型**采用贝叶斯知识追踪（或其简化变体），在每次交互后更新每个课程节点的能力掌握概率。**课程图谱**是一个 Neo4j 数据库，包含概念节点与先修关系边；策略沿图谱遍历以选择下一个概念。**记忆**采用情景+语义存储（类 agentmemory 架构），保存历史交互、错误记录和学习偏好。

用户体验支持多模态。文本输入用于键入答案。语音输入通过 LiveKit + Whisper 实现（复用毕业设计03）。数学题拍照输入通过 dots.ocr 或 PaliGemma 2 实现。语音输出通过 Cartesia Sonic-2 实现。安全机制使用 Llama Guard 4 加上适龄过滤器（拦截成人内容、暴力、自残内容），并配合符合 COPPA 要求的记忆保留策略。

有效性研究是最终交付物。招募10名学习者，进行前测与后测，周期为两周。报告学习增益差值及置信区间。并与非自适应基线组（相同内容按线性顺序呈现，无辅导策略）进行对比。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12 代数或入门级 Python（任选其一以保证深度）
- 辅导策略：基于 Claude Sonnet 4.7 的 LangGraph（启用提示词缓存）
- 学员模型：经典贝叶斯知识追踪（BKT）或用于间隔调度的 FSRS
- 课程图谱：Neo4j 存储的概念节点 + 先修关系边 + OER（开放教育资源）内容
- 记忆：类 agentmemory 架构的持久化向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用毕业设计03子栈）
- 拍照数学：dots.ocr 或 PaliGemma 2 用于公式识别
- 安全：Llama Guard 4 + 自定义适龄过滤器
- 评估：布鲁姆分类学层级题目生成、前后测框架、有效性研究工具链

## 开发步骤

1. **课程图谱。** 构建包含 50-150 个概念节点的 Neo4j 数据库（例如 K-12 代数从“数轴”到“求根公式”），并添加先修关系边。为每个节点附加 OER 内容（Open Textbook、OpenStax）。

2. **学员模型。** 初始化贝叶斯知识追踪，设置先验参数：猜测率、失误率、学习率。在每次交互后更新各概念的能力掌握概率。按学习者持久化存储。

3. **辅导策略。** 使用 LangGraph 构建节点：`read_signal`（判断学习者答案是否正确/部分正确/卡壳）、`select_concept`（遍历课程图谱，选择优先级最高的概念）、`scaffold`（苏格拉底式提问）、`update_mastery`。

4. **记忆。** 每次交互写入情景存储。错误记录和学习偏好晋升至语义记忆。实施符合 COPPA 的保留策略：1年后自动删除，家长可访问管理。

5. **语音链路。** LiveKit Agents 工作节点绑定至辅导策略。ASR 使用 Whisper-v3-turbo。TTS 使用 Cartesia Sonic-2。支持打断功能（复用毕业设计03的交互机制）。

6. **拍照数学链路。** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2 识别公式；将结构化结果输入辅导系统。

7. **安全。** 所有模型输出均需经过 Llama Guard 4 和适龄过滤器检查（拦截自残、成人内容、暴力内容）。记忆访问按学习者 ID 隔离；提供家长端界面用于数据删除。

8. **有效性研究。** 招募10名学习者，进行前测（标准化30题基线），进行为期两周的辅导交互（每周3次会话），最后进行后测。与10名学习者的非自适应基线组（相同内容）进行对比。

9. **每周进度报告。** 为每位学习者自动生成 PDF 总结，包含探索过的主题、能力掌握轨迹及推荐下一步学习内容。

## 使用指南

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付说明

`outputs/skill-ai-tutor.md` 是最终交付物。一个特定学科的自适应辅导系统，具备多模态输入、学员模型、记忆、安全机制，并提供可量化的有效性验证。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 学习增益差值 | 10人两周研究的前后测分数差值 |
| 20 | 苏格拉底策略遵循度 | 对话转录样本的量表评分 |
| 20 | 多模态体验 | 端到端的语音+照片+文本连贯性 |
| 20 | 安全与隐私合规 | Llama Guard 4 通过率 + 符合 COPPA 的记忆保留策略 |
| 15 | 课程广度与图谱质量 | 概念覆盖率 + 先修图谱一致性 |
| **100** | | |

## 练习

1. 分别在有和没有自适应学员模型（随机概念顺序）的情况下运行有效性研究。报告差值。预期自适应模型表现更好，但具体差值大小才是关键指标。

2. 增加多模态探针测试：用文本、语音和照片三种形式呈现同一概念问题。测量学习者是否在他们偏好的模态下收敛得更快。

3. 构建家长仪表盘：展示练习过的主题、能力掌握轨迹、即将学习的概念、安全事件（任何护栏触发记录）。需符合 COPPA 规范。

4. 增加语言切换模式：辅导系统接受西班牙语输入并用西班牙语授课。测量 X-Guard 的覆盖范围。

5. 压力测试记忆隐私：验证即使通过语音片段重新注入攻击，学习者A也无法查看学习者B的数据。记录尝试访问的行为并发出警报。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 苏格拉底策略 | “只提问，不直给” | 辅导系统提出引导性问题，而非直接给出答案 |
| 贝叶斯知识追踪 | “BKT” | 用于计算每个概念掌握概率的经典学员模型方程 |
| FSRS | “自由间隔重复调度器” | 2024年推出的间隔重复调度算法，性能优于 SM-2 |
| 课程图谱 | “概念有向无环图” | 包含概念节点与先修关系边的 Neo4j 数据库 |
| 情景记忆 | “单次交互日志” | 存储每次交互记录以便后续检索 |
| 语义记忆 | “模式学习库” | 从情景记忆中提炼压缩的错误记录与偏好 |
| COPPA | “儿童隐私法” | 美国限制收集13岁以下儿童数据的法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) —— 参考级消费端 K-12 辅导产品
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) —— 参考级语言学习辅导产品
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) —— 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) —— 替代参考方案
- [Synthesis Tutor](https://www.synthesis.com) —— 初创公司参考案例
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) —— 间隔重复调度算法
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) —— 学员模型经典文献
- [LiveKit Agents](https://github.com/livekit/agents) —— 语音技术栈
