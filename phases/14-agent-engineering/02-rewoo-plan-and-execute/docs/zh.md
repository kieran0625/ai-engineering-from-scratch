# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 将思考与行动交织在单一流程中。ReWOO 将它们分离：先制定一个完整的计划，然后执行。Token 消耗减少 5 倍，在 HotpotQA 上准确率提升 4%，且可以将规划器蒸馏至 7B 模型。Plan-and-Execute 将其泛化为一种模式；Plan-and-Act 则将其扩展至网页导航场景。

**类型：** 构建
**语言：** Python（标准库）
**前置条件：** Phase 14 · 01（Agent Loop）
**耗时：** 约 60 分钟

## 学习目标

- 解释为何 ReWOO 的 Planner / Worker / Solver 拆分比 ReAct 的交织循环更节省 Token 且更具鲁棒性。
- 实现一个计划 DAG、依赖顺序执行器以及组合 Worker 输出的 Solver —— 全部仅使用标准库。
- 根据 2026 年“五种工作流模式”框架（Anthropic），判断任务应采用“先规划后执行”还是 ReAct 交织模式。
- 识别何时需要 Plan-and-Act 的合成计划数据来处理长周期网页或移动端任务。

## 问题所在

ReAct 的交织式“思考-行动-观察”循环简单灵活，但每次工具调用都必须携带完整的历史上下文——包括之前所有的思考内容。Token 消耗随深度呈二次方增长。更糟的是：当循环中途某个工具调用失败时，模型必须从错误观察结果中重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023 年 5 月）注意到了这一点并做出了尝试：预先规划整体流程，并行获取证据，最后组合答案。一次 LLM 调用用于规划，N 次工具调用用于获取证据（可并行），一次 LLM 调用用于求解。这种权衡牺牲了灵活性（计划是静态的），换来了更高的 Token 效率和更清晰的故障模式。

## 核心概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点指定一个工具、其参数以及依赖的前置节点（引用如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 将所有结果拼接起来。

### 为何能减少 5 倍 Token

ReAct 的提示词长度随步骤数线性增长。在第 10 步时，提示词包含思考 1、行动 1、观察 1、思考 2、行动 2、观察 2，依此类推。每个中间步骤还会冗余地包含原始提示词。

ReWOO 只需支付一次 Planner 提示词（较大）、N 个小型 Worker 提示词（仅包含工具调用，无链条）和一次 Solver 提示词。论文在 HotpotQA 上的测量结果显示，Token 消耗减少约 5 倍，同时绝对准确率提升 4%。

### 为何更具鲁棒性

若在 ReAct 中 Worker 3 失败，循环必须在中途推理如何从错误中恢复。而在 ReWOO 中，Worker 3 返回错误字符串；Solver 会结合原始计划的上下文看到该错误，从而优雅降级。故障定位是按节点而非按步骤进行的。

### 规划器蒸馏

论文的第二个结论：由于 Planner 不接触观察结果，你可以基于 175B 教师模型的 Planner 输出来微调一个 7B 模型。小模型负责规划，推理时无需大模型。这现已成为标准做法——许多 2026 年的生产级 Agent 采用小规划器搭配大执行器，或反之。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年 8 月的文章将 ReWOO 泛化为一种模式名称：Plan-and-Execute。前置规划器输出步骤列表，执行器运行每个步骤，可选的重规划器可在观察结果后进行修订。这比 ReWOO 更接近 ReAct（重规划器将观察结果重新引入规划），但保留了 Token 节省的优势。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展至长周期网页和移动端 Agent。其核心贡献在于合成计划数据：带标签的轨迹生成器产出计划显式的训练数据。用于微调规划器模型，使其在类似 WebArena 的任务中能稳定运行超过 30–50 步，而单个 ReAct 轨迹在此类任务中会丧失连贯性。

### 如何选择

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、未知环境、需要响应式异常处理 |
| ReWOO | 结构清晰的任务、已知工具、对 Token 敏感、证据可并行获取 |
| Plan-and-Execute | 类似 ReWOO，但在部分执行后可进行重规划 |
| Plan-and-Act | 长周期任务（>30 步）、网页/移动端/计算机操作 |
| Tree of Thoughts | 值得为搜索付出代价时（见 Lesson 04） |

Anthropic 2024 年 12 月的建议：从最简单的方案开始。如果任务仅需一次工具调用加总结，不要构建 ReWOO。如果任务是 40 步的研究作业，不要单独使用 ReAct。

## 动手实现

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner` —— 基于提示词生成计划 DAG 的脚本策略。
- `Worker` —— 通过注册表分发每个节点的工具调用。
- `Solver` —— 读取证据并生成最终答案的脚本组合逻辑。
- 依赖解析 —— 引用占位符（如 `#E1`）会在分派时被替换为前置 Worker 的输出。

演示示例回答的问题是：“法国首都的人口是多少（四舍五入到百万位）？”它采用两步计划：(1) 查询首都，(2) 查询人口，然后求解。

运行方式：

```
python3 code/main.py
```

追踪日志首先显示完整计划，接着是 Worker 结果，最后是 Solver 组合过程。将 Token 计数（我们打印了粗略的字符数）与 ReAct 风格的交织运行进行比较——在这类结构化任务上，ReWOO 表现更优。

## 实际应用

LangGraph 将 Plan-and-Execute 作为内置配方提供（ReAct 对应 `create_react_agent`，计划执行对应自定义图）。CrewAI 的 Flows 直接编码了该模式：你预先定义任务，Flow DAG 负责执行它们。Plan-and-Act 的合成数据方法目前仍主要处于研究阶段；而运行时模式（显式计划 DAG）已通过 LangGraph 和 CrewAI Flows 投入生产环境。

## 部署上线

`outputs/skill-rewoo-planner.md` 根据用户请求和工具目录生成 ReWOO 计划 DAG。它在移交执行器之前会验证计划（确保无环、所有引用已解析、所有工具均存在）。

## 练习

1. 为独立的计划节点并行化 Worker 执行。在一个包含 2 个并行组的 6 节点 DAG 上，它能带来什么收益？
2. 添加一个重规划节点，当任何 Worker 返回错误时触发。对 ReWOO 做最小改动以演变为 Plan-and-Execute 是什么？
3. 用小型模型（7B 级别）替换 `Planner`，并将 `Solver` 保留在前沿模型上。对比端到端质量——拆分点在哪里会失效？
4. 阅读 ReWOO 论文第 4 节关于规划器蒸馏的内容。从概念上复现 175B -> 7B 的结果：你需要哪些训练数据，以及如何评估计划质量？
5. 将玩具项目移植到 Plan-and-Act 的轨迹格式：计划是序列而非 DAG。权衡取舍会发生什么变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | “无观察推理” | 先规划，再并行获取证据，最后求解——规划提示词中不包含观察结果 |
| Plan-and-Execute | “LangChain 的计划-执行模式” | ReWOO 加上执行后的可选重规划节点 |
| Plan-and-Act | “扩展版计划-执行” | 显式的规划器/执行器拆分，配合长周期任务的合成计划训练数据 |
| Evidence reference | “#E1, #E2, ...” | 计划节点占位符，分派时替换为前置 Worker 的输出 |
| Planner distillation | “小规划器，大执行器” | 基于大教师模型的规划轨迹微调小模型 |
| Token efficiency | “更少往返” | 论文中 HotpotQA 相比 ReAct 减少 5 倍 Token |
| DAG executor | “拓扑调度器” | 按依赖顺序执行计划节点；每层内并行 |

## 延伸阅读

- [Xu 等人，ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) —— 奠基性论文
- [Erdogan 等人，Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) —— 配合合成计划的扩展版规划器-执行器
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview) —— 框架配方
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 选择最简单且有效的模式
