# Capstone 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的角色提示、AutoGen 0.4 的类型化 Actor 图、Cognition 的 Devin 以及 Factory 的 Droids 均收敛于同一种 2026 年形态：架构师负责规划，N 名程序员在并行工作树中协作，审查员把关，测试员验证。并行工作树将实际耗时转化为吞吐量。共享状态与交接协议则成为系统的故障面。本次毕业设计旨在组建该团队，在 SWE-bench Pro 上进行评估，并报告哪些交接环节会出错及其频率。

**类型：** Capstone
**语言：** Python / TypeScript（智能体）、Shell（工作树脚本）
**前置要求：** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 15 阶段（自主智能体）、第 16 阶段（多智能体）、第 17 阶段（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P16 · P17
**预计时长：** 40 小时

## Problem

单智能体编码框架在处理大型任务时会遇到瓶颈。并非因为单个智能体能力不足，而是因为 20 万 Token 的上下文无法同时容纳架构规划、四个并行代码库切片、审查员批注以及测试输出。多智能体工厂将问题拆分：架构师负责规划，程序员在并行工作树中独立实现，审查员把关，测试员验证。SWE-AF 的“工厂”架构、MetaGPT 的角色划分、AutoGen 的类型化 Actor 图——这三种表述描述的是同一种系统形态。

系统的故障面在于交接环节。架构师规划了程序员无法实现的内容；程序员产生冲突的代码差异；审查员批准了幻觉修复方案；测试员与仍在编写代码的程序员发生竞态。你将构建这样一个团队，在 50 个 SWE-bench Pro 问题上运行它，追踪每一次交接，并发布事后复盘报告。

## Concept

角色即类型化智能体。**架构师**（Claude Opus 4.7）读取 Issue，编写规划，并将其拆解为带有明确接口的子任务。**程序员**（Claude Sonnet 4.7，N 个并行实例，每个位于 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查员**（GPT-5.4）读取合并后的代码差异，决定批准或要求具体修改。**测试员**（Gemini 2.5 Pro）在隔离环境中运行测试套件，并附带产物报告通过/失败结果。

通信通过共享任务板进行（基于文件或 Redis）。每个角色仅处理其被授权的任务。交接通过符合 A2A 协议类型的消息完成。协调关注点包括：合并冲突解决（由协调员角色或自动三方合并处理）、共享状态同步（一旦程序员开始工作，规划即冻结；重新规划是独立事件）以及审查员门禁机制（审查员不能批准自己提出或编写的更改）。

Token 放大效应是隐藏成本。每个角色边界都会增加摘要提示词和交接上下文。一次 40 轮的单智能体运行，在四个角色间分配后会变成总共 160 轮交互。评分标准特别权衡了 Token 效率与单智能体基线，因为核心问题不是“多智能体是否可行”，而是“它是否能在单位成本下胜出”。

## Architecture

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Stack

- 编排：LangGraph（含共享状态 + 每个智能体的子图）
- 消息传递：A2A 协议（Google 2025），用于智能体间的类型化消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（程序员）、GPT-5.4（审查员）、Gemini 2.5 Pro（测试员）
- 工作树隔离：每位程序员使用 `git worktree add` + Daytona 沙箱
- 合并协调器：自定义三方合并 + LLM 辅助冲突解决
- 评估：SWE-bench Pro（50 个 Issue）、SWE-AF 场景、HumanEval++（单元测试）
- 可观测性：Langfuse（带角色标签的 Span，按智能体统计 Token）
- 部署：K8s（每个角色作为独立的 Deployment + 基于积压任务的 HPA）

## Build It

1. **任务板**。基于文件的 JSONL 格式，包含类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅特定标签。

2. **架构师**。读取 GitHub Issue，使用包含明确子任务接口要求的规划模板运行 Opus 4.7（需指定影响的文件、公开函数、测试影响）。输出一条 `plan_request`，内含子任务的 DAG。

3. **程序员**。N 个并行工作节点，每个从任务板认领一个子任务。每个节点创建一个全新的 `git worktree add` 分支及 Daytona 沙箱。实现该子任务。输出 `diff_ready`，附带补丁与测试差异。

4. **合并协调器**。所有程序员完成后，将 N 个分支三方合并至暂存分支。仅在存在文件级重叠时才启用 LLM 辅助冲突解决。

5. **审查员**。GPT-5.4 读取合并后的代码差异。不能批准自己编写的差异。输出 `approved`（无操作）或 `review_feedback`，并将具体的修改请求路由回对应的程序员。

6. **测试员**。Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获产物。输出 `test_passed` 或 `test_failed`（附带堆栈跟踪）。失败的测试将循环退回给负责该失败子任务的程序员。

7. **交接记账**。每条跨越角色边界的消息都会在 Langfuse 中生成一个 Span，记录负载大小和使用的模型。计算每个子任务的 Token 放大率（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估**。在 50 个 SWE-bench Pro 问题上运行。将 pass@1 和单问题解决成本（$）与单智能体基线（单个 Sonnet 4.7 在单一工作树中）进行比较。

9. **事后复盘**。针对每个失败的 Issue，定位出错的交接环节（规划过于模糊、合并冲突、审查员误通过、测试不稳定）。生成交接失败直方图。

## Use It

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Ship It

`outputs/skill-multi-agent-team.md` 为最终交付物。给定 Issue URL 和并行度，团队将生成一个可直接合并的 PR，并附带按角色划分的 Token 记账数据。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 个 Issue 子集，pass@1 |
| 20 | 并行加速比 | 实际耗时 vs 单智能体基线 |
| 20 | 审查质量 | 注入 Bug 探测中的误通过率 |
| 20 | Token 效率 | 每解决一个 Issue 的总 Token 数 vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## Exercises

1. 在运行中途向代码差异中注入一个明显 Bug（在主逻辑前添加额外的 `return None`）。测量审查员的误通过率。调整审查员提示词，直到误通过率降至 5% 以下。

2. 减少至两名程序员（架构师 + 程序员 + 审查员 + 测试员，程序员顺序执行两个子任务）。比较实际耗时与通过率。

3. 用单写入者约束替换合并协调器（子任务触及不重叠的文件集）。衡量架构师的规划负担。

4. 将审查员从 GPT-5.4 替换为 Claude Opus 4.7。测量误通过率与 Token 成本差值。

5. 增加第五个角色：文档撰写员（Haiku 4.5）。审查通过后，生成变更日志条目。衡量文档质量是否值得额外的 Token 开销。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 并行工作树 (Parallel worktree) | “隔离分支” | `git worktree add`，为每位程序员生成独立的工作树 |
| 任务板 (Task board) | “共享消息总线” | 文件或 Redis 存储的类型化消息，智能体据此订阅 |
| 交接 (Handoff) | “角色边界” | 任何从一个角色上下文跨越到另一个角色的消息 |
| Token 放大效应 (Token amplification) | “多智能体开销” | 所有角色的总 Token 数 / 同一任务的单智能体 Token 数 |
| A2A 协议 (A2A protocol) | “智能体对智能体” | Google 2025 年发布的类型化智能体间消息规范 |
| 合并协调器 (Merge coordinator) | “集成器” | 执行三方合并并调解冲突的组件 |
| 误通过 (False approval) | “审查员幻觉” | 审查员批准了已知包含 Bug 的代码差异 |

## Further Reading

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) —— 2026 年多智能体工厂的参考实现
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) —— 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) —— 微软的类型化 Actor 框架
- [Cognition AI (Devin)](https://cognition.ai) —— 参考产品
- [Factory Droids](https://www.factory.ai) —— 替代参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) —— 智能体间消息传递规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) —— 隔离底层支撑
- [SWE-bench Pro](https://www.swebench.com) —— 评估目标
