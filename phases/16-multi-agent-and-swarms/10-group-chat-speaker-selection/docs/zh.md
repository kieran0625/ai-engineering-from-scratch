# 群聊与发言者选择

> AutoGen 的 GroupChat 和 AG2 的 GroupChat 在 N 个智能体之间共享一次对话；一个选择器函数（基于 LLM、轮询或自定义）决定下一个发言者。这是涌现式多智能体对话的原型——智能体并不知晓自己在静态图中的角色，它们仅对共享的消息池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中被保留下来；AutoGen v0.4 将其重写为事件驱动的 Actor 模型。微软于 2026 年 2 月将 AutoGen 转入维护模式，并将其与 Semantic Kernel 合并至 Microsoft Agent Framework（2026 年 2 月发布 RC 版）。GroupChat 这一基础原语在 AG2 和 Microsoft Agent Framework 中均得以保留——学一次，处处可用。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置知识：** 第 16 阶段 · 04（基础原语模型）
**耗时：** 约 60 分钟

## 问题

当工作流已知时，静态图（如 LangGraph）非常适用。但真实对话并非静态：有时程序员会询问审查员，有时是研究员，有时是撰稿人。硬编码所有可能的交接会导致边数爆炸。你需要的是*智能体对共享池做出反应*，并由某个函数决定下一个发言者。

这正是 AutoGen GroupChat 所做的。

## 概念

### 结构形态

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个智能体都能看到所有消息。每一轮都会调用选择器函数来挑选下一个发言者。

### 三种选择器模式

**轮询（Round-robin）。** 固定循环。确定性高。随 N 线性扩展，但忽略上下文——即使主题是法律审查，轮到程序员时依然会发言。

**LLM 选择。** 调用 LLM 读取最近的共享池，并返回最佳下一位发言者。具备上下文感知能力，但较慢：每轮都会增加一次 LLM 调用。这是 AutoGen 的默认方式。

**自定义。** 包含任意逻辑的 Python 函数。典型用法：LLM 选择配合回退规则（例如，“程序员发言后，总是将回合交给验证者”）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个智能体完成一轮发言后，管理器会调用选择器，后者返回下一个智能体。循环将持续直到满足终止条件。

### 终止机制

三种常见模式：

- **最大轮数。** 对总回合数设置硬性上限。
- **“TERMINATE”令牌。** 智能体可以发送哨兵消息；管理器检测到该消息后即停止。
- **目标达成检查。** 轻量级验证者在每轮运行，完成后停止聊天。

### AutoGen 与 AG2 的分叉及 Microsoft Agent Framework 的合并

2025 年初，微软围绕事件驱动的 Actor 模型开始对 AutoGen（v0.4）进行大规模重写。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，微软宣布 AutoGen 将进入维护模式，其事件驱动 Actor 模型将合并至 **Microsoft Agent Framework**（2026 年 2 月发布 RC 版，现已与 Semantic Kernel 合并）。GroupChat 概念在这两条技术路线中均得以保留，但实现细节有所不同。对于 v0.2 兼容代码，AG2 是首选的上游项目。

### GroupChat 适用场景

- **涌现式对话。** 你不想预先硬编码所有可能的下一位发言者。
- **角色混合任务。** 程序员询问研究员，研究员询问档案员，档案员再问回程序员。流程不是有向无环图（DAG）。
- **探索性解决问题。** 想想“头脑风暴会议”，而不是“流水线”。

### GroupChat 失效场景

- **严格确定性要求。** LLM 选择器可能不一致。相同的提示词，不同运行结果可能产生不同的下一位发言者。
- **阿谀奉承级联。** 智能体会倾向于服从发言最自信的人。需通过显式提示词进行对抗。
- **上下文膨胀。** 每个智能体都阅读所有消息；经过 10 轮后上下文会变得极其庞大。使用投影（见第 15 课）来限制视图范围。
- **高频发言者。** 一个智能体主导了对话，因为选择器偏向其擅长领域。可将发言者平衡作为选择器的特性引入。

### 群聊与主管模式对比

使用相同的基础原语，但默认行为不同：

- 主管模式：一个智能体负责规划，其他智能体执行。选择器逻辑为“询问规划者下一步该做什么”。
- 群聊模式：所有智能体均为对等节点；选择器是基于共享池的函数。

两者均使用第 04 课中的四个基础原语。群聊默认采用 LLM 选择编排和全池共享状态。

## 构建实现

`code/main.py` 使用标准库从零实现了一个 GroupChat。包含三个智能体（程序员、审查员、管理器），提供轮询和 LLM 选择两种变体，并在检测到 `TERMINATE` 令牌时终止。

演示程序会打印对话记录，以及两种变体下选择器的决策追踪日志。

运行：

```
python3 code/main.py
```

## 使用指南

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——支持轮询、LLM 选择或自定义，并指定选择器使用的输入（最近消息、智能体专长、回合计数）。

## 交付清单

检查清单：

- **最大轮数上限。** 必须设置。典型任务建议 10-20 轮。
- **发言者平衡指标。** 跟踪每个智能体的回合数；当失衡超过阈值时发出警报。
- **终止令牌。** 使用 `TERMINATE` 或专用的验证者智能体。
- **投影或受限内存。** 在约 10 条消息后，考虑仅向每个智能体提供受限视图，以防止上下文膨胀。
- **选择器日志。** 对于 LLM 选择变体，需同时记录选择器的输入和输出选择。否则无法调试。

## 练习

1. 运行 `code/main.py`。对比轮询与 LLM 选择下的对话。哪种模式下哪个智能体占据主导？
2. 在选择器中添加“每个智能体最大发言次数”规则。它对对话记录有何影响？
3. 实现目标达成终止机制：当审查员返回“approved”时停止。在达到轮数上限前，它触发的频率如何？
4. 阅读 AutoGen 稳定版关于 GroupChat 的文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），并将其 v0.2 GroupChat 与 v0.4 事件驱动版本进行比较。v0.4 增加了哪些具体特性（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| GroupChat | “在一个聊天室里的智能体” | 共享消息池 + 选择器函数。AutoGen / AG2 的基础原语。 |
| Speaker selection | “谁接下来发言” | 挑选下一位智能体的函数。可以是轮询、LLM 选择或自定义。 |
| GroupChatManager | “会议主持人” | 拥有选择器并循环处理回合的 AutoGen 组件。 |
| ConversableAgent | “基础智能体” | AutoGen 基类；能够发送和接收消息的智能体。 |
| Termination token | “‘停止’指令” | 结束聊天的哨兵字符串（通常为 `TERMINATE`）。 |
| Hot speaker | “一个智能体占主导” | 选择器反复挑选同一智能体的故障模式。 |
| Context bloat | “池子无限增长” | 每个智能体都阅读所有历史消息；上下文随回合数增长。 |
| Projection | “受限视图” | 针对特定角色的共享池视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) —— 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) —— 社区维护的 AutoGen v0.2 延续版本
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) —— 合并后的继任者，2026 年 2 月 RC 版
- [AutoGen v0.4 发行说明](https://microsoft.github.io/autogen/stable/) —— 事件驱动 Actor 模型重写详情
