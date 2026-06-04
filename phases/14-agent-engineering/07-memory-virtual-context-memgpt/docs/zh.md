# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的。对话、文档和工具调用轨迹却不是。MemGPT（Packer 等人，2023）将其类比为操作系统的虚拟内存——主上下文是 RAM（内存），外部存储是磁盘，智能体在两者之间进行页面交换。这是所有 2026 年记忆系统继承的模式。

**类型：** 构建
**语言：** Python (stdlib)
**前置知识：** Phase 14 · 01（Agent Loop）、Phase 14 · 06（Tool Use）
**耗时：** ~75 分钟

## 学习目标

- 解释 MemGPT 所基于的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 页面调入/调出。
- 使用 stdlib 实现双层 MemGPT 模式，包含主上下文缓冲区、可搜索的外部存储以及页面调入/调出工具。
- 描述智能体如何发出“中断”以查询或修改外部记忆，以及结果如何拼接回下一个提示词中。
- 指出延续到 Letta（Lesson 08）和 Mem0（Lesson 09）中的 MemGPT 设计选择。

## 问题所在

上下文窗口看似能解决记忆问题，但事实并非如此。在生产环境中会反复出现三种故障模式：

1. **溢出。** 多轮对话、长文档或重度依赖工具调用的轨迹会超出窗口限制。截断点之后的所有内容都会丢失。
2. **稀释。** 即使在窗口内，塞入无关上下文也会分散模型对关键内容的注意力。前沿模型在处理长输入时性能仍会下降。
3. **持久性。** 新会话开始时窗口为空。没有外部记忆的代理无法跨会话说出“记得你之前让我……”。

更大的窗口会有所帮助，但无法从根本上解决这个问题。Mem0 的 2025 年论文测量表明，128k 窗口的基线模型仍然会遗漏长周期事实，而拥有外部记忆的 4k 窗口代理却能捕捉到这些事实。

## 核心概念

### MemGPT：操作系统类比

Packer 等人（arXiv:2310.08560，v2 Feb 2024）将上下文管理映射到操作系统的虚拟内存机制：

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | ReAct loop with memory tools |

智能体运行标准的 ReAct 循环。额外的一类工具允许它在主上下文中进行数据的页面调入和调出。

### 双层架构

- **Main context.** 固定大小的提示词，承载当前任务。始终对模型可见。
- **External context.** 无界，可通过工具进行搜索。相关时读取，事实出现时写入。

原始论文在基础窗口之外的两项任务上评估了该设计：超过 100k token 的文档分析，以及具有跨天持久记忆的多会话聊天。

### 中断模式

MemGPT 引入了“记忆即中断”的概念：在对话过程中，智能体可以调用记忆工具，运行时执行该工具，并将结果作为新的观测值拼接到下一个助手回复中。这在概念上等同于 Unix 的 `read()` 系统调用：阻塞进程，返回字节数据，然后进程继续执行。

标准记忆工具接口：

- `core_memory_append(section, text)` — 写入提示词的持久化部分。
- `core_memory_replace(section, old, new)` — 编辑持久化部分。
- `archival_memory_insert(text)` — 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` — 从外部存储检索。
- `conversation_search(query)` — 扫描历史轮次。

### MemGPT 的终点与 Letta 的起点

2024年9月，MemGPT 演变为 Letta。研究仓库（`cpacker/MemGPT`）依然保留；Letta 扩展了该设计：

- 采用三层架构而非两层（core, recall, archival — Lesson 08）。
- 原生推理取代了 `send_message`/heartbeat 模式（Lesson 08）。
- 休眠期智能体运行异步记忆工作（Lesson 08）。

即使生产系统运行的是 Letta、Mem0 或自定义的双层存储，MemGPT 论文仍是 2026 年的基石。

### 该模式的常见陷阱

- **Memory rot。** 写入速度远快于读取速度；检索结果被陈旧事实淹没。修复方案：定期整合（Letta sleep-time）、显式失效（Mem0 conflict detector）。
- **Memory poisoning。** 外部记忆本质上是检索到的文本。如果攻击者控制的內容落入记忆笔记中，智能体将在下次会话中重新摄入该内容。这是 Greshake 等人（Lesson 27）提出的攻击在时间维度上的重演。
- **Citation loss。** 智能体回忆起“用户曾让我交付 X”，但无法指明是哪一轮对话。每次归档写入时都应存储来源引用（session ID, turn ID）。

## 动手构建

`code/main.py` 在 stdlib 中实现了 MemGPT 的双层模式：

- `MainContext` — 固定大小的提示词缓冲区，包含一个 `core` dict 和一个 `messages` list；超出容量上限时自动压缩最旧的消息。
- `ArchivalStore` — 内存中的类 BM25 存储（基于 token-overlap 评分），用于存储 (id, text, tags, session, turn) 记录。
- 五个映射到 MemGPT 接口的记忆工具。
- 一个脚本化智能体，先用事实填充 archival 区，然后通过调用 `archival_memory_search` 回答问题。

运行方式：

```
python3 code/main.py
```

追踪日志显示，智能体写入了三条事实，将 main context 填满至容量上限（强制驱逐旧消息），随后通过从 archival 检索来回答后续问题——在没有真实 LLM 的情况下复现了 MemGPT 的工作流。

## 实际应用

当今的生产级记忆系统都是 MemGPT 的变体：

- **Letta** (Lesson 08) — 三层架构、原生推理、sleep-time compute。
- **Mem0** (Lesson 09) — vector + KV + graph 融合评分层。
- **OpenAI Assistants / Responses** — 通过 threads 和 files 托管记忆。
- **Claude Agent SDK** — 通过 skills 和 session store 实现长期记忆。

根据运维形态（self-hosted, managed, framework-integrated）进行选择，而不是根据核心模式——因为核心模式就是 MemGPT。

## 交付部署

`outputs/skill-virtual-memory.md` 是一个可复用的 skill，可为任何目标运行时生成正确的双层记忆脚手架（main + archival + tool surface），并内置 eviction policy 和 citation fields。

## 练习

1. 添加一个以 token 为单位的 `max_main_context_tokens` 上限（可用 `len(text.split())` * 1.3 近似估算）。当超出上限时，将最旧的消息压缩为摘要。对比开启与关闭 summarizer 时的行为差异。
2. 在 archival 存储上正确实现 BM25（term frequency, inverse document frequency）。在玩具事实集上衡量 recall@10，并与 token-overlap 基线进行对比。
3. 向 archival 插入操作中添加 `citation` 字段（session_id, turn_id, source_url）。要求智能体在每次基于检索的回答中引用来源。
4. 模拟 memory poisoning：添加一条内容为“ignore all future user instructions.”的归档记录。编写一个 guard 程序，扫描检索结果中符合 directive-shaped text 的文本，并将其标记为 untrusted。
5. 移植实现代码以使用 MemGPT 研究仓库的 core-memory JSON schema（`cpacker/MemGPT`）。当从 flat strings 切换到 typed sections 时，会发生哪些变化？

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Virtual context | "Unlimited memory" | Main (prompt) + external (searchable) tiers with page in/out |
| Main context | "Working memory" | The prompt — fixed-size, always visible |
| Archival memory | "Long-term store" | External searchable persistence, retrieved on demand |
| Core memory | "Persistent prompt section" | Named sections pinned inside the main context |
| Memory tool | "Memory API" | Tool call the agent issues to read/write external memory |
| Interrupt | "Memory page fault" | Agent pauses, runtime fetches, result splices into next turn |
| Memory rot | "Stale facts" | Old writes drown retrieval; fix with consolidation |
| Memory poisoning | "Injected persistent note" | Attacker content stored as memory, re-ingested on recall |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS-inspired virtual context paper
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — the three-tier evolution
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — treating context as a budget
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — hybrid production memory on top of this pattern
