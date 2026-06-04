# 浏览器智能体与长周期网页任务

> ChatGPT Agent（2025年7月）将 Operator 和深度研究合并为一个浏览器/终端智能体，并在 BrowseComp 上创下 68.9% 的 SOTA。OpenAI 于 2025 年 8 月 31 日关闭了独立的 Operator——这是产品层的整合。Anthropic 收购 Vercept 后，Claude Sonnet 在 OSWorld 上的得分从不到 15% 跃升至 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修正了原版 WebArena 中 11.3 个百分点的假阴性率，并发布了包含 258 个任务的 Hard 子集。数据是真实的。攻击面同样真实：OpenAI 的安全准备负责人公开表示，针对浏览器智能体的间接提示注入“不是一个能完全打补丁修复的 Bug”。已记录的 2025–2026 年攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**类型：** 学习
**语言：** Python（标准库、间接提示注入攻击面模型）
**前置知识：** 第 15 阶段 · 10（权限模式）、第 15 阶段 · 01（长周期智能体）
**预计时间：** 约 45 分钟

## 问题所在

浏览器智能体是一种长周期智能体，它会读取不受信任的内容并执行具有实际影响的动作。智能体访问的每一个页面都是用户未编写的输入。页面上的每一个表单都是一个潜在的命令通道。2025–2026 年的攻击案例库表明这并非假设：Tainted Memories 允许攻击者通过精心构造的页面将恶意指令绑定到智能体的记忆中；HashJack 将命令隐藏在智能体访问的 URL 片段中；Perplexity Comet 的一键劫持只需一次点击即可触发。

防御现状令人不安。OpenAI 的安全准备负责人直言不讳地指出：间接提示注入“不是一个能完全打补丁修复的 Bug”。这是因为攻击存在于智能体的“读取”与“执行”边界之中，而该边界在架构上是模糊的——理论上，模型读取的每个 token 都可能被当作指令来解析。

本课将明确攻击面，梳理基准测试格局（BrowseComp、OSWorld、WebArena-Verified），并构建一个最小化的间接提示注入场景，以便你在第 14 和第 18 课中推演实际的防御方案。

## 核心概念

### 2026 年格局：各系统简述

**ChatGPT Agent（OpenAI）。** 2025 年 7 月发布。整合了 Operator（浏览）和 Deep Research（多小时深度研究）。于 2025 年 8 月 31 日下线独立版 Operator。在 BrowseComp 上达到 68.9% 的 SOTA；在 OSWorld 和 WebArena-Verified 上也表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 的重点在于计算机操作能力。使 Claude Sonnet 在 OSWorld 上的得分从不足 15% 提升至 72.5%。Claude Computer Use 作为工具 API 提供。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成提供了计算机操作控制；FSF v3（2026 年 4 月，第 20 课）专门追踪机器学习研发领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个众所周知的问题：原版 WebArena 存在约 11.3% 的假阴性率（即实际已完成但被标记为失败的任务）。Verified 版本采用人工制定的成功标准重新评分，并新增了一个包含 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准测试 | 衡量内容 | 时间跨度 |
|---|---|---|
| BrowseComp | 在限时压力下于开放互联网查找特定事实 | 分钟级 |
| OSWorld | 智能体操作完整桌面环境（鼠标、键盘、Shell） | 数十分钟级 |
| WebArena-Verified | 模拟网站中的事务性网页任务 | 分钟级 |
| Hard subset | 涉及多页面状态转换的 WebArena-Verified 任务 | 数十分钟级 |

评估维度各不相同。BrowseComp 高分仅说明智能体能找到事实，并不代表它能预订航班。OSWorld 分数更接近“能否在我的桌面上正常运行”。WebArena-Verified 则更接近“能否完成一个业务流程”。任何生产环境的决策都需要选择与任务分布相匹配的基准测试。

### 攻击面剖析

1. **间接提示注入。** 不受信任的页面内容包含指令。智能体读取它们。智能体执行它们。公开案例：2024 年 Kai Greshake 等人论文、2025 年 Tainted Memories 论文、2026 年 HashJack（Cato Networks）。
2. **URL 片段/查询参数注入。** 爬取 URL 的 `#fragment` 或查询字符串中包含命令。不会在界面上可见渲染，但仍处于智能体的上下文中。
3. **记忆绑定攻击。** 页面指示智能体写入持久化记忆（第 12 课涵盖持久状态）。在下一次会话中，该记忆会触发有效载荷，且无可见触发器。
4. **针对已认证会话的类 CSRF 攻击。** Tainted Memories 类别：智能体已在某处登录；攻击者的页面发起状态变更请求，智能体会使用用户的 Cookie 执行这些请求。
5. **一键劫持。** 一个视觉上无害的按钮携带着智能体随后会执行的载荷。Comet 类别。
6. **智能体宿主表面的 Content-Security-Policy 漏洞。** 渲染层和工具层本身就可能成为攻击向量；“浏览器嵌套于浏览器智能体”的栈结构非常宽泛。

### 为何“无法完全修补”

该攻击与智能体的能力是同构的。智能体必须读取不受信任的内容才能完成任务。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户的实际需求不符。防御措施（信任边界、分类器、工具白名单、关键操作的 HITL）提高了攻击成本并缩小了其影响范围，但无法彻底消除此类风险。

这与罗伯定理（Lob's theorem，第 8 课）的推理模式相同：智能体无法证明下一个 token 是安全的；它只能建立一个系统，使得不安全的 token 更容易被检测出来。

### 可落地的防御姿态

- **读/写边界。** 读取操作永远不具备实质性影响。如果触发内容来自信任边界之外，写入操作（提交表单、发布内容、调用有副作用的工具）必须经过人类重新审批。
- **按任务配置的工具白名单。** 智能体可以浏览网页；除非该工具被明确授权用于当前任务，否则不能发起电汇等操作。第 13 课涵盖预算控制。
- **会话隔离。** 浏览器智能体会话仅使用作用域受限的凭据运行。不使用生产环境凭证，不关联个人邮箱。保留所有 HTTP 请求日志以供审计。
- **内容清洗器。** 获取的 HTML 在被拼接进模型上下文之前，会剥离已知有害模式。（可减少简单攻击，但无法阻止复杂载荷。）
- **关键操作的 HITL（人在回路）。** 提议-提交模式（第 15 课）。
- **记忆中的金丝雀令牌。** 如果记忆条目被触发，用户会看到提示（第 14 课）。

## 实践应用

`code/main.py` 模拟了一个微型浏览器智能体对三个合成页面的运行过程。其中一个页面是良性的，一个在可见文本中包含直接提示注入载荷，另一个包含 URL 片段注入（不可见但位于智能体上下文中）。该脚本展示了：(a) 朴素智能体会做什么，(b) 读/写边界能拦截什么，(c) 内容清洗器能拦截什么，(d) 两者都无法拦截什么。

## 部署指南

`outputs/skill-browser-agent-trust-boundary.md` 规划了拟议的浏览器智能体部署范围：它将触及哪些信任区域、被授权执行哪些写入操作，以及在首次运行前必须落实哪些防御措施。

## 练习

1. 运行 `code/main.py`。识别出内容清洗器能拦截但读/写边界无法拦截的攻击，以及仅由读/写边界拦截的攻击。

2. 扩展内容清洗器以检测一类 HashJack 风格的 URL 片段注入。测量其在带有合法片段的良性 URL 上的误报率。

3. 选择一个你熟悉的真实浏览器智能体工作流（例如“预订航班”）。列出所有的读取和写入操作。标记哪些写入操作需要 HITL 及其原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。找出原版 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决该问题。

5. 为浏览器智能体环境设计一个记忆金丝雀令牌。你会存储什么、存储在哪里，以及什么会触发警报？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Indirect prompt injection | “坏页面文本” | 智能体读取的页面中，不受信任的内容包含智能体会执行的指令 |
| Tainted Memories | “记忆攻击” | 智能体将攻击者提供的指令写入持久化记忆；在下次会话中触发 |
| HashJack | “URL 片段攻击” | 隐藏在 URL 片段/查询字符串中的载荷位于智能体上下文中，但未可见渲染 |
| One-click hijack | “坏按钮” | 可见的操作入口携带智能体随后会执行的后续载荷 |
| BrowseComp | “网页搜索基准” | 在开放互联网上查找特定事实；分钟级时间跨度 |
| OSWorld | “桌面基准” | 完整操作系统控制；多步骤 GUI 任务 |
| WebArena-Verified | “修复版网页任务基准” | ServiceNow 重新评分的 WebArena，附带 Hard 子集 |
| Read/write boundary | “副作用闸门” | 读取永远无实质影响；若内容超出信任范围，写入需重新审批 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) —— Operator 与深度研究的整合；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) —— Operator 的技术谱系及演变为 ChatGPT Agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) —— 原始基准测试。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) —— ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 包含针对计算机操作智能体的攻击面讨论。
