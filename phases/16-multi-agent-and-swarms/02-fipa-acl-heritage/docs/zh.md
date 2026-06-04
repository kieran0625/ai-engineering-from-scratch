# FIPA-ACL 与言语行为的传承

> 在 MCP 和 A2A 之前，早有 FIPA-ACL。2000 年，IEEE 智能物理代理基金会（IEEE Foundation for Intelligent Physical Agents）正式批准了一种代理通信语言，包含二十种施为性动作（performatives）、两种内容语言以及一套交互协议——合同网、订阅/通知、请求-当条件成立。它之所以淡出工业界，是因为其本体负担对 Web 而言过于沉重；但如今大语言模型（LLM）复兴的多代理系统正在悄然重新实现这些理念，只是去掉了形式化语义：JSON 契约替代了施为性动作，自然语言替代了本体。本课程将严肃对待 FIPA-ACL，以便你能看清 2026 年的哪些协议决策是重复造轮子，哪些是真正的新颖设计，以及当前浪潮将在何处重新发现 2000 年代早已解决的问题。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** 阶段 16 · 01（为何需要多代理）
**耗时：** 约 60 分钟

## 问题

2026 年的代理协议生态十分繁忙：MCP 用于工具调用，A2A 用于代理间通信，ACP 用于企业审计，ANP 用于去中心化信任，NLIP 用于自然语言内容，此外还有 CA-MCP 以及数十项研究提案。每项规范都宣称自己是基础性的。

客观来看，其中大多数项目只是在重新发现一棵非常具体的二十年前的决策树。Austin（1962）和 Searle（1969）的言语行为理论告诉我们“话语即行动”。KQML（1993）将其转化为网络传输协议。FIPA-ACL（2000 年批准）确立了参考标准：二十种施为性动作、SL0/SL1 内容语言、针对合同网和订阅/通知的交互协议。JADE 和 JACK 是当时的 Java 参考平台。该努力在 2010 年左右逐渐式微，因为本体负担过重，而 Web 技术取得了胜利。

当你审视 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的其实是 FIPA 决策的更柔和、原生 JSON 化的重述。了解这段历史能告诉你两件事：哪些新的“创新”实际上是重复造轮子，以及新规范将重新发现哪些旧有的失败模式。

## 核心概念

### 一段话讲清言语行为

Austin 注意到，有些句子并不描述世界——它们改变世界。“我承诺。”“我请求。”“我宣布。”他将这些称为施为性话语。Searle 将其形式化为五类：断言型、指令型、承诺型、表达型、宣告型。KQML（Finin 等人，1993）使这一概念在软件代理中可操作：一条消息由一个施为性动作（即行动本身）加上内容（行动的指向对象）组成。FIPA-ACL 修复了 KQML 的缺陷，并围绕二十种施为性动作进行了标准化。

### 二十种 FIPA 施为性动作（部分列表）

| Performative | Intent |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我要求你执行 X” |
| `query-if` | “P 是否为真？” |
| `query-ref` | “X 的值是多少？” |
| `propose` | “我建议我们执行 X” |
| `accept-proposal` | “我接受该提议” |
| `reject-proposal` | “我拒绝该提议” |
| `agree` | “我同意执行 X” |
| `refuse` | “我拒绝执行 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “你的消息无法解析” |
| `cfp` | “征集关于 X 的提案” |
| `subscribe` | “当 X 发生变化时通知我” |
| `cancel` | “取消进行中的 X” |
| `failure` | “我尝试过 X 但失败了” |

完整列表见 `fipa00037.pdf`（FIPA ACL 消息结构）。重点不在于死记硬背——而在于指出这些中的每一个都对应着 LLM 协议最终会重新引入的基础原语。

### 标准 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；其中一个字段（`content`）承载有效载荷。其余字段正是你每次向 JSON 协议附加重试机制、线程管理和本体论时所必须重新发明的东西。

### 两款遗留平台

**JADE**（Java Agent DEvelopment 框架，1999–2020 年代）是使用最广泛的符合 FIPA 标准的运行时环境。代理继承基类，交换 ACL 消息，在容器内运行，并使用“行为”进行协调。其交互协议库内置了合同网、订阅/通知、请求-当条件成立以及提议-接受。

**JACK**（Agent Oriented Software，商业版）强调在 FIPA 消息之上进行 BDI（信念-欲望-意图）推理。更加形式化，但采用率较低。

一旦 Web 技术栈吞噬了多代理用例，两者便双双衰落。MCP 和 A2A 即是 2026 年的运行时“容器”。

### FIPA 为何式微

- **本体负担。** FIPA 需要共享本体来解析 `content`。就本体达成共识是一个长达数年的标准制定过程。而 Web 直接使用了 HTTP + JSON。
- **无人使用的形式化语义。** SL（语义语言）提供了严格真值条件，但大多数生产系统使用自由格式内容并忽略了形式化体系。
- **工具链锁定。** JADE 仅限 Java；JACK 为商业软件。多语言团队绕开了两者。
- **互联网赢得了技术栈。** REST，随后是 JSON-RPC，再到 gRPC，取代了 ACL 的传输层。

### LLM 的复兴本质上是轻量版 FIPA

比较 FIPA 的 `request` 与 MCP 的 `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

信封相同，语法不同。两者均携带：发送者、接收者、意图、有效载荷、关联 ID。二者之间没有谁颠覆谁的说法——它们是对同一设计的不同权衡。

Liu 等人 2025 年的调查报告（《代理互操作性协议综述：MCP、ACP、A2A、ANP》，arXiv:2505.02279）明确指出了这一谱系关系：MCP 对应工具使用类言语行为，A2A 对应代理对等类言语行为，ACP 对应审计追踪类言语行为，ANP 对应去中心化身份扩展。新规范是带有 JSON 语法和更宽松语义的 ACL 后代。

### 直白的权衡

**FIPA 提供但现代规范舍弃的内容：**

- 形式化语义——你可以证明 `inform` 意味着发送者相信该内容。
- 标准化的施为性动作目录——你无需再争论“我们是否需要一个 `cancel`？”。
- 数十年的交互协议模式——合同网、订阅/通知、提议-接受——具备已知的正确性属性。

**现代规范提供但 FIPA 不具备的内容：**

- 兼容所有现代工具的 JSON 原生有效载荷。
- 无需手工编码本体即可被 LLM 解释的自然语言内容。
- Web 技术栈传输层（HTTP、SSE、WebSocket）。
- 通过自描述文档进行能力发现（MCP `listTools`、A2A 代理卡片）。

以意图语义的放宽换取实现的简便。这就是确切的权衡。

### 值得迁移的交互协议

FIPA 发布了约 15 种交互协议。其中三种值得带入 LLM 多代理系统：

1. **合同网协议（CNP）。** 管理器发布 `cfp`（征集提案）；竞标者回复 `propose`；管理器接受/拒绝。这是典型的任务市场模式（阶段 16 · 16 谈判）。
2. **订阅/通知。** 订阅者发送 `subscribe`；每当主题发生变化时，发布者发送 `inform`。这正是 2026 年所有事件总线的雏形。
3. **请求-当条件成立。** “当条件 Y 满足时执行 X。”带前置条件的延迟动作。2026 年的对应物是持久工作流引擎中的延迟任务（阶段 16 · 22 生产扩展）。

它们均可清晰地映射到现代消息队列、HTTP + 轮询或 SSE 流式传输。

### 丢弃本体后会发生什么

在没有共享本体的情况下，代理会从自然语言内容中推断含义。2026 年记录的故障模式是**语义漂移**：两个代理对细微不同的概念使用相同的词（`"customer"`），接收方代理基于错误的理解采取行动，且没有任何模式验证器能捕获此错误。FIPA 的本体要求在解析阶段就会拒绝该消息。

不采用完整本体的缓解措施：

- 对 `content` 使用 JSON Schema —— 在网络传输层拒绝结构错误。
- 类型化工件（A2A） —— 拒绝错误的模态。
- 信封中显式声明施为性动作 —— 即使内容是自然语言，也能确保意图无歧义。

### 2026 年规范与言语行为遗产的映射

| Modern spec | FIPA analog | What it keeps | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、关联 ID | 形式化语义、本体 |
| MCP `resources/read` | `query-ref` | 显式意图、关联 ID | 形式化语义 |
| A2A Task lifecycle | contract-net + request-when | 异步生命周期、状态转换 | 形式化完备性保证 |
| A2A streaming events | subscribe/notify | 异步推送 | 类型谓词订阅 |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | 多写入共享内存 | 逻辑一致性模型 |
| NLIP | natural-language content | LLM 原生 | 模式验证 |

从上到下阅读该表，规律显而易见：保留结构原语，舍弃形式化体系，让 LLM 去掩盖歧义。

## 动手构建

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 转换器。它编码和解码标准 ACL 信封，并展示每个 MCP/A2A 消息形状如何归约为相同的七个字段。演示内容包括：

- 将五条 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等效格式。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一名管理器和三名竞标者之间运行玩具合同网协商。

运行：

```
python3 code/main.py
```

输出结果是一份并排追踪日志，展示每条现代消息的 2026 JSON 格式与其 FIPA-ACL 格式，随后是一次合同网竞标的往返过程。相同的协议原语在往返过程中得以保留；仅语法不同。

## 实际应用

`outputs/skill-fipa-mapper.md` 是一项技能，可读取任意代理协议规范并生成 FIPA-ACL 映射。在采用新协议前使用它来回答：“这究竟是真正的创新，还是仅仅换了 JSON 语法的 `inform`？”

## 交付上线

不要重新引入 FIPA-ACL。请带回它的检查清单：

- 每条消息的意图原语（施为性动作）是什么？
- 是否存在用于请求-响应和取消的关联 ID？
- 是否明确了内容语言（JSON-RPC、纯文本、结构化类型工件）？
- 交互协议是否是一等公民，还是你在从头重新实现合同网？
- 当两个代理对内容含义存在分歧时（语义漂移）会发生什么？

在任何新协议投入生产环境之前，请将这五个问题记录下来。

## 练习

1. 运行 `code/main.py`。观察往返编码过程。指出哪个 FIPA 施为性动作分别对应 `tools/call`、`resources/read` 以及 A2A 任务创建。
2. 在合同网演示中增加一个 `cancel` 施为性动作，允许管理器在竞标中途撤回任务。`cancel` 解决了单纯重试无法解决的哪种故障场景？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1–4.3 节。选择本课未涵盖的一个施为性动作，并描述其现代 JSON-RPC 对应物。
4. 阅读 Liu 等人，arXiv:2505.02279。针对 MCP、A2A、ACP、ANP，分别列出它们保留和舍弃的 FIPA 施为性动作家族。
5. 为你系统中 `request` 施为性动作的 `content` 字段设计一个最小 JSON Schema。该模式带来了纯自然语言所不具备的什么优势，又付出了什么代价？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|------------------------|
| Speech act | “产生某种作用的语句” | Austin/Searle：语句即行动。ACL 的理论源头。 |
| FIPA | “那个古老的 XML 玩意儿” | IEEE 智能物理代理基金会。于 2000 年标准化了 ACL。 |
| ACL | “代理通信语言” | FIPA 的信封格式：施为性动作 + 内容 + 元数据。 |
| Performative | “动词” | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA 的前身” | 知识查询与操作语言（1993）。更简单，范围更窄。 |
| Ontology | “共享词汇表” | 对内容语言所讨论的概念的形式化定义。 |
| SL0 / SL1 | “FIPA 内容语言” | 语义语言的第 0 级和第 1 级——形式化内容语言家族。 |
| Contract Net | “任务市场” | 管理器发布 cfp；竞标者提出方案；管理器接受。经典的交互协议。 |
| Interaction protocol | “消息模式” | 具有已知正确性的施为性动作序列：请求-当条件成立、订阅/通知等。 |

## 延伸阅读

- [Liu 等人 — 《代理互操作性协议综述：MCP、ACP、A2A、ANP》](https://arxiv.org/html/2505.02279v1) —— 连接现代规范与 FIPA 遗产的经典 2025 年调查报告
- [FIPA ACL 消息结构规范 (fipa00037)](http://www.fipa.org/specs/fipa00037/) —— 2000 年批准的信封格式
- [FIPA 交际行为库规范 (fipa00037)](http://www.fipa.org/specs/fipa00037/) —— 完整的施为性动作目录
- [MCP 规范 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— `request`/`query-ref` 的现代工具使用等效物
- [A2A 规范](https://a2a-protocol.org/latest/specification/) —— 合同网和订阅/通知的现代代理对等等效物
