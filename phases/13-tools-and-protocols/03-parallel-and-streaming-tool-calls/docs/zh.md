# 并行工具调用与流式工具处理

> 三次独立的气象查询若串行执行，需要三次往返。并行运行它们则总耗时将缩减至最慢的那一次调用。目前所有主流大模型提供商均支持在单轮对话中发出多个工具调用。收益是实实在在的；但底层实现机制却颇为微妙。本课程将分两部分讲解：并行扇出（fan-out）与流式参数重组，重点强调 ID 关联陷阱。

**类型：** 构建实践
**语言：** Python（标准库、线程池 + 流式测试框架）
**前置知识：** 第 13 阶段 · 02（函数调用深度解析）
**预计时间：** 约 75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 的存在原因及何时应禁用它。
- 在并行扇出期间，将流式参数块正确关联到对应的工具调用 ID。
- 将部分 `arguments` 字符串重组为完整的 JSON，避免过早解析。
- 运行一个三城市气象基准测试，演示串行与并行延迟的差异。

## 问题所在

若无并行调用，回答“班加罗尔、东京和苏黎世天气如何”的代理会按以下步骤执行：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还需承担执行器延迟。总耗时约为理想墙钟时间的 4 倍。

采用并行调用时：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

仅需一次 LLM 往返。执行器耗时取三者中的最大值而非总和。在 OpenAI、Anthropic 和 Gemini 上的生产环境基准测试表明，扇出工作负载的墙钟时间可减少 60% 至 70%。

代价是关联复杂度。当这三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型将它们对齐。当结果流式返回时，你必须在执行前将部分参数片段组装成完整的 JSON。Gemini 3 引入唯一 ID 的部分原因正是为了解决这一现实问题：两个针对同一工具的并行调用无法区分。

## 核心概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设置 `false` 可强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 实现并行（Claude 3.5 及以上版本默认开启）。设置 `true` 可切换为串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 交由模型自行决定。

当工具存在顺序依赖关系（先 `create_file` 再 `write_file`）、某个调用的输出作为另一个调用的输入，或速率限制器无法处理扇出请求时，请禁用并行。

### ID 关联

模型发出的每个调用都包含一个 `id`。主机返回的每个结果都必须包含相同的 ID。否则结果将无法明确对应。

- **OpenAI。** 在每个工具角色消息上标记 `tool_call_id`。
- **Anthropic。** 在每个 `tool_result` 块上标记 `tool_use_id`。
- **Gemini。** 在每个 `functionResponse` 上标记 `id`（Gemini 3 及以上版本；Gemini 2 仅通过名称匹配，导致同名并行调用出错）。

### 并发执行调用

主机在独立的线程、协程或远程工作节点上运行每个调用的执行器。最简单的测试框架使用线程池；生产环境通常使用带有 `asyncio.gather` 的 asyncio 或结构化并发。完成顺序不可预测——ID 才是唯一的标识符。

一个常见错误：按调用列表顺序而非完成顺序回复结果。这通常能正常工作，因为模型只关心 `tool_call_id`，但如果结果丢失或重复，乱序提交会使调试更加困难。建议按完成顺序回复并显式附带 ID。

### 流式工具调用

当模型流式输出时，`arguments` 是分片到达的。三个并行调用的独立数据流在网络传输时会交错。你需要为每个 ID 维护一个累加器。

各提供商的数据结构差异如下：

- **OpenAI。** 每个数据块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。数据块携带 `index`（在调用列表中的位置）。你按索引进行累加，首次出现 `id` 时读取它，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件首先是 `message_start`，随后每个块有一个 `content_block_start`，类型为 `tool_use`（包含 id、name 和空 input）。`content_block_delta` 事件携带 `input_json_delta` 数据块。`content_block_stop` 用于关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上版本）发出的数据块带有 `functionCallId`，使得调用能够干净地交错。在 Gemini 3 之前，流式输出是一次返回一个完整调用。

### 部分 JSON 与过早解析陷阱

在 `arguments` 完整之前，你无法对其进行解析。诸如 `{"city": "Beng` 这样的部分 JSON 无效且会引发异常。正确的触发条件是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop` 或 Gemini 的流结束事件。只有此时才尝试 `json.loads`。更稳健的方法是使用增量 JSON 解析器，它在结构完成时逐次产出事件；OpenAI 的流式指南推荐此方法以实现显示实时“思考”指示器的用户体验。通过统计花括号数量来判断完整性并不可靠（引号内的花括号或转义内容会导致误判），仅可作为非正式的调试启发式手段。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

主机回复仍必须引用这些 ID：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回复的顺序不影响正确性。只要 ID 匹配，Gemini 接受任意顺序。

### 基准测试：串行与并行

`code/main.py` 中的测试框架模拟了三个延迟分别为 400、600 和 800 毫秒的执行器。串行执行总耗时为 1800 毫秒。并行执行耗时为 max(400, 600, 800) = 800 毫秒。这种差异是恒定的而非成比例的，因此随着工具数量的增加，节省的时间也会相应增长。

现实注意事项：并行调用会给下游 API 带来压力。向受速率限制的服务发起 10 路扇出将会失败。第 13 阶段 · 17 将涵盖网关级别的背压机制；重试语义计划在后续阶段补充。

### 流式扇出的墙钟时间

如果模型本身支持流式输出，你可以在单个调用的参数完整后立即开始执行，而无需等待所有调用完成。这是 OpenAI 文档中记录的一项优化，但并非所有 SDK 都暴露了该功能。本课程的测试框架实现了这一点：一旦模拟流产出一个完整的参数对象，主机便会立即启动该调用。

## 动手实践

`code/main.py` 分为两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 依次以串行和并行方式运行三个模拟气象调用，并打印墙钟时间。第二部分重放一个伪造的流式响应——三个并行调用的 `arguments` 数据块在一个流中交错——并使用 `StreamAccumulator` 按 ID 进行重组。不涉及 LLM，也不涉及网络通信，仅展示重组逻辑。

重点关注：

- 串行计时器达到 1.8 秒。在相同的模拟延迟下，并行计时器达到 0.8 秒。
- 累加器通过按 ID 缓冲来处理乱序到达的数据块，仅在每个调用的 JSON 完整时才进行解析。
- 执行器在某个 ID 的参数确定后立即启动，而不是等待所有流结束。

## 交付成果

本课程将生成 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能会审计哪些工具可以安全地并行化、哪些存在顺序依赖关系、哪些会压垮下游速率限制——最终返回一份修订后的注册表，其中包含每个工具的 `parallel_safe` 标志。

## 练习

1. 运行 `code/main.py` 并调整模拟延迟。确认并行与串行的比例约为 `max/sum`（由于线程调度、序列化和测试框架开销，实际运行结果会略偏离理想值）。在何种延迟分布下，并行优势不再明显？

2. 扩展累加器以处理“调用在流式传输中途被取消”的情况，具体做法是丢弃其缓冲区并发出一个 `cancelled` 事件。哪个提供商明确记录了这种情况？查阅 Anthropic 的 `content_block_stop` 语义以及 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该能看到异步带来的小幅性能提升，因为其上下文切换成本更低，但这仅在执行器执行真实 I/O 操作时成立。

4. 选择两个不应并行化的工具（例如先 `create_file` 再 `write_file`）。向注册表中添加一个 `ordering_dependency` 图，并基于该图控制并行扇出。这是实现感知依赖关系调度的最小化机制，将在未来的智能体工程阶段进行形式化定义。

5. 阅读 OpenAI 的并行函数调用章节以及 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行化的那一种现实世界工具类型。（提示：对同一资源产生实质性变更的操作。）

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|------------------------|
| 并行工具调用 | “一轮对话内扇出” | 模型在单条助手消息中发出多个工具调用 |
| `parallel_tool_calls` | “OpenAI 的标志位” | 启用或禁用多调用发射 |
| `disable_parallel_tool_use` | “Anthropic 的反向标志” | 退出标志；默认启用并行 |
| 工具调用 ID | “关联句柄” | 结果消息必须回显的每调用唯一标识符 |
| 累加器 | “流式缓冲区” | 用于部分 `arguments` 数据块的按 ID 字符串缓冲区 |
| 乱序完成 | “最快优先” | 并行调用以不可预测的顺序完成；ID 是粘合剂 |
| 依赖图 | “顺序约束” | 输出作为其他工具输入的工具有序依赖；无法并行化 |
| 过早解析陷阱 | “JSON.parse 爆炸” | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | “Gemini 3 特性” | 带有每调用唯一 ID 的流式参数数据块 |
| 完成顺序回复 | “不必等全部” | 按到达顺序回复结果，以 ID 为键 |

## 延伸阅读

- [OpenAI — 并行函数调用](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) —— 默认行为与退出标志
- [Anthropic — 工具使用：实现工具调用](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) —— `disable_parallel_tool_use` 与结果批处理
- [Google — Gemini 函数调用并行章节](https://ai.google.dev/gemini-api/docs/function-calling) —— 来自 Gemini 3 的 ID 关联并行调用
- [OpenAI — 带工具的流式响应](https://platform.openai.com/docs/api-reference/responses-streaming) —— OpenAI 流式的分块参数重组
- [Anthropic — 流式消息](https://docs.anthropic.com/en/api/messages-streaming) —— `content_block_delta` 配合 `input_json_delta`
