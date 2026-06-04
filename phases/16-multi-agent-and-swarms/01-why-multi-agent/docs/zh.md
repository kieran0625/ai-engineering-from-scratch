# 为什么需要多智能体？

> 单个智能体遇到瓶颈时，明智的做法不是造一个更大的智能体，而是增加智能体的数量。

**类型：** 学习
**语言：** TypeScript
**前置条件：** 第 14 阶段（智能体工程）
**预计时间：** 约 60 分钟

## 学习目标

- 识别单智能体天花板（上下文溢出、混合专业知识、顺序瓶颈），并解释何时将任务拆分为多个智能体是正确选择
- 比较编排模式（流水线、并行扇出、主管、分层），并根据给定任务结构选择合适的模式
- 设计具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多智能体复杂性（延迟、成本、调试难度）与单智能体简单性之间的权衡

## 问题所在

你在第 14 阶段构建了一个单智能体。它运行良好。它可以读取文件、执行命令、调用 API，并对结果进行推理。然后你让它处理一个真实的代码库：200 个文件、三种编程语言、依赖基础设施的测试，以及要求在编写代码前研究外部 API。

智能体卡住了。不是因为 LLM 太笨，而是因为任务超出了单个智能体循环能处理的范围。上下文窗口被文件内容填满。智能体忘记了 40 次工具调用前读过的内容。它试图同时扮演研究员、程序员和审查员的角色，结果三样都做不好。

这就是单智能体天花板。每当任务需要以下条件时，你都会碰到它：

- **超出单个窗口的上下文容量** - 读取 50 个文件会轻松超过 20 万 token
- **不同阶段需要不同的专业知识** - 研究所需的提示词与代码生成截然不同
- **可以并行处理的工作** - 既然可以同时读取三个文件，为什么要按顺序逐个读取？

## 核心概念

### 单智能体天花板

单个智能体意味着一个循环、一个上下文窗口、一个系统提示词。想象一下它的结构：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

会出现三个问题：

1. **上下文饱和** - 工具结果不断堆积。到第 30 轮时，智能体已经消耗了 15 万 token 的文件内容、命令输出和之前的推理过程。第 5 轮的关键细节就此丢失。

2. **角色混淆** - 一个写着“你是研究员、程序员、审查员和测试员”的系统提示词，会产生一个半吊子研究员、半吊子程序员，且永远无法完成审查的智能体。

3. **顺序瓶颈** - 智能体依次读取文件 A、文件 B、文件 C。三次串行的 LLM 调用。三次串行的工具执行。没有并行性。

### 多智能体解决方案

拆分工作。让每个智能体只负责一项任务，拥有一个独立的上下文窗口，并使用针对该任务优化的系统提示词：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个智能体具备：
- 专注的系统提示词（例如：“你是一个代码审查员。你的唯一任务是发现 bug。”）
- 独立的上下文窗口（不会被其他智能体的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际采用此架构的系统

**Claude Code 子智能体** - 当 Claude Code 使用 ``Task`` 派生子智能体时，它会创建一个具有限定任务的子智能体。父级保持上下文干净，子级执行专注任务并返回摘要。

**Devin** - 运行规划器智能体、编码器和浏览器智能体。规划器将工作分解为步骤。编码器编写代码。浏览器研究文档。每个智能体都有独立的上下文。

**多智能体编码团队（SWE-bench）** - 在 SWE-bench 上表现顶尖的系统通常包含一个读取代码库的研究员、一个设计修复方案的规划器，以及一个实现修复的编码器。单智能体系统的得分较低。

**ChatGPT Deep Research** - 并行启动多个搜索智能体，各自探索不同角度，最后综合结果。

### 演进光谱

多智能体不是非黑即白的。它是一个连续的光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单智能体** - 一个循环，一个提示词。适用于简单任务。

**子智能体** - 父级为专注的子任务派生子级。父级维护计划，子级汇报结果。这正是 Claude Code 的做法。

**流水线** - 智能体按顺序运行。智能体 A 的输出成为智能体 B 的输入。适用于分阶段工作流：研究 -> 编码 -> 审查 -> 测试。

**团队** - 智能体通过共享消息总线并行运行。每个智能体有明确角色，由协调器统筹。适用于需要同时运用不同技能的场景。

**群体（Swarm）** - 大量相同或高度相似的智能体共享状态。没有固定的协调器。智能体从队列中领取任务。适用于高吞吐量的并行任务。

### 四种多智能体模式

#### 模式 1：流水线

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个智能体转换数据并将其传递下去。易于理解。但某一阶段失败会导致后续所有阶段阻塞。

#### 模式 2：扇出 / 扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将工作分配给并行智能体，然后合并结果。适用于可分解为独立子任务的工作。

#### 模式 3：协调器-工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个智能的协调器决定做什么，委派给工作者，并综合结果。协调器本身也是一个带有创建工作者工具的 Agent。

#### 模式 4：对等群体

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央协调器。智能体之间点对点通信。决策通过交互涌现。调试较难，但可扩展至大量智能体。

### 何时不应使用多智能体

多智能体会增加复杂性。智能体之间的每条消息都可能成为故障点。调试从“阅读一次对话”变成“跨五个智能体追踪消息”。

**满足以下条件时请保持单智能体：**
- 任务适合放入单个上下文窗口（工作数据少于约 10 万 token）
- 不同阶段不需要不同的系统提示词
- 顺序执行速度足够快
- 任务足够简单，拆分它带来的开销大于收益

**复杂性成本：**
- 每个智能体边界都是一次有损压缩步骤：智能体 A 的完整上下文会被摘要成一条消息发送给智能体 B
- 协调逻辑（谁做什么、何时做、以什么顺序）本身就是 bug 的来源
- 延迟增加：N 个智能体意味着至少 N 次串行 LLM 调用，如果需要来回沟通则更多
- 成本倍增：每个智能体独立消耗 token

经验法则：如果任务所需工具调用少于 20 次且能放入 10 万 token 的上下文中，请保持单智能体架构。

## 动手实践

### 步骤 1：超负荷的单智能体

这是一个试图包揽一切的单智能体。它有一个庞大的系统提示词，以及一个同时容纳研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的缺点：
- 上下文窗口随每个阶段不断膨胀。到了审查步骤，其中已包含研究笔记、代码和之前的推理过程。
- 系统提示词过于通用，无法针对每个阶段进行优化。
- 没有任何部分可以并行运行。

### 步骤 2：专家智能体

现在将其拆分。每个智能体只负责一项工作：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家智能体都有专注的提示词。每个都获得干净的上下文窗口，仅包含其所需的输入。

### 步骤 3：通过消息协调

通过显式消息传递将专家智能体连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个智能体仅接收发给自己的消息。无上下文污染。研究员那 5 万 token 的文档阅读记录永远不会进入审查员的上下文。

### 步骤 4：对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多智能体版本使用的总 token 更多（三个智能体，三次独立的 LLM 调用），但每个智能体的上下文保持干净。由于系统提示词经过专门优化，各阶段的质量得到提升。

## 实际应用

本课程提供了一个可复用的提示词，用于判断何时采用多智能体架构。详见 ``outputs/prompt-multi-agent-decision.md``。

## 练习

1. 添加第四个专家：“测试员”智能体，它接收来自编码器的代码和来自审查员的反馈，然后编写测试用例
2. 修改流水线，使审查员可以将反馈发回给编码器进行修订循环（最多 2 轮）
3. 将顺序流水线转换为扇出模式：并行运行研究员和“需求分析员”智能体，然后在传递给编码器之前合并它们的输出

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Swarm（群体） | “AI 智能体的蜂巢思维” | 一组共享状态且无固定领导者的对等智能体。行为通过局部交互自然涌现。 |
| Orchestrator（协调器） | “老板智能体” | 其工具包含创建和管理其他智能体的 Agent。它负责规划和委派，但不一定亲自执行具体工作。 |
| Coordinator（调度器） | “交通警察” | 非智能体组件（通常是普通代码而非 LLM），根据规则在智能体之间路由消息。 |
| Consensus（共识） | “智能体达成一致” | 一种协议，要求多个智能体在继续下一步前必须达成共识。用于解决冲突输出的场景。 |
| Emergent behavior（涌现行为） | “智能体自己搞定了” | 源于智能体交互的系统级模式，并未被显式编程。可能有益也可能有害。 |
| Fan-out / fan-in（扇出/扇入） | “智能体的 MapReduce” | 将任务分配给并行智能体（扇出），然后合并它们的结果（扇入）。 |
| Message passing（消息传递） | “智能体互相交谈” | 智能体间的通信机制：从一个智能体发送到另一个智能体的结构化数据，用于替代共享上下文窗口。 |

## 延伸阅读

- [新兴 AI 智能体架构全景](https://arxiv.org/abs/2409.02977) - 多智能体模式综述
- [AutoGen：赋能下一代 LLM 应用](https://arxiv.org/abs/2308.08155) - 微软的多智能体对话框架
- [Claude Code 子智能体文档](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何使用 Task 进行委派
- [CrewAI 文档](https://docs.crewai.com/) - 基于角色的多智能体框架
