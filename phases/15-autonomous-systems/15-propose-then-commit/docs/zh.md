# Human-in-the-Loop: Propose-Then-Commit

> 2026 年对 HITL 的共识非常明确。它不是“代理请求，用户点击批准”。而是“先提议后提交”（propose-then-commit）：提议的操作会附带幂等键持久化到持久化存储中；向审核者展示时包含意图、数据血缘、涉及的权限、爆炸半径和回滚计划；仅在获得明确确认后才会提交执行；执行后会进行验证，以确认副作用确实发生。LangGraph 的 `interrupt()` 配合 PostgreSQL 检查点机制、Microsoft Agent Framework 的 `RequestInfoEvent`，以及 Cloudflare 的 `waitForApproval()` 均实现了相同的结构。典型的故障模式是“橡皮图章式批准”：未经审查就点击了“批准？”。文档记录的缓解措施是采用带有明确清单的挑战与响应机制。

**类型：** 学习
**语言：** Python（标准库，带幂等性的提议-提交状态机）
**前置知识：** 第 15 阶段 · 第 12 课（持久化执行）、第 15 阶段 · 第 14 课（Tripwires）
**预计时间：** 约 60 分钟

## The Problem

代理执行某项操作。用户必须做出决定：批准还是不批准。如果决策是瞬间完成的，那很可能算不上真正的审查。如果决策是结构化的，虽然速度较慢，但更可靠。工程上的核心问题是如何让结构化审查成为阻力最小的路径。

2023 年代的 HITL 模式是一种同步提示：“代理想向 X 发送内容为 Y 的邮件——是否批准？”用户点击批准。大家都觉得系统很安全。但在实践中，这种界面极易被“橡皮图章式”地快速通过：用户批准得很快，批准行为本身预测性很低，而当代理出错时，审计日志会显示出一长串用户根本记不起的批准记录。

2026 年的模式——“先提议后提交”——将 HITL 迁移到了持久化基础架构上，附加了结构化元数据，并要求明确的提交动作。每个托管型代理 SDK 都提供了相应版本：LangGraph 的 `interrupt()`、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()`。API 名称各不相同，但底层结构一致。

## The Concept

### The propose-then-commit state machine

1. **Propose.** 代理生成一项提议操作。将其持久化到持久化存储中（PostgreSQL、Redis、Durable Object）。包含以下信息：
   - intent（代理为什么要执行此操作）
   - data lineage（哪些来源数据导致了此项提议）
   - permissions touched（涉及哪些作用域 / 文件 / 端点）
   - blast radius（最坏情况是什么）
   - rollback plan（如果提交执行，如何撤销它）
   - idempotency key（每个提议唯一；重复提交返回相同记录）
2. **Surface.** 审核者查看所有元数据的完整提议。审核者必须是真人（而非代理自我审查）。
3. **Commit.** 获得明确确认。操作开始执行。
4. **Verify.** 执行完成后，回读并确认副作用。如果验证步骤失败，系统处于已知的错误状态，并触发告警。

### The idempotency key

如果没有幂等键，在临时故障后的重试可能会导致已批准的操作被执行两次。具体示例：用户批准“从 A 转账 $100 到 B”。网络抖动。工作流重试。用户只批准了一次，但转账执行了两次。幂等键将批准绑定到单一且唯一的副作用上；第二次执行将成为空操作（no-op）。

这与 Stripe 和 AWS API 使用的幂等模式相同。在 Microsoft Agent Framework 文档中，明确指出了将其复用于代理批准的做法。

### Durability: why approvals outlast processes

批准等待区是一段代理不拥有的状态。工作流处于暂停状态（第 12 课）。当批准到达时，工作流会从确切的位置恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点机制配对使用，而不是仅依赖内存状态——即使两天后才收到批准，工作流依然保持完整。

### Rubber-stamp approvals and the challenge-and-response mitigation

HITL 的默认 UI（“批准”/“拒绝”按钮）会导致无实质审查的快速批准。文档记录的缓解方案是：采用挑战与响应清单，要求对特定问题给出肯定回答后，才能启用“批准”按钮。具体形式如下：

- “你清楚此操作涉及哪些资源吗？[ ]”
- “你已验证爆炸半径在可接受范围内吗？[ ]”
- “如果失败，你有回滚计划吗？[ ]”

这并非为了官僚主义而设——而是一种强制约束机制。无法勾选这些框的审核者要么请求澄清（升级上报），要么拒绝（安全默认值）。Anthropic 的代理安全研究明确指出，基于清单的 HITL 是缓解橡皮图章式批准模式的有效手段。

### What counts as consequential

并非所有操作都需要“先提议后提交”。2026 年的指导原则如下：

- **Consequential actions**（始终需要 HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **Reversible actions**（有时需要 HITL）：本地文件编辑、预发环境变更、具有明确回滚方案的可逆写入。
- **Reads and inspections**（从不需 HITL）：读取文件、列出资源、调用只读 API。

### Post-action verification

“提交已运行”不等于“副作用已发生”。网络分区和竞态条件可能导致工作流认为自身已成功，而后端并未持久化数据。验证步骤会在提交后重新读取目标资源以进行确认。这与带有 `RETURNING` 子句的数据库事务，或 AWS 的 `GetObject` 后置 `PutObject` 模式相同。

### EU AI Act Article 14

第 14 条规定，欧盟高风险 AI 系统必须实施有效的人工监督。“有效”绝非装饰性要求。监管语言明确排除了橡皮图章式模式。在 Microsoft Agent Governance Toolkit 的合规文档中，结合挑战与响应的“先提议后提交”是唯一能通过第 14 条审查的结构。

## Use It

`code/main.py` 使用 Python 标准库实现了一个“先提议后提交”状态机。持久化存储为 JSON 文件。幂等键是 `(thread_id, action_signature)` 的哈希值。驱动程序模拟了三种场景：干净的批准流程、临时故障后的重试（绝不应导致双重执行），以及橡皮图章默认模式与挑战-响应模式的对比。

## Ship It

`outputs/skill-hitl-design.md` 会审查提议中的 HITL 工作流是否符合“先提议后提交”结构，并标记缺失的元数据、幂等性、验证机制或挑战-响应层。

## Exercises

1. 运行 `code/main.py`。确认对已批准提议的重试会使用持久化记录，且不会重新执行。现在修改幂等键以包含时间戳，并演示重试如何导致双重执行。

2. 在提议记录中扩展一个 `rollback` 字段。模拟一个验证步骤失败的执行过程。展示回滚如何自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出该 API 包含但玩具引擎缺失的一个元数据字段。添加它并解释它能防范何种风险。

4. 为特定操作设计一份挑战与响应清单（例如，“发布到公开 Twitter 账号”）。审核者必须回答哪三个问题？为什么是这三个？

5. 选择一个同步“是否批准？”提示足以胜任的场景（无需持久化存储）。解释原因，并说明你正在接受的风险类别。

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Propose-then-commit | “两阶段批准” | 持久化提议 + 明确提交 + 事后验证 |
| Idempotency key | “防重试令牌” | 每个提议唯一；第二次执行视为空操作 |
| Data lineage | “数据来源” | 直接导致该项提议的具体源内容 |
| Blast radius | “最坏情况” | 操作出错时的影响范围 |
| Rubber-stamp | “快速批准” | 未经实质审查即点击“批准” |
| Challenge-and-response | “强制清单” | 审核者必须对特定问题给出肯定确认 |
| RequestInfoEvent | “MS Agent Framework 基础组件” | 带有结构化元数据的持久化 HITL 请求 |
| `interrupt()` / `waitForApproval()` | “框架基础组件” | 与上述结构等效的 LangGraph / Cloudflare 实现 |

## Further Reading

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久化批准机制。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 与 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 HITL 作为缓解长周期风险的手段。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督机制的宪法级框架。
