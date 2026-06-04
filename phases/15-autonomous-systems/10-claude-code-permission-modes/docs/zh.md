# Claude Code 作为自主智能体：权限模式与 Auto Mode

> Claude Code 提供了七种权限模式。“plan”会在每次操作前询问，“default”仅对高风险操作进行询问，“acceptEdits”会自动批准文件写入但仍会确认 Shell 执行，而“bypassPermissions”则批准所有操作。Auto Mode（2026年3月24日）用两阶段并行安全分类器取代了逐操作审批机制：单次 token 的快速检查会对每项操作运行；被标记的操作将触发基于思维链的深度审查。操作预算通过 `max_turns` 和 `max_budget_usd` 进行强制实施。Auto Mode 以研究预览版形式发布——Anthropic 明确表示，该分类器本身并不足以提供完整保障。

**类型：** 学习
**语言：** Python（标准库、两阶段分类器模拟器）
**前置知识：** 第 15 阶段 · 01（长周期智能体）、第 15 阶段 · 09（编码智能体生态）
**耗时：** 约 45 分钟

## 问题所在

你机器上的自主编码智能体属于一个独立的安全类别。其攻击面涵盖智能体能够触及的一切——文件系统、网络、凭据、剪贴板、任意浏览器标签页、任意打开的终端。Bruce Schneier 等人已公开指出这一点：计算机使用型智能体并非聊天机器人的“功能更新”，而是一种具有全新风险特征的新型工具。

Claude Code 的权限系统是 Anthropic 给出的答案。它没有采用单一的“自主/非自主”开关，而是提供了跨越能力阶梯的七种模式：plan → default → acceptEdits → … → bypassPermissions。每种模式在速度与逐操作审查之间提供了不同的权衡。Auto Mode（2026年3月）引入了一种两阶段分类器，对于分类器判定为安全的操作，将其审批移出用户的关键路径；而对于分类器标记的操作，则保留审查层。

工程层面的核心问题是：该系统能拦截什么？会漏掉什么？某项具体任务究竟适合哪种模式？

## 核心概念

### 七种权限模式

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `plan` | 智能体提出计划；用户批准整个计划；每项操作在执行前均经过审查 | 陌生任务；接近生产环境的代码；首次在该仓库使用智能体 |
| `default` | 智能体执行操作；对任何“高风险”操作（Shell 执行、破坏性操作、网络调用）提示用户 | 大多数交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；Shell 执行和网络调用仍会提示 | 跨多个文件的重构流程 |
| `acceptExec` | 在精心维护的白名单内自动批准 Shell 命令；文件写入自动批准 | 紧密的内循环，其中每条 Shell 命令均为 `npm test` 或类似操作 |
| `autoMode` | 两阶段安全分类器；被标记的操作升级至人工审查 | 受限工作区内的长周期无人值守运行 |
| `yolo` | 跳过大部分提示；仍会运行工具白名单/黑名单 | 临时沙箱、CI 任务、研究脚本 |
| `bypassPermissions` | 批准所有操作 | 官方文档注明“仅在你愿意随时销毁的临时容器内使用” |

（上述名称与公开的 Claude Code 文档一致；“yolo”是用于指代中间宽松模式的非正式简称。）

### Auto Mode 一页概览

Auto Mode（2026年3月24日发布）是首个将逐操作审批权委托给模型的模式。其结构如下：

1. **Stage 1 — single-token classifier.** 对每项提议的操作运行。成本低廉。返回“safe”或“flag”。设计为与主智能体循环并行运行，以确保获批操作不会排队等待。
2. **Stage 2 — secondary policy/safety review.** 仅在标记操作上运行。针对操作、当前会话状态及声明的任务进行聚焦的策略与安全评估。若置信度较低，则升级至用户 HITL。

预算控制与分类器并行运作：

- `max_turns` —— 会话中的总迭代次数上限。
- `max_budget_usd` —— 美元金额上限，达到后中止会话。
- 各工具的操作数量限制（例如 `WebFetch` 调用不超过 N 次等）。

### 系统能拦截的内容

- 直接注入工具输入的提示词攻击，且注入的指令映射到已知的高风险操作形态。
- 重复的工具调用循环 —— 分类器可识别出第 N+1 次操作与前一次几乎完全相同，且连续发生五次。
- 在原本仅限文件编辑的会话中，明显超出范围的 Shell 命令。

### 系统可能遗漏的内容

- **Subtle prompt injection**：在不产生任何被标记操作的情况下微妙地改变行为。间接提示词注入并非可通过补丁完全修复的漏洞（OpenAI 准备度负责人，2025年，关于浏览器智能体的观点——见第 11 课）。
- **Semantic-level misbehavior**。单个操作看起来都是安全的，但组合而成的轨迹却具有危害性。分类器仅判断单个操作，并不会重新推导用户的意图。
- **Exfiltration through legitimate channels**。将你拥有的文件写入数据，随后 `git push` 到公共仓库，这是一系列被允许的操作，其组合方式才是问题所在。

### 研究预览版的定位

Anthropic 以研究预览版的形式发布了 Auto Mode。文档明确指出，该分类器只是一个防护层，而非终极解决方案：用户需将 Auto Mode 与预算控制、白名单、隔离工作区及轨迹审计相结合（见第 12–16 课）。预览版的定位也反映了文档中提到的评估与部署之间的差距（见第 1 课）——一个在离线评估中表现良好的分类器，在实际会话中（尤其是用户上下文模糊时）可能会表现出不同的行为。

### 该权限阶梯在工作流中的应用

- 陌生任务：从 `plan` 开始。阅读计划比回滚一次失败的运行成本更低。
- 已知重构：使用 `acceptEdits` 可节省大量确认点击。
- 无人值守后台运行：仅在已测量影响范围的工作区内使用 `autoMode`（无凭据、无生产环境挂载、未主动选择加入的出站流量）。
- 临时容器：当且仅当容器及其凭据可随时销毁时，才可使用 `yolo` / `bypassPermissions`。

## 动手实践

`code/main.py` 模拟了两阶段分类器。第一阶段是对提议操作运行的低成本关键词规则；第二阶段是较慢的多规则审查器。驱动模块输入一段简短的合成轨迹（安全操作、一次提示词注入尝试、一个重复循环），并展示分类器在何处拦截成功、在何处失效。

## 交付应用

`outputs/skill-permission-mode-picker.md` 将任务描述匹配到正确的权限模式、预算上限及所需的隔离环境。

## 练习

1. 运行 `code/main.py`。哪种合成操作类型从未被 Stage 1 标记，但总是被 Stage 2 拦截？哪种类型两者都未能拦截？

2. 扩展 Stage 1 规则集以捕获特定的已知恶意形态（例如 `curl $ATTACKER/exfil`）。在良性操作样本上测量误报率。

3. 阅读 Anthropic 的《How the agent loop works》文档。列出在 `default` 模式下智能体默认会触碰的所有外部状态。在无人值守运行 `autoMode` 之前，哪些状态需要单独设置访问控制？

4. 设计一个 24 小时无人值守运行的预算方案：包含 `max_turns`、`max_budget_usd`、各工具上限及白名单。为每个数值提供合理性说明。

5. 描述一种轨迹，其中每个单独的操作都被 Stage 1 和 Stage 2 批准，但组合后的行为却存在偏差。（第 14 课介绍了如何通过 kill switches 和 canary tokens 来解决此问题。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Permission mode | “智能体能做多少事” | 七种命名策略之一，用于控制逐操作审批 |
| plan mode | “做任何事前先问” | 智能体编写计划；用户批准后方可执行 |
| acceptEdits | “让它写文件” | 文件写入自动批准；Shell 执行仍会提示 |
| autoMode | “自动批准” | 两阶段安全分类器；被标记的操作会升级审查 |
| bypassPermissions | “Full YOLO” | 批准所有操作；专为临时容器设计 |
| Stage 1 classifier | “快速 token 检查” | 对提议操作运行单 token 规则；并行执行 |
| Stage 2 classifier | “深度审查” | 对被标记的操作进行思维链推理 |
| Research preview | “尚未正式发布 (GA)” | Anthropic 对失败模式仍在探索中的功能的定位表述 |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) —— 权限模式、预算控制、操作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) —— 托管服务执行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) —— 功能概览与 Auto Mode 发布公告。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) —— 塑造分类器判断的依据层。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) —— 关于长周期权限设计的内部视角。
