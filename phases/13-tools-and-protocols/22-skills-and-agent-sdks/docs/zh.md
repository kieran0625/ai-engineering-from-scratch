# Skills 与 Agent SDK —— Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说明“有哪些工具可用”。Skills 说明“如何完成任务”。2026 年的技术栈将两者分层叠加。Anthropic 的 Agent Skills（开放标准，2025年12月）以 SKILL.md 形式发布，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加上组件元数据。AGENTS.md（现已出现在 60,000 多个仓库中）位于仓库根目录，作为项目级智能体上下文。本课将明确各自覆盖的范围，并构建一个可在不同智能体间跨平台运行的最小化 SKILL.md + AGENTS.md 包。

**类型：** 学习
**语言：** Python（标准库、SKILL.md 解析器与加载器）
**前置要求：** 阶段 13 · 07（MCP 服务器）
**预计时间：** 约 45 分钟

## 学习目标

- 区分三个层级：AGENTS.md（项目上下文）、SKILL.md（可复用经验）、MCP（工具）。
- 编写带有 YAML frontmatter 和渐进式披露功能的 SKILL.md。
- 以文件系统风格将技能加载到智能体运行时环境中。
- 将技能与 MCP 服务器及 AGENTS.md 组合，使单个包能在 Claude Code、Cursor 和 Codex 中通用。

## 问题背景

工程师将一个撰写版本更新日志的工作流提炼为多步提示词：“读取最新合并的 PR。按领域分组。总结每个 PR。按照团队风格撰写变更日志条目。发布到 Slack 草稿。”他们将其放入团队的 Notion 文档中。

现在他们希望从 Claude Code、Cursor 和 Codex CLI 中使用此工作流。每个智能体加载指令的方式各不相同：Claude Code 使用斜杠命令，Cursor 使用规则，Codex 使用 `.codex.md`。工程师不得不复制三次该工作流并维护三份副本。

AGENTS.md 和 SKILL.md 共同解决了这一问题：

- **AGENTS.md** 位于仓库根目录。所有兼容的智能体在会话启动时都会读取它。“这个项目是如何运作的？遵循什么规范？哪些命令用于运行测试？”
- **SKILL.md** 是一个可移植的包：包含 YAML frontmatter（名称、描述）+ Markdown 正文 + 可选资源。支持技能的智能体会按需通过名称加载它们。
- **MCP**（阶段 13 · 06-14）负责处理技能需要调用的工具。

三个层级，一个可移植的工件。

## 核心概念

### AGENTS.md (agents.md)

于 2025 年底推出，至 2026 年 4 月已被 60,000 多个仓库采用。仓库根目录下仅一个文件。格式如下：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

智能体在会话启动时读取此文件，并据此校准其在该项目中的行为。2026 年的所有代码智能体均支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（作为开放标准于 2025 年 12 月发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明了技能的标识信息。正文部分是在技能加载时展示给模型的提示词。

### 渐进式披露

技能可以引用子资源，智能体仅在需要时才会获取这些资源。示例如下：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 中写明“样式规则请参阅 style-guide.md”。智能体仅在技能实际运行时才拉取 style-guide.md。这避免了用模型可能不需要的细节导致提示词冗长。

### 文件系统发现

智能体运行时环境会扫描已知目录以查找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目级 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载依据文件夹名称和 frontmatter 中的 `name` 进行。Claude Code、Anthropic Claude Agent SDK 以及 SkillKit（跨智能体）均遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话启动时加载技能，并将其作为运行时内部可调用的“智能体”暴露出来。当用户调用某个技能时，智能体循环会将其分派给对应的技能执行。

### OpenAI Apps SDK

于 2025 年 10 月推出；直接构建于 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个开发者界面下。Apps SDK 应用包含：

- 一个 MCP 服务器（工具、资源、提示词）。
- 附加 ChatGPT UI 所需的组件元数据。
- 附加可选的 MCP Apps `ui://` 资源，用于交互式界面。

相同的协议，更丰富的用户体验。

### 通过 SkillKit 实现跨智能体可移植性

借助 SkillKit 等类似跨智能体分发层的工具，可将单个 SKILL.md 转换为 32+ 种 AI 智能体（如 Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。单一事实来源；多方消费。

### 三层架构

| 层级 | 文件 | 加载时机 | 用途 |
|------|------|----------|------|
| AGENTS.md | 仓库根目录 | 会话启动时 | 项目级规范 |
| SKILL.md | 技能目录 | 技能被调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用操作 |

三者可组合使用：智能体在会话启动时读取 AGENTS.md，用户调用某个技能，该技能的指令中包含 MCP 工具调用，智能体通过 MCP 客户端进行分派执行。

## 动手实践

`code/main.py` 提供了一个基于标准库的 SKILL.md 解析器与加载器。它会扫描 `./skills/` 下的技能，解析 YAML frontmatter 与 Markdown 正文，并生成以技能名称为键的字典。随后模拟一个智能体循环，按名称调用 `release-notes-writer`。

重点关注：

- 使用极简标准库解析器解析 YAML frontmatter（无需依赖 `pyyaml`）。
- 技能正文原样存储；智能体在调用时会将其前置到系统提示词中。
- 通过 `read_subresource` 函数演示渐进式披露功能，按需拉取引用的文件。

## 最终产出

本课将产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能会生成整合后的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，实现跨智能体可移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，并确认加载器能正确捕获。

2. 为本课程仓库编写一份 AGENTS.md。包含测试命令、样式规范以及阶段 13 的心智模型。

3. 将团队内部文档中的多步工作流迁移至 SKILL.md。验证其能否在 Claude Code 中成功加载。

4. 手动将该技能转换为 Cursor 和 Codex 的原生规则格式。统计格式间的差异数量——这正是 SkillKit 自动化的转换范围。

5. 阅读 Anthropic Agent Skills 博客文章。指出 Claude Agent SDK 中本课加载器未涵盖的一项功能。（提示：智能体子调用。）

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SKILL.md | “技能文件” | 包含 YAML frontmatter 与 Markdown 正文，由智能体运行时加载 |
| AGENTS.md | “仓库根目录智能体上下文” | 项目级规范文件，在会话启动时读取 |
| Progressive disclosure | “懒加载子资源” | 技能正文引用文件，仅在需要时拉取 |
| Frontmatter | “顶部的 YAML 块” | 位于 `---` 分隔符内的元数据（名称、描述） |
| Claude Agent SDK | “Anthropic 的技能运行时” | `@anthropic-ai/claude-agent-sdk`，负责加载技能并进行路由 |
| OpenAI Apps SDK | “MCP + 组件元数据” | 基于 MCP 构建并集成 ChatGPT UI 钩子的 OpenAI 开发者界面 |
| Skill discovery | “文件系统扫描” | 遍历已知目录查找 SKILL.md，以名称为键索引 |
| Cross-agent portability | “一个技能适配多个智能体” | 通过类似 SkillKit 的工具将单个 SKILL.md 转换为 32+ 种智能体的格式 |
| Agent Skill | “可移植的经验知识” | 独立于 MCP 工具概念的可复用任务模板 |
| Apps SDK | “MCP 加 ChatGPT UI” | 基于 MCP 统一的 Connectors 与 Custom GPTs |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) —— 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) —— SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) —— 面向 ChatGPT 的基于 MCP 的开发者平台
- [agents.md](https://agents.md/) —— AGENTS.md 格式与采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) —— 官方技能示例
