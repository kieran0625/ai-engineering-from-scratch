# 毕业设计 09 —— 代码迁移代理（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移器树立了 2026 年的标杆。Moderne 的 OpenRewrite 能够大规模执行确定性的 AST 重写。Grit 使用类 codemod 的 DSL 解决同一问题。生产环境模式将两者结合：用于安全重写的确定性底层框架，加上处理模糊情况的代理层，以及支持按分支构建的沙箱，还有在 PR 打开前就能确保测试通过的测试套件。本毕业设计的目标是迁移 50 个真实仓库，并发布通过率与失败分类学。

**类型：** 毕业设计
**语言：** Python（代理）、Java / Python（目标）、TypeScript（仪表盘）
**前置要求：** 第 5 阶段（NLP）、第 7 阶段（transformers）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（代理）、第 15 阶段（自主）、第 17 阶段（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**耗时：** 30 小时

## 问题

大规模代码迁移是 2026 年编码代理最清晰的生产应用场景之一。其“地面真值”显而易见（迁移后测试套件是否通过？），回报切实可行（一次 Java-8 集群迁移相当于一个百人规模的项目），且基准测试公开（MigrationBench 50 个仓库子集）。Moderne 的 OpenRewrite 负责处理确定性部分。代理层则处理 OpenRewrite 规则无法覆盖的所有内容：模糊的重写场景、构建系统版本漂移、长尾语法以及传递依赖破坏。

你将构建一个代理，接收一个 Java 8 仓库（或 Python 2 仓库），并生成一个 CI 全绿的迁移分支。你需要测量通过率、测试覆盖率保留率、每个仓库的成本，并构建失败分类学。将其与仅使用确定性规则的基线进行对比，可以明确代理的实际价值所在。

## 概念

该流水线包含两层。**确定性底层框架**（Java 使用 OpenRewrite，Python 使用 libcst）安全地执行大部分机械性重写：导入语句、方法签名、空安全修改、try-with-resources 转换、废弃 API 替换。它速度快且能生成可审计的差异（diff）。**代理层**（基于 Claude Opus 4.7 和 GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph）处理规则无法覆盖的情况：构建文件升级（Maven/Gradle/pyproject.toml）、传递依赖冲突、测试不稳定（flakes）、自定义注解。

每个仓库都会分配一个预装了目标运行时的 Daytona 沙箱。代理会迭代执行：运行构建、分类失败原因、应用修复、重新运行。硬性限制为：每个仓库 30 分钟、成本不超过 $8、最多 20 次代理交互轮次。如果所有测试通过且覆盖率差值不为负，则分支会创建 PR。否则，该仓库将根据证据归入某个失败类别。

失败分类学是本次交付的核心成果。在 50 个仓库中，究竟哪些环节出了问题？传递依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类别都会有统计数量和典型差异示例。未来的规则作者可以针对排名前三的问题进行优化。

## 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技术栈

- 确定性底层框架：OpenRewrite（Java）或 libcst（Python）
- 代理：基于 Claude Opus 4.7 + GPT-5.4-Codex 的 OpenAI Agents SDK 或 LangGraph
- 沙箱：按分支划分的 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准测试：Amazon MigrationBench 50 个仓库子集（Java 8 到 17）、Google App Engine Py2-to-Py3 仓库
- 测试套件：并行运行器，覆盖率统计通过 Jacoco（Java）或 coverage.py（Python）实现
- 可观测性：Langfuse + 每个仓库附带每次 diff 块的追踪包
- 仪表盘：失败分类学仪表盘，展示各类别数量及典型差异示例

## 构建步骤

1. **规则执行阶段**。首先运行 OpenRewrite（Java）或 libcst（Python）规则。捕获 70%-80% 属于机械性操作的迁移。提交为“rule”提交。

2. **构建尝试**。Daytona 沙箱：安装目标运行时，运行构建。如果通过（绿色），跳过至测试阶段；如果失败（红色），移交代理处理。

3. **代理循环**。使用 LangGraph 配合工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。代理对失败进行分类（依赖、语法、测试、构建工具）并应用针对性修复。然后重新运行。

4. **预算上限**。每个仓库实际耗时 30 分钟，成本 $8，最多 20 次代理交互轮次。任何超限情况都将停止执行，并将当前差异归档至“budget_exhausted”类别。

5. **测试与覆盖率门禁**。构建通过后，运行测试套件。将覆盖率与基础仓库对比。如果覆盖率下降超过 2%，则归档至“coverage_regression”类别。

6. **创建 PR**。成功时，推送分支，创建 PR，附上差异说明以及已应用的规则和代理生成的提交摘要。

7. **失败分类学**。为每个失败的仓库打上类别标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50 仓库运行**。在 MigrationBench 子集上执行。报告各类别的通过率、单仓库成本、覆盖率保留率，并与仅使用确定性规则的基线进行对比。

## 使用方式

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付物

`outputs/skill-migration-agent.md` 是本次交付的核心组件。给定一个仓库，它会先执行确定性规则，随后进入代理循环，以生成 CI 全绿的迁移分支，或将该仓库归档至某个分类学类别。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 个仓库子集的 pass@1 |
| 20 | 测试覆盖率保留率 | 相对于基础仓库的平均覆盖率差值 |
| 20 | 迁移后单仓库成本 | 成功运行的 $/repo |
| 20 | 代理/确定性工具集成度 | OpenRewrite 处理的修复比例 vs 代理编写的修复比例 |
| 15 | 失败分析文档 | 分类学的完整性及典型示例 |
| **100** | | |

## 练习

1. 仅使用 OpenRewrite 运行迁移流水线（无代理）。将通过率与完整流水线对比。找出仅靠代理才能解决的案例。

2. 实现“代码风格检查”：迁移后运行风格检查工具（Java 使用 spotless，Python 使用 ruff）。如果出现新的 lint 错误，则拒绝 PR。统计覆盖率保留但代码风格退化的比例。

3. 添加“最小差异”优化器：代理分支通过测试后，进行第二轮扫描以剔除不必要的变更。报告差异体积的缩减比例。

4. 扩展至第三种迁移场景：Node 18 到 Node 22。复用沙箱封装；将规则层替换为自定义 codemod。

5. 将首次构建通过时间（TTFGB）作为用户体验指标进行测量。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | “规则引擎” | OpenRewrite / libcst：带有安全性保证的声明式 AST 重写 |
| Codemod | “代码修改程序” | 一种机械性更改源代码的重写规则 |
| Build drift | “工具版本偏差” | 主版本之间 Maven / Gradle / uv 行为的细微变化 |
| Failure class | “分类桶” | 标记仓库未迁移的原因：依赖、语法、测试、构建工具、预算 |
| Coverage delta | “覆盖率保留” | 从基础仓库到迁移分支的测试覆盖率百分比变化 |
| Agent turn | “工具调用轮次” | 代理循环中的一次“规划 -> 执行 -> 观察”周期 |
| Budget exhaustion | “触及上限” | 仓库在未通过的情况下耗尽了 30 分钟 / $8 / 20 轮次的限制 |

## 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) —— 2026 年的权威基准
- [Moderne.io OpenRewrite platform](https://www.moderne.io) —— 确定性底层框架参考
- [OpenRewrite documentation](https://docs.openrewrite.org) —— 规则编写指南
- [Grit.io](https://www.grit.io) —— 替代型 codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) —— Agents SDK 参考
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) —— 替代型迁移基准
- [libcst](https://github.com/Instagram/LibCST) —— Python 确定性底层框架
- [Daytona sandboxes](https://daytona.io) —— 参考级按分支沙箱
