# 基准测试：WebArena 与 OSWorld

> WebArena 在四个自托管应用中测试网页代理（web-agent）的能力。OSWorld 在 Ubuntu、Windows 和 macOS 上测试桌面代理（desktop-agent）的能力。在发布时（2023–2024），两者都显示出顶级代理与人类之间存在巨大差距。这一差距正在缩小，但失败模式并未改变。

**类型：** 学习
**语言：** Python (stdlib)
**前置要求：** 第 14 阶段 · 第 19 阶段（SWE-bench, GAIA）
**时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用，并说明基于执行的评估为何重要。
- 解释为什么 OSWorld 使用真实的操作系统截图，而不是辅助功能 API。
- 指出 OSWorld 的两个主要失败模式：GUI 定位（grounding）和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准之上增加了什么内容。

## 问题背景

通用代理能够调用工具。但它们能否通过 20 次点击驱动浏览器完成购物结账？能否仅使用键盘和鼠标配置一台 Linux 机器？这正是 WebArena 和 OSWorld 所要回答的问题。

## 核心概念

### WebArena (Zhou et al., ICLR 2024)

- 涵盖四个自托管 Web 应用的 812 个长周期任务：一个购物网站、一个论坛、一个类 GitLab 的开发工具、一个企业 CMS。
- 附加实用工具：地图、计算器、草稿纸。
- 评估基于执行结果，通过 gym API 进行——订单是否已提交？问题是否已关闭？CMS 页面是否已更新？
- 发布时：表现最好的 GPT-4 代理成功率为 14.41%，而人类为 78.24%。

自托管的设定至关重要——该基准测试不会出现不稳定的测试结果（flaky），因为目标应用版本固定且可复现。

### 扩展项目

- **VisualWebArena** —— 视觉定位任务，其成功取决于对图像的解释（将截图作为一等观测数据）。
- **TheAgentCompany**（2024年12月）—— 增加终端操作与代码编写；更贴近真实的远程办公环境。

### OSWorld (Xie et al., NeurIPS 2024)

- 覆盖 Ubuntu、Windows、macOS 的 369 个真实计算机任务。
- 自由形式的键盘和鼠标控制真实应用程序。
- 以 1920×1080 分辨率的截图作为观测输入。
- 发布时：表现最好的模型成功率为 12.24%，而人类为 72.36%。

### 主要失败模式

1. **GUI 定位（Grounding）**。像素到元素的映射。模型难以在 1920×1080 分辨率下可靠地定位 UI 元素。
2. **操作知识**。哪个菜单包含该设置、哪个键盘快捷键、哪个偏好设置面板。这是人类多年积累的知识长尾。

### 后续进展

- **OSWorld-G** —— 包含 564 个样本的定位套件及 Jedi 训练集。将定位与规划解耦，以便分别测量。
- **OSWorld-Human** —— 人工筛选的黄金动作轨迹。显示顶级代理使用的步骤数比必要值多出 1.4 至 2.7 倍（即轨迹效率差距）。

### 为什么这很重要

Claude Computer Use、OpenAI CUA、Gemini 2.5 Computer Use（见第 21 课）均在受 WebArena 和 OSWorld 塑造的工作负载上进行训练。这些基准测试是目标，而生产级模型则是交付的答案。

### 基准测试中的常见误区

- **仅依赖截图的评估**。OSWorld 由截图驱动；在 OSWorld 上评估使用 DOM 或辅助功能 API 的代理，会忽略定位挑战。
- **忽视轨迹长度**。仅评分成功率会掩盖 OSWorld-Human 所揭示的 1.4 至 2.7 倍步骤低效问题。
- **过时的自托管应用**。WebArena 的应用固定了特定版本；若未经重新筛选直接更新，会破坏可比性。

## 动手实践

`code/main.py` 实现了一个玩具级网页代理框架：

- 一个极简的“购物应用”状态机：list_items, add_to_cart, checkout。
- 3 个任务的黄金轨迹。
- 一个尝试执行每个任务的脚本化代理。
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤数对比黄金轨迹）。

运行它：

```
python3 code/main.py
```

输出：每个任务的成功率与轨迹效率，镜像 OSWorld-Human 的方法论。

## 应用场景

- **WebArena Verified**：部署在内部集群上的自托管版本，用于持续评估。
- **OSWorld**：在虚拟机集群中运行，用于桌面代理测试。
- **计算机使用代理**（第 21 课）—— Claude、OpenAI CUA、Gemini —— 均在类似的工作负载上进行训练。
- **你自己的产品流程** —— 捕获前 20 个核心任务的黄金轨迹；每周让代理针对这些轨迹运行测试。

## 交付指南

`outputs/skill-web-desktop-harness.md` 构建了一个支持基于执行评估和轨迹效率指标的网页/桌面代理框架。

## 练习

1. 在玩具框架中添加第二个应用（论坛）。编写 3 个任务及其黄金轨迹。
2. 添加按任务划分的轨迹效率报告。在你的玩具框架中，代理的步骤数是黄金轨迹的 1 倍、2 倍还是 3 倍？
3. 实现一个“干扰”工具——即黄金轨迹从未使用过的工具。脚本化代理会被诱惑去使用它吗？
4. 阅读 OSWorld-G 的相关资料。你会如何在自己的评估中将定位失败与规划失败区分开来？
5. 阅读 WebArena 的应用 README。当你升级其中一个固定版本的应用时，会发生什么破坏？

## 关键术语

| 术语 | 通常的说法 | 实际含义 |
|------|------------|----------|
| WebArena | “网页代理基准测试” | 覆盖 4 个自托管应用的 812 个任务；采用 gym 风格的评估 |
| VisualWebArena | “视觉版 WebArena” | 视觉定位版 WebArena；将截图作为观测输入 |
| OSWorld | “桌面代理基准测试” | 在真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI grounding | “像素到元素映射” | 模型在 1920x1080 分辨率下定位 UI 元素 |
| Operational knowledge | “操作系统经验” | 知道哪个菜单、哪个快捷键、哪个偏好设置面板 |
| OSWorld-G | “定位套件” | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | “黄金轨迹” | 人工记录专家动作序列，用于衡量效率 |
| Trajectory efficiency | “超出黄金轨迹的步骤数” | 代理步数除以人类最少步数 |

## 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) —— 四应用网页基准测试
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) —— 跨操作系统桌面基准测试
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) —— 受基准测试塑造的 Claude 能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) —— OSWorld 与 WebArena 的数据
