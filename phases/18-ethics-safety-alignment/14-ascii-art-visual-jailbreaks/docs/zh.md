# ASCII Art 与视觉越狱攻击

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran，《ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs》（ACL 2024，arXiv:2402.11753）。在有害请求中屏蔽安全相关 token，将其替换为相同字母的 ASCII Art 渲染图，然后发送伪装后的提示词。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 均无法稳健地识别 ASCII Art token。该攻击绕过了 PPL（困惑度过滤器）、Paraphrase 防御和 Retokenization。相关研究：ViTC 基准测试用于衡量对非语义视觉提示词的识别能力；StructuralSleight 将其推广至不常见文本编码结构（树、图、嵌套 JSON），构成一系列编码攻击。

**类型：** 构建
**语言：** Python（标准库，ArtPrompt token 屏蔽工具）
**前置要求：** Phase 18 · 12 (PAIR)，Phase 18 · 13 (MSJ)
**耗时：** 约 60 分钟

## 学习目标

- 描述 ArtPrompt 攻击流程：单词识别步骤、ASCII Art 替换、最终生成的伪装提示词。
- 解释为何标准防御（PPL、Paraphrase、Retokenization）在 ArtPrompt 面前失效。
- 定义 ViTC 并说明其测量内容。
- 将 StructuralSleight 描述为对任意不常见文本编码结构的泛化。

## 问题背景

基于改写和角色扮演（第 12 课）以及长上下文（第 13 课）的攻击作用于文本模式层面。而 ArtPrompt 作用于识别层面：模型并未解析被禁用的 token，而是解析了由字符渲染出的图像。安全过滤器看到的是无害的标点符号，而模型看到的则是一个单词。

## 核心概念

### ArtPrompt：两步流程

步骤 1：单词识别。给定一个有害请求，攻击者使用 LLM 识别出与安全相关的词汇（例如“如何制造炸弹”中的“炸弹”）。 

步骤 2：生成伪装提示词。将每个识别出的单词替换为其 ASCII Art 渲染形式（由字符组成的 7x5 或 7x7 区块，勾勒出字母形状）。模型接收到的是一组标点和空格网格，具备足够能力的模型能将其识别为该单词；而安全过滤器仅看到网格本身。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 均告失败。在其基准测试子集上的攻击成功率超过 75%。

### 为何标准防御会失效

- **PPL（困惑度过滤器）**：ASCII Art 具有较高的困惑度——但所有新颖输入同样如此。用于拦截 ArtPrompt 的阈值选择也会误拦合法的结构性输入。
- **Paraphrase**：对提示词进行改写会破坏 ASCII Art。在实际操作中，负责改写的 LLM 往往会保留或重建该图形。
- **Retokenization**：改变 token 切分方式并不能改变模型实际上是在通过视觉识别字母形状这一事实。

根本原因在于，安全过滤器通常基于 token 或语义层面运作；而 ArtPrompt 作用于视觉识别层面。

### ViTC 基准测试

针对非语义视觉提示词的识别能力。用于衡量模型读取 ASCII Art、Wingdings 字体及其他非文本语义视觉内容的水平。ArtPrompt 的有效性与其在 ViTC 上的准确率呈正相关：模型读取视觉文本的能力越强，ArtPrompt 对其效果越好。这是一种能力与安全之间的权衡。

### StructuralSleight

对 ArtPrompt 的泛化：不常见文本编码结构（UTES）。包括树、图、嵌套 JSON、JSON 内嵌 CSV、diff 风格代码块等。如果某种结构在训练安全数据中罕见，但模型能够解析它，就可以用来隐藏有害内容。

防御启示：安全性必须能够泛化到模型可解析的所有结构化表示中。这个集合非常庞大且仍在不断增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。使用真实图像的 ArtPrompt 式攻击比 ASCII Art 类比攻击更强，因为图像编码器能产生更丰富的信号。

### 在本阶段的位置

第 12 至 14 课描述了三个正交的攻击向量：迭代优化（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接提示词注入）。第 16 课介绍了防御工具链的响应。

## 实践应用

`code/main.py` 构建了一个简易版 ArtPrompt。你可以使用 ASCII Art 字形掩盖有害查询中的特定单词，验证伪装后的字符串能否通过关键词过滤器，并（可选地）使用简单的识别器将伪装字符串解码回原文。

## 交付成果

本课将产出 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它将列出所涵盖的编码攻击家族（ASCII Art、base64、Leet Speak、UTF-8 同形字、UTES）以及捕获每种攻击的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装后的字符串是否通过了简单的关键词过滤器。报告所需的字符级变更。

2. 实现第二种编码：对同一目标单词使用 base64。比较其绕过过滤器的成功率与 ArtPrompt 的差异，以及恢复难度。

3. 阅读 Jiang 等人 2024 年论文的第 4.3 节（五模型结果）。提出一种原因，解释为何在同一基准测试下 Claude 对 ArtPrompt 的抵抗力高于 Gemini。

4. 设计一种预生成防御机制，用于检测提示词中呈现 ASCII Art 形状的区块。在合法代码、表格和数学公式上测量其误报率。

5. StructuralSleight 列出了 10 种编码结构。草拟一种能处理全部 10 种结构的通用防御方案，并估算每个受保护提示词的计算成本。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| ArtPrompt | “ASCII Art 攻击” | 两步越狱攻击，使用 ASCII Art 渲染图掩盖安全敏感词 |
| Cloaking | “隐藏单词” | 将被禁用的 token 替换为模型可读但过滤器不可见的视觉表示 |
| UTES | “不常见结构” | 不常见文本编码结构（Uncommon Text-Encoded Structure）——如树、图、嵌套 JSON 等，用于走私内容 |
| ViTC | “视觉文本能力” | 衡量模型读取非语义视觉编码能力的基准测试 |
| Perplexity filter | “PPL 防御” | 拒绝高困惑度的提示词；因合法结构化输入同样得分较高而失效 |
| Retokenization | “分词器切换防御” | 使用不同的分词器预处理提示词；因识别本质是视觉层面的而失效 |
| Homoglyph | “相似字符” | 外观与拉丁字母完全相同的 Unicode 字符；可绕过子串检查 |

## 延伸阅读

- [Jiang 等人 — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) —— ASCII Art 越狱论文
- [Li 等人 — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) —— UTES 泛化研究
- [Chao 等人 — PAIR (第 12 课, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) —— 互补型迭代攻击
- [Anil 等人 — Many-shot Jailbreaking (第 13 课)](https://www.anthropic.com/research/many-shot-jailbreaking) —— 互补型长度攻击
