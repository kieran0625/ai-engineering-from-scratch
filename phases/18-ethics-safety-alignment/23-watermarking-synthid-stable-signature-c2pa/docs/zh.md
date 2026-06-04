# 水印技术 —— SynthID、Stable Signature、C2PA

> 三项技术构成了 2026 年 AI 生成内容的溯源体系。SynthID（Google DeepMind）—— 图像水印于 2023 年 8 月发布，文本+视频于 2024 年 5 月推出（Gemini + Veo），文本部分于 2024 年 10 月通过 Responsible GenAI Toolkit 开源，统一的多媒体检测器于 2025 年 11 月随 Gemini 3 Pro 一同发布。文本水印以人眼不可察觉的方式调整下一个 token 的采样概率；图像/视频水印能够抵抗压缩、裁剪、滤镜和帧率变化。Stable Signature（Fernandez 等人，ICCV 2023，arXiv:2303.15435）—— 微调潜在扩散解码器，使每个输出都包含一条固定的消息；在仅保留 10% 内容的裁剪图像中，误报率（FPR）<1e-6 时检测率仍 >90%。后续研究《Stable Signature is Unstable》（arXiv:2405.07145，2024 年 5 月）表明，微调可以去除水印同时保持图像质量。C2PA —— 具有密码学签名且防篡改的元数据标准（C2PA 2.2 Explainer 2025）。水印与 C2PA 互为补充：元数据可能被剥离但携带更丰富的溯源信息；水印在转码过程中得以保留但携带的信息量较少。

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## 学习目标

- 描述基于 token 的水印技术（SynthID-text 风格）及其可检测机制。
- 描述 Stable Signature 以及 2024 年使其失效的移除攻击。
- 说明 C2PA 的作用及其与水印技术互补的原因。
- 描述关键局限性：模型特异性信号、同义改写下的鲁棒性不足，以及保意攻击（arXiv:2508.20228）。

## 问题背景

2023 至 2024 年间，深度伪造（deepfakes）和 AI 生成内容大规模进入政治与消费领域。水印技术被视为一种提议的技术溯源信号：在生成时进行标记，并在事后进行检测。2025 年的证据表明：没有任何水印是绝对鲁棒的，但若与 C2PA 元数据结合使用，则能提供一套可用的溯源方案。

## 核心概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等人（2023）提出的机制，已由 Google 投入生产环境：

1. 在每一步解码时，对前 K 个 token 进行哈希运算，将词表伪随机地划分为“绿色”和“红色”集合。
2. 向绿色集合对应的 logit 值加上偏移量 δ，从而在采样时偏向绿色 token。
3. 生成的文本中绿色 token 的数量会高于随机期望值。

检测方式：对每个前缀重新进行哈希计算，统计生成文本中的绿色 token 数量并计算 z-score。带水印文本的 z-score > 0，人类文本的 z-score ≈ 0。

特性：
- 对人类读者不可察觉（δ 足够小，质量损失微乎其微）。
- 只要掌握词表划分函数即可检测。
- 无法抵抗同义改写（paraphrase）—— 重写文本会破坏水印信号。

SynthID-text 已于 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez 等人，ICCV 2023。微调潜在扩散解码器，使每张生成的图像都在潜在表示中嵌入一条固定的二进制消息。检测时通过神经网络解码器从潜在空间中还原该消息。对于仅保留 10% 内容的裁剪图像，在误报率（FPR）<1e-6 的条件下检测率仍超过 90%。

2024 年 5 月《Stable Signature is Unstable》（arXiv:2405.07145）：对解码器进行微调可在保持图像质量的同时移除水印。对抗性的生成后微调成本极低，因此该水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 一同发布：一个多媒体检测器，可通过单一 API 读取文本、图像、音频和视频中的 SynthID 信号。统一了 Google 的溯源技术栈。

### C2PA

内容来源与真实性联盟（Coalition for Content Provenance and Authenticity）。一种具有密码学签名且防篡改的元数据标准。参见 C2PA 2.2 Explainer（2025）。C2PA 清单（manifest）记录由创作者密钥签名的溯源声明（包括创建者、创建时间、所经历的转换操作等）。

与水印技术互补：
- 元数据容易被剥离；水印则难以轻易去除。
- 元数据信息丰富（完整的溯源链）；水印仅携带少量比特位。
- C2PA 依赖平台方的采纳与集成；水印可自动嵌入。

Google 已在搜索、广告及“关于此图片”功能中同时集成了这两项技术。

### 局限性

- **模型特异性。** SynthID 仅对启用了 SynthID 的模型生成的内容进行水印标记。未启用 SynthID 的模型生成的内容不会带有水印，因此“无 SynthID 信号”不能作为内容真实性的证明。
- **同义改写。** 文本水印无法在保意改写（meaning-preserving paraphrase）下存活。
- **变换攻击。** arXiv:2508.20228（2025）展示了能够同时破坏文本水印和多种图像水印的保意攻击方法。
- **微调移除。** 根据《Stable Signature is Unstable》的研究，生成后的微调操作会移除已嵌入的水印。

### 欧盟《人工智能法案》第 50 条

针对 AI 生成内容标签的透明度准则（初稿发布于 2025 年 12 月，第二稿发布于 2026 年 3 月，根据[欧盟委员会状态页面](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)预计最终版将于 2026 年 6 月出台）。截至 2026 年 4 月，该准则仍处于草案阶段，时间表可能发生变化。这是要求底层技术实现的监管层。深度伪造内容必须进行标注。

### 在本阶段（Phase 18）中的定位

第 22-23 课关注模型输出了什么（隐私数据、溯源信号）。第 27 课涵盖训练数据治理。第 24 课则是要求实施这些技术措施的监管框架。

## 动手实践

`code/main.py` 构建了一个简易的文本水印示例。Token 被映射为整数 0..N-1；带水印的采样过程会偏向哈希定义的绿色集合。检测器通过计算绿色 token 的 z-score 进行判断。你可以观察 1000 token 长度生成时的检测效果，见证同义改写如何破坏信号，并测量人类文本上的误报率。

## 项目交付

本课将产出 `outputs/skill-provenance-audit.md`。给定一个带有溯源声明的内容部署方案，它将执行以下审计：水印机制（如有）、C2PA 签名链（如有）、各自的对抗鲁棒性，以及按模态划分的覆盖率。

## 练习

1. 运行 `code/main.py`。报告 1000 token 长度的带水印生成文本与人类撰写文本的 z-score。确定在 95% 置信度阈值下的误报率。

2. 实现一种同义改写攻击，将 30% 的 token 替换为近义词。重新测量 z-score。

3. 阅读 Kirchenbauer 等人（2023）第 6 节关于鲁棒性的论述。为什么文本水印在同义改写下会失效，而图像水印却能经受住裁剪？

4. 设计一个结合 SynthID-text 与 C2PA 元数据的部署方案。描述消费者端看到的溯源链。指出每个组件的一个故障模式。

5. 2024 年《Stable Signature is Unstable》的研究表明微调会移除图像水印。设计一项部署控制措施来限制此类攻击——例如，要求微调后的检查点（checkpoints）必须经过签名发布。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SynthID | “Google 的水印” | 跨模态溯源信号；覆盖文本、图像、音频、视频 |
| Token watermark | “Kirchenbauer 风格” | 基于偏置采样的文本水印，可通过绿色 token 的 z-score 检测 |
| Stable Signature | “图像水印” | 微调解码器水印；发表于 ICCV 2023 |
| C2PA | “元数据标准” | 具有密码学签名且防篡改的溯源元数据 |
| Paraphrase robustness | “改写会不会破坏它” | 文本水印的特性；目前鲁棒性有限 |
| Fine-tune removal | “对抗性去水印” | 通过对解码器进行微调来移除图像水印的攻击手段 |
| Cross-modal detector | “统一 SynthID” | 2025 年 11 月发布的跨模态统一 API |

## 延伸阅读

- [Kirchenbauer 等人 —— 大型语言模型的水印（ICML 2023，arXiv:2301.10226）](https://arxiv.org/abs/2301.10226) —— token 级水印机制
- [Fernandez 等人 —— Stable Signature（ICCV 2023，arXiv:2303.15435）](https://arxiv.org/abs/2303.15435) —— 图像水印论文
- [《Stable Signature is Unstable》（arXiv:2405.07145）](https://arxiv.org/abs/2405.07145) —— 水印移除攻击
- [Google DeepMind —— SynthID](https://deepmind.google/models/synthid/) —— 跨模态水印
- [C2PA 2.2 Explainer（2025）](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) —— 元数据标准
