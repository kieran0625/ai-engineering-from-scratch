# 生产环境中的 EAGLE-3 投机解码

> 投机解码将快速的草稿模型与目标模型配对。草稿模型提出 K 个 token；目标模型在一次前向传播中进行验证；被接受的 token 相当于免费。在 2026 年，EAGLE-3 是面向生产的版本——它在目标模型的隐藏状态上训练草稿头（draft head），而不是在原始 token 上，从而将接受率 alpha 推升至通用对话场景下的 0.6-0.8 区间。正确的问题不是“草稿有多快”，而是“在我的流量下 alpha 是多少？”如果 alpha 低于 ~0.55，在高并发下投机解码的净收益为负，因为每个被拒绝的草稿都会导致一次额外的目标模型前向传播。本课程教你先测量 alpha，再决定是否开启该功能。

**类型：** 学习
**语言：** Python（标准库、简易接受率模拟器）
**前置知识：** 阶段 17 · 04（vLLM 服务内部原理）、阶段 10 · 18（多 Token 预测）
**耗时：** 约 60 分钟

## 学习目标

- 说出投机解码的三个代际，并解释 EAGLE-3 相较于 EAGLE-2 和经典草稿模型做出了哪些改变。
- 定义接受率 alpha，根据 alpha 和 K（草稿长度）计算预期加速比，并找出适合你目标并发的临界 alpha 值。
- 解释为什么在 2026 版的 vLLM 中投机解码是可选开启（opt-in）而非默认开启的，以及为什么在未测量 alpha 的情况下盲目开启它属于生产环境的反模式。
- 编写测量方案：确定使用哪个基准测试、哪种提示词分布、哪个并发点、以哪个指标作为放行条件。

## 问题背景

解码过程受内存带宽限制。在运行 Llama 3.3 70B FP8 的 H100 上，每解码一个 token 需要读取约 140 GB/s 的权重数据，并输出一个 token。GPU 计算单元在解码期间几乎处于空闲状态——瓶颈在于 HBM 带宽，而非矩阵乘法吞吐量。

投机解码利用了这一差距。使用廉价的草稿模型生成 K 个候选 token，然后让目标模型在一次前向传播中验证所有 K 个 token。每个被验证通过的 token 实际上相当于免费的（其成本分摊到目标模型无论如何都必须执行的一次 batch-of-K 前向传播中）。

经典的草稿模型方法使用同家族的小型模型（例如用 Llama 3.2 1B 为 Llama 3.3 70B 起草）。这种方法有效，但接受率表现平平——小型模型的分布与目标模型存在偏差。EAGLE、随后是 EAGLE-2，再到 EAGLE-3，直接在目标模型的内部状态上训练轻量级的草稿头，使得草稿的分布能更紧密地跟踪目标模型。这就是为什么 alpha 从草稿模型的 0.4 提升到了 EAGLE-3 的 0.6-0.8。

关键点：在 2026 版的 vLLM 中，EAGLE-3 是可选开启的。必须显式设置 `speculative_config`。没有该标志位，就没有加速效果。许多团队在未测量真实流量下 alpha 的情况下盲目开启它，结果往往发现尾部延迟反而恶化了。

## 核心概念

### 投机解码实际带来的收益

不开启投机解码时，每个 token 的成本是一次目标模型前向传播。开启投机解码且草稿长度为 K、接受率为 alpha 时，每次目标前向传播的预期 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的额外开销。对于 K=5，alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数字通常集中在 2-3 倍左右，因为生产流量的 alpha 很少那么高，且 epsilon 会随批量大小增加而增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 并不会消失——它们会迫使对第一个被拒绝的 token 执行第二次目标前向传播。在 alpha 降至 0.4 的工作负载下，你需要支付草稿开销加上验证开销再加上重新生成的开销。在高并发下（例如 256 并发），解码批次已经足够大，以至于“仅目标模型”和“带验证的目标模型”之间的内存带宽差距缩小了。在大多数 2026 年的硬件上，当 alpha 低于 0.55 时，投机解码的净收益为负。

alpha 因工作负载而异。在类似 ShareGPT 的通用对话中，在 ShareGPT 上训练的 EAGLE-3 能达到 0.6-0.8 的 alpha。在领域特定的流量（代码、医疗、法律）上，基于通用数据训练的草稿头 alpha 会降至 0.4-0.6。训练领域特定的草稿头可以恢复 alpha——与目标模型微调相比，这是一项轻量且快速的训练任务。

### EAGLE 代际一览

- **经典草稿模型**：同家族的小型模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，草稿模型每次目标前向传播运行 K 次。
- **EAGLE-1 (2024)**：在目标隐藏状态（最后一层）上训练的单个草稿头。Alpha 约 0.5-0.6。在目标模型之上仅有少量的参数开销。
- **EAGLE-2 (2025)**：自适应草稿长度和基于树的草稿（在一次目标传递中验证多个分支）。Alpha 约 0.6-0.7。草稿调度器更复杂。
- **EAGLE-3 (2025-2026)**：在多个目标层（不仅是最后一层）上训练的草稿头，对齐效果更好。在通用对话中 Alpha 约 0.6-0.8。

### 2026 生产环境部署步骤

1. 纯部署目标模型。在目标并发度下测量基线 TTFT、ITL 和吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行基准测试。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度即可得到 alpha。
4. 如果在生产流量分布下 alpha < 0.55，请禁用投机解码或训练领域特定的 EAGLE-3 草稿头。
5. 在生产并发度下重新运行。确认 P99 ITL 没有恶化。

### 生产环境陷阱：P99 尾部延迟

开启投机解码后平均 ITL 会下降。如果不进行调优，P99 可能会恶化。被拒绝的草稿会触发两遍序列（草稿 + 验证失败 + 重新生成）。在满批处理情况下，这两遍操作会串行执行。请密切关注 P99 ITL，而不是 P50。

### EAGLE-3 的现有部署案例

Google 于 2025 年在 AI Overviews 中部署了投机解码（质量相同，响应更快）。vLLM V1 提供 `speculative_config` 作为文档记录的接口；V1 中的 N-gram GPU 投机解码是与分块预填充（chunked prefill）兼容的变体。SGLang 支持 EAGLE-3 作为前缀密集型工作负载的推荐草稿路径。

### 一行公式：盈亏平衡计算

预期加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 可解出 alpha：`alpha_breakeven = verify_overhead / K`。对于典型的 verify_overhead ~0.15 和 K=5：`alpha_breakeven = 0.03`。但这只是原始解码数学。在高并发下，验证开销会增加，且解码批次已经在不同序列间分摊了内存读取，因此实际有效的 alpha_breakeven 会攀升至约 0.45-0.55。

### 何时不应使用投机解码

- 延迟不重要的 Batch-1 离线生成。直接使用目标模型。
- 极短的输出（少于 50 个 token）。草稿开销和验证成本占主导。
- 没有经过领域训练的草稿头的专业领域。Alpha 过低。
- vLLM v0.18.0 加上草稿模型投机解码再加上 `--enable-chunked-prefill`。此组合无法编译。文档记录的例外情况是 V1 中的 N-gram GPU 投机解码。

## 动手实践

`code/main.py` 模拟了一个在不同 alpha 值和草稿长度 K 下有无投机解码的解码循环。它会打印盈亏平衡 alpha、实测加速比和尾部行为。在多个 (alpha, K) 组合上运行它，以精确观察投机解码在何处不再产生收益。

## 交付成果

本课程将产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，它将生成一个分阶段的 EAGLE-3  rollout 计划——基准测试基线、启用配置、测量 alpha、以 alpha >= 0.55 为放行条件、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，实现 2 倍加速需要多少 alpha？3 倍加速呢？这对 verify_overhead 的敏感度如何？
2. 假设生产流量分为 70% 通用对话和 30% 代码。在 ShareGPT 上训练的 EAGLE-3 在通用对话上 alpha 为 0.7；代码上 alpha 为 0.4。混合 alpha 是多少？投机解码是否净正收益？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram），并指出哪一种与分块预填充兼容。
4. 开启 EAGLE-3 后，你发现平均 ITL 下降了 25%，但 P99 ITL 上升了 15%。请诊断原因并提出缓解措施。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的内存成本。它与将 Llama 3.2 1B 作为经典草稿模型运行的成本相比如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 投机解码 | “草稿加验证” | 使用廉价模型提出 K 个 token，在一次目标前向传播中验证所有 K 个 |
| 接受率 alpha | “spec accept rate” | 目标模型接受的草稿 token 比例；唯一重要的指标 |
| 草稿长度 K | “spec k” | 每次目标前向传播草稿提出的 token 数量；通常为 4-8 |
| 验证开销 epsilon | “spec overhead” | 验证并重试相对于纯目标前向传播的额外成本；随批次增大而增长 |
| EAGLE-3 | “最新 EAGLE” | 2025-2026 版本；在多个目标层上训练草稿头；通用对话中 alpha 0.6-0.8 |
| `speculative_config` | “vLLM spec config” | vLLM V1 中的显式可选开启项；无默认值意味着无加速 |
| N-gram 投机解码 | “N-gram draft” | 基于提示词中 N-gram 查找的 GPU 端草稿；兼容分块预填充 |
| 盈亏平衡 alpha | “no-op alpha” | 投机解码加速比为 1 时的 alpha 值；在生产并发下需重点关注 |
| 拒绝草稿双遍 | “reroll cost” | 草稿被拒时执行两次目标前向传播；导致 P99 尾部延迟升高 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) —— 关于 V1 中 `speculative_config` 及分块预填充兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) —— 确切的字段集合。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— 原始 EAGLE 草稿头公式。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 自适应草稿与树结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) —— 带有投机解码的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) —— 生产环境部署清单。
