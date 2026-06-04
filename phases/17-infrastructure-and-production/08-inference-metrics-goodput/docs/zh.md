# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定了一个推理部署是否有效。TTFT 是预填充（prefill）加上排队延迟和网络传输时间。TPOT（等同于 ITL）是每个 token 的显存受限解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐量是整个集群聚合后的每秒 token 数。但对产品真正重要的是 goodput —— 同时满足所有 SLO 的请求比例。低 goodput 下的高吞吐量意味着你正在处理那些无法及时送达用户的 token。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数值：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。务必报告 P50、P90、P99 —— 绝不要只报平均值。还要警惕测量陷阱：GenAI-Perf 在计算 ITL 时排除了 TTFT，而 LLMPerf 则包含它；两个工具对同一次运行的 TPOT 结果不一致。

**类型：** 学习
**语言：** Python（标准库、简易百分位计算器与 goodput 报告器）
**前置知识：** 第 17 阶段 · 04（vLLM 服务内部机制）
**耗时：** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量与 goodput，并指出每个指标所衡量的具体组件。
- 解释为什么平均值不适用于 LLM 服务统计，以及如何正确解读 P50/P90/P99。
- 构建 SLO 多约束条件（例如 TTFT < 500 ms 且 TPOT < 15 ms 且 E2E < 2 s），并据此计算 goodput。
- 列举两种在同次运行中对 TPOT 结果不一致的基准测试工具，并解释原因。

## 问题所在

“我们的吞吐量是每秒 15,000 个 token。”那又怎样？如果有 40% 的请求端到端延迟超过了 2 秒，用户就会放弃会话。仅凭吞吐量无法告诉你产品是否可用。

推理过程存在多个延迟维度，且各自的故障模式不同。预填充（Prefill）受计算能力限制，随提示词长度扩展。解码（Decode）受显存带宽限制，随批处理大小扩展。排队延迟是运维调度问题。网络延迟是物理距离问题。你需要为每个维度设置独立的指标，需要百分位数，还需要一个能回答“用户是否得到了预期体验”的综合指标 —— 这就是 goodput。

## 核心概念

### TTFT —— 首字生成时间（Time To First Token）

`TTFT = queue_time + network_request + prefill_time`

当提示词较长时，预填充占主导。在 H100 上运行 Llama-3.3-70B FP8 模型时，32k 长度的提示词仅纯预填充就需要约 800 ms。排队时间反映的是高负载下的调度器行为。网络请求时间包含 TLS 握手等在内的线路传输时间。TTFT 是用户在收到任何流式输出之前所感知的延迟。

### TPOT / ITL —— 词元间延迟（Inter-Token Latency）

同一个量有不同的称呼。`TPOT`（每个输出 token 的时间）、`ITL`（词元间延迟）、`decode latency per token` —— 本质相同。它指的是首个 token 之后，连续流式输出 token 之间的时间间隔。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一套搭载分块预填充（chunked prefill）的 Llama-3.3-70B H100 环境中，TPOT 均值约为 7 ms。若未启用分块预填充，当相邻序列进行长预填充时，TPOT 可能飙升至 50 ms。请重点关注 P99，而非平均值。

### E2E 延迟（端到端延迟）

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 个 token），E2E 主要由 TPOT 主导。对于短输出但提示词较长的情况，E2E 主要由 TTFT 主导。应按输出长度条件分别报告 E2E。

### 吞吐量（Throughput）

`throughput = total_output_tokens / elapsed_time`

聚合型指标。反映整个集群的效率。无法反映单个请求的健康状况。

### Goodput —— 你真正应该关注的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多约束条件。只有当所有约束均被满足时，该请求才算“合格”。Goodput 即为合格请求的比例。60% goodput 下的高吞吐量属于失败；99% goodput 下的较低吞吐量才是目标。

在 2026 年，goodput 已被用于 MLPerf Inference v6.0 的提交数据，以及各大 AI 平台提供商的内部 SLA 追踪中。

### 为什么平均值是错误的统计方式

LLM 延迟分布呈右偏态。一个解码批次中若有一个长预填充的邻居请求，可能导致 500 个 token 以 ~7 ms 的 TPOT 发出，而另外 20 个 token 的 TPOT 高达 ~60 ms。平均 TPOT 为 9 ms，但 P99 TPOT 为 65 ms。用户经常遭遇 P99 级别的延迟 —— 这正是他们流失的原因。

务必始终报告三组数据（P50、P90、P99）。针对用户体验优化时，应聚焦 P99。

### 参考数值 —— 2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：根据分块预填充配置的不同，在 10-25 ms 之间波动。

这些是 NVIDIA 官方公布的参考基准。它们会随模型规模（70B 模型会高出 3-5 倍）、硬件（H100 与 B200 相差约 3 倍）以及负载而变化。

### 测量陷阱

2026 年最常用的两款基准测试工具在同次运行中对 TPOT 的结果不一致：

- **NVIDIA GenAI-Perf**：在计算 ITL 时排除了 TTFT。ITL 从第 2 个 token 开始计算。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始计算。

对于一个 TTFT 为 500 ms、总解码时间为 700 ms 且输出 100 个 token 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，而 LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具的选择直接改变了数值。

务必注明所使用的工具。务必公开具体的定义。

### 构建 SLO

2026 年面向消费者的 70B 聊天模型合理的 SLO 如下：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- 输出长度 <300 个 token 时，E2E P99 <= 3 s。
- Goodput 目标值 >= 99%。

企业级 SLO 通常会收紧 TTFT（200-400 ms）并放宽 E2E。关键在于将其书面化，测量全部三项指标，并将 goodput 作为单一综合指标进行追踪。

### 如何测量

- 运行真实流量或逼真的合成流量（使用带 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150` 的 LLMPerf）。
- 基准测试的运行并发数应达到峰值并发数的 2 倍。
- 运行 30-50 次迭代，合并样本后计算百分位数。
- 发布时需注明工具名称、工具版本、模型、硬件、并发数及提示词分布。

## 动手实践

`code/main.py` 是一个简易的 goodput 计算器。它会生成合成的延迟分布，应用 SLO 约束并计算 goodput。同时还会展示在同一条追踪记录下 GenAI-Perf 与 LLMPerf 的 TPOT 差异。

## 交付成果

本课程将产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它将生成一份可直接用于 CI/CD 的基准测试配方，确保部署决策基于 goodput 而非吞吐量。

## 练习

1. 运行 `code/main.py`。生成带有 1% 尾部尖峰的延迟分布。当将 P99 TPOT 的阈值从 30 ms 收紧至 15 ms 时，goodput 会发生怎样的变化？
2. 某厂商报价称“在 H100 上运行 Llama 3.3 70B 可达 15,000 tok/s”。在采信该数据前，你会提出哪三个问题？
3. 为什么分块预填充（chunked prefill）能保护 P99 TPOT，却无法改善平均 TPOT？
4. 为语音助手构建一项消费者级 SLO（首个 token 是被听到而非读到）。哪项指标对用户感知最明显？
5. 阅读 LLMPerf 的 README 和 GenAI-Perf 的文档。找出另外三项工具存在分歧的指标。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| TTFT | “首字生成时间” | 排队 + 网络 + 预填充；长提示词下由预填充主导 |
| TPOT | “每个输出 token 的时间” | 首个 token 之后的显存受限解码成本 |
| ITL | “词元间延迟” | 大多数工具中与 TPOT 相同（并非全部——参见 GenAI-Perf） |
| E2E | “端到端” | TTFT + TPOT × 输出长度；额外叠加响应侧网络延迟 |
| Throughput | “每秒 token 数” | 集群效率；缺乏延迟百分位数时无意义 |
| Goodput | “SLO 达标率” | 同时满足所有 SLO 约束的请求比例 |
| P99 | “尾部延迟” | 1/100 的最差情况延迟；用户体验核心指标 |
| SLO multi-constraint | “联合约束” | 三项延迟上限的 AND 逻辑；任一违反即判定失败 |
| GenAI-Perf vs LLMPerf | “工具陷阱” | 工具对 ITL 是否包含 TTFT 存在分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) —— TTFT、ITL、TPOT 的标准定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) —— 替代性定义与测量方法。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) —— 真实部署中的实际应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) —— 基于 Ray 的开源基准测试。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) —— NVIDIA 官方基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) —— 业界公认的基于 goodput 的基准测试。
