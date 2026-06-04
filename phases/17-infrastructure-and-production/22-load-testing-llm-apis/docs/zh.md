# 对 LLM API 进行负载测试——为什么 k6 和 Locust 会“说谎”

> 传统的负载测试工具并非为流式响应、可变输出长度、Token 级指标或 GPU 饱和度而设计。大多数团队都会踩中两个陷阱。GIL 陷阱：Locust 的 Token 级测量在 Python GIL 下运行分词操作，在高并发时会与请求生成争抢资源；分词积压会导致报告的 Token 间延迟虚高——瓶颈在你的客户端，而非服务端。提示词均匀性陷阱：循环中使用完全相同的提示词只能测试 Token 分布中的一个点；真实流量具有多变的长度和丰富的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年工具选型映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于实现 Token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式感知，基于 Kubernetes 原生通过 TestRun/PrivateLoadZone CRD 进行分布式部署，最适合 CI/CD 门禁；Vegeta 用于 Go 语言的恒定速率饱和测试；Locust 2.43.3 仅在使用 LLM-Locust 扩展支持流式响应时使用。负载模式：稳态（steady-state）、爬坡（ramp）、尖峰（spike，用于自动扩缩容测试）、浸泡（soak，用于内存泄漏检测）。

**类型：** 构建实践
**语言：** Python（标准库，含简易的真实提示词生成器 + 延迟收集器）
**前置知识：** 第 17 阶段 · 08（推理指标），第 17 阶段 · 03（GPU 自动扩缩容）
**预计耗时：** 约 75 分钟

## 学习目标

- 解释导致通用负载测试工具在测试 LLM API 时产生误导的两个反模式（GIL 陷阱、提示词均匀性陷阱）。
- 根据特定用途选择合适的工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成测试）、GenAI-Perf（NVIDIA 参考工具）。
- 设计四种负载模式（稳态、爬坡、尖峰、浸泡），并指出每种模式能捕获的故障模式。
- 使用输入 Token 的平均值 + 标准差来构建真实的提示词分布，而非使用固定长度。

## 问题所在

你用 k6 在 500 个并发用户下测试了 LLM 端点。它扛住了。你发布了上线。但在生产环境中，实际只有 200 个用户时服务就崩溃了——P99 TTFT（首 Token 延迟）飙升，GPU 被完全占满。

发生了两件事。首先，k6 发送了 500 个完全相同的提示词——你的请求合并和前缀缓存机制让它看起来像是在处理 500 个并发解码，但实际上只处理了一个。其次，k6 无法以人眼体验的方式追踪流式响应中的 Token 间延迟；它看到的只是一个 HTTP 连接，而不是 500 个以不同间隔到达的 Token。

LLM 的负载测试是一门独立的学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 基于 Python，并在客户端 GIL 下运行分词操作。在高并发时，分词器会在请求生成之后排队。报告的 Token 间延迟包含了客户端的分词积压。你以为服务端很慢，其实是测试框架拖了后腿。

解决方案：使用 LLM-Locust 扩展将分词移至独立进程，或使用编译型语言的测试框架（如 k6、基于 tokenizers.rs 的 LLMPerf）。

### 提示词均匀性陷阱

所有已知的负载测试工具都允许你配置单个提示词。在 10,000 次迭代的循环测试中，每次发送的都是完全相同的提示词。服务端每次都看到相同的前缀——前缀缓存命中率接近 100%，吞吐量看起来非常优秀。

解决方案：从提示词分布中进行采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——长度多样，内容各异。

### 四种负载模式

1. **稳态（Steady-state）**——持续 30-60 分钟的恒定 RPS（每秒请求数）。捕获：基线性能回退。
2. **爬坡（Ramp）**——在 15 分钟内将 RPS 从 0 线性增加至目标值。捕获：容量临界点、预热异常。
3. **尖峰（Spike）**——突然增至 3-10 倍 RPS 持续 2 分钟，然后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡（Soak）**——持续 4-8 小时的稳态负载。捕获：内存泄漏、连接池漂移、可观测性数据溢出。

### 2026 年工具选型映射

**LLMPerf**（Anyscale）——基于 Python 但由 Rust 驱动分词。支持均值/标准差提示词。流式感知。性能测试的首选默认工具。

**NVIDIA GenAI-Perf**——NVIDIA 官方参考工具。使用 Triton 客户端；指标覆盖全面。注意其 ITL（Token 间延迟）不包含 TTFT；而 LLMPerf 包含。同一台服务器用这两个工具测出的 TPOT（每 Token 时间）会不同。

**LLM-Locust**（TrueFoundry）——修复 GIL 陷阱的 Locust 扩展。保留熟悉的 Locust DSL 并增加流式指标。

**guidellm**——大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go 语言编写，编译型，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生的分布式测试。
- 最适合用于 CI/CD 门禁和 SLA 测试。

**Vegeta**——Go 语言编写，比 k6 更简单。恒定速率 HTTP 饱和测试。不具备 LLM 感知能力，但非常适合网关/限流测试。

**原生 Locust 2.43.3**——存在针对 LLM 的 GIL 陷阱。必须搭配 LLM-Locust 扩展使用。

### CI 中的 SLA 门禁

在 PR（Pull Request）上运行 k6，配置如下：

- 每个基准 RPS 下进行 30-50 次迭代。
- 门禁条件：P50/P95 TTFT 达标，5xx 错误率 < 5%，TPOT 低于阈值。
- 一旦突破阈值则中断构建。

### 真实的提示词分布

基于真实流量样本（如果有）或已发布的公开分布（例如聊天用的 ShareGPT 提示词、代码用的 HumanEval）构建。将平均值 + 标准差输入 LLMPerf。无论如何都要避免“单提示词循环”的做法。

### 需要记住的关键数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：支持流式感知指标。
- 典型 LLMPerf 运行：并发度 X 下发起 100-1000 个请求。
- 典型 CI 门禁：每个 PR 执行 30-50 次迭代。
- 四种模式：稳态、爬坡、尖峰、浸泡。

## 动手实践

`code/main.py` 模拟了一次带有真实提示词分布的负载测试，测量有效 TPOT，并演示了均匀提示词陷阱。

## 交付成果

本课程将产出 `outputs/skill-load-test-plan.md`。根据工作负载和 SLA 要求，选择合适工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。对比均匀分布与真实分布的差异——差距出现在哪里？
2. 编写用于 CI 门禁的 k6 脚本：100 并发下 TTFT P95 < 800 ms，运行时长 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。列出三种可能原因，并说明如何选用监控手段来区分它们。
4. 尖峰测试从 10 RPS 升至 100 RPS。如果已部署 Karpenter + vLLM 生产环境栈（第 17 阶段 · 03 + 18），预期的恢复时间是多少？
5. 在同一台服务器上，GenAI-Perf 报告 TPOT=6ms；LLMPerf 报告 TPOT=11ms。请解释原因。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| LLMPerf | “LLM 测试框架” | Anyscale 基准测试工具，支持流式感知 |
| GenAI-Perf | “NVIDIA 工具” | NVIDIA 官方参考测试框架 |
| LLM-Locust | “面向 LLM 的 Locust” | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | “合成基准测试” | 大规模合成测试工具 |
| k6 Operator | “K8s 版 k6” | 基于 CRD 的分布式 k6 调度器 |
| GIL trap | “Python 客户端开销” | 分词积压导致报告的延迟虚高 |
| Prompt-uniformity trap | “单提示词谎言” | 循环使用相同提示词命中缓存，虚增吞吐量 |
| Steady-state | “恒定负载” | 持续 N 分钟的平稳 RPS |
| Ramp | “线性爬坡” | 在指定时间内从 0 升至目标值 |
| Spike | “突发测试” | 瞬间倍增后恢复 |
| Soak | “长时测试” | 持续数小时以检测内存泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
