# 综合项目 14 —— 投机解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在实际流量下实现了 2.5-3 倍的吞吐量提升。P-EAGLE（AWS 2026）将并行投机推向了新高度。SGLang 的 SpecForge 大规模训练了草稿头（draft heads）。Red Hat 的 Speculators 中心为常见开源模型发布了校准的草稿。TensorRT-LLM 在 NVIDIA 平台上将投机解码提升为一等公民。2026 年的生产级推理栈采用 vLLM 或 SGLang，搭配 EAGLE 系列草稿、FP8 或 INT4 量化，并基于队列等待时间配置 HPA。本综合项目的目标是：以超过基线 2.5 倍的吞吐量服务两个开源模型，并提供完整的尾延迟报告。

**类型：** 综合项目
**语言：** Python（推理服务）、C++ / CUDA（内核检查）、YAML（配置文件）
**前置要求：** 阶段 3（深度学习）、阶段 7（Transformer）、阶段 10（从零构建大语言模型）、阶段 17（基础设施）
**涉及阶段：** P3 · P7 · P10 · P17
**耗时：** 30 小时

## 问题描述

投机解码在 2026 年已成为标配。EAGLE-3 草稿头在目标模型的隐藏状态上进行训练，可提前预测 N 个 token；目标模型只需单次前向传播即可完成验证。60-80% 的接受率可转化为 2-3 倍的端到端吞吐量。vLLM 0.7 已原生集成该功能。SGLang + SpecForge 提供了完整的训练流水线。Red Hat 的 Speculators 为 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 发布了校准的草稿模型。

核心难点在于推理运维，而非模型本身。接受率会随流量分布的变化而发生漂移（例如 ShareGPT 对话数据与代码数据或垂直领域数据的差异）。拒绝采样时的尾延迟比不使用投机时更差——你必须报告多个 batch size 下的 p99 延迟，而不仅仅是稳态下的 tokens/sec。与 Anthropic / OpenAI API 相比，每百万 token 的成本是衡量可信度的关键指标。

## 核心概念

投机解码包含两层架构。**草稿（draft）**模型（如 EAGLE-3 head、ngram 或较小的目标对齐模型）在每个步骤中提出 k 个候选 token。**目标（target）**模型在一次前向传播中验证所有 k 个 token；任何被接受的前缀都会替换原有的贪婪解码路径。接受率取决于草稿模型与目标模型的校准程度以及输入数据的分布。

在大多数流量场景下，EAGLE-3 的表现优于 ngram 草稿。P-EAGLE 通过并行投机支持更深的草稿树。其权衡在于：由于验证步长变大，拒绝情况下的 P99 延迟会更高。因此，推理配置必须按 batch size 分桶报告延迟，以暴露这一现象。

部署环境为 Kubernetes。vLLM 0.7 每个 GPU 或张量并行分片运行一个副本。HPA 基于队列等待时间（queue-wait）而非 CPU 使用率进行自动扩缩容。FP8（Marlin）和 INT4（AWQ）量化可将显存占用控制在 H100 / H200 的容量范围内。端到端报告需包含吞吐量、接受率、batch 1/8/32 下的 p50/p99 延迟，以及每百万 token 的成本（$/1M tokens）。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 推理服务：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 草稿头、P-EAGLE 并行投机、ngram 降级方案
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；基于 queue-wait 指标的 HPA
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval（用于跨领域接受率测量）
- 参考实现：TensorRT-LLM 投机解码（厂商基线对比）

## 实施步骤

1. **准备目标模型。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在单卡 H100（或双卡张量并行）上通过 vLLM 0.7 部署。

2. **获取草稿模型。** 从 Red Hat Speculators 拉取校准的 EAGLE-3 草稿头（或通过 SpecForge 自行训练）。将其加载到 vLLM 的投机解码配置中。

3. **记录基线数据。** 启用投机前：记录 batch 1/8/32 下的 tokens/s、p50/p99 延迟及 GPU 利用率。并发布结果。

4. **启用 EAGLE-3。** 切换配置；重新运行相同的基准测试。报告加速比、接受率及 p99 尾延迟变化量。

5. **P-EAGLE。** 启用并行投机；测量更深草稿树与串行 EAGLE-3 的性能差异。报告 P-EAGLE 性能由正转负的拐点。

6. **领域流量测试。** 将 ShareGPT、HumanEval 及特定领域流量送入同一服务器。按不同分布测量接受率。识别草稿模型发生漂移的场景。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行相同流水线。草稿生成更具挑战性（存在 MoE 路由噪声）。提交报告。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。演示负载增加三倍时的自动扩容效果。

9. **成本对比。** 在同一评估集上计算每百万 token 的成本，并与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 进行对比。发布结果。

## 使用方式

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付要求

`outputs/skill-inference-server.md` 描述了交付物要求。包含经过测量的带投机解码的推理栈、完整的基准测试报告以及 K8s 部署配置。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 相对于基线的实测加速比 | 在两个模型上保持质量一致的前提下，吞吐量提升 2.5 倍以上 |
| 20 | 真实流量下的接受率 | 按不同流量分布提交的接受率报告 |
| 20 | P99 尾延迟规范 | 有/无投机情况下，batch 1/8/32 的 p99 延迟数据 |
| 20 | 运维能力 | K8s 部署、基于 queue-wait 的 HPA、平滑发布 |
| 15 | 文档与方法论 | 清晰说明变更内容及原因 |
| **100** | | |

## 练习

1. 测量草稿模型落后于目标模型一个版本时的接受率衰减（例如 Llama 3.3 到 3.4 的版本漂移）。构建相应的监控告警。

2. 实现 ngram 降级机制：若 EAGLE-3 接受率低于阈值，则切换至 ngram 草稿。报告可靠性提升情况。

3. 运行受控的 MoE 实验：对比注入路由噪声与未注入噪声的同一 Qwen3-Coder-30B。测量草稿接受率的敏感度。

4. 扩展至 H200（141 GB）。报告每个副本可用的模型尺寸余量，并评估是否可直接服务未量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上对 TensorRT-LLM 投机解码进行基准测试。报告其相较于 vLLM 的优势场景。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 草稿模型 | “投机器” | 提出 N 个 token 供目标模型验证的小型模型 |
| EAGLE-3 | “2026 草稿架构” | 基于目标隐藏状态训练的草稿头；接受率约 75% |
| P-EAGLE | “并行投机” | 在一次目标模型前向传播中验证的草稿分支树 |
| 接受率 | “命中率” | 无需重采样即被接受的草稿 token 比例 |
| 量化 | “FP8 / INT4” | 降低精度以在 GPU 显存中容纳更大模型的权重压缩技术 |
| 队列等待 | “HPA 指标” | 请求在推理开始前于待处理队列中等待的时间 |
| Speculators 中心 | “校准草稿” | Red Hat Neural Magic 提供的针对常见开源模型的 EAGLE 草稿集合 |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) —— 参考推理栈文档
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) —— 并行投机解码论文与集成指南
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) —— 草稿头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) —— 校准草稿中心
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) —— 厂商替代方案
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) —— 商业参考案例
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) —— 方法论文
- [vLLM repository](https://github.com/vllm-project/vllm) —— 源码与基准测试
