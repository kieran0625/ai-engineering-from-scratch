# vLLM Serving 内部机制：PagedAttention、Continuous Batching、Chunked Prefill

> 2026 年 vLLM 的统治地位并非依赖单一技巧，而是三项默认配置的叠加效应。PagedAttention 始终开启。Continuous Batching 会在每次 decode 迭代之间将新请求注入活跃批次。Chunked Prefill 将长 prompt 切片，确保 decode token 永不饥饿。同时开启这三项后，单张 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下可推至 2,200-2,400 tok/s —— 比 vLLM 自身默认值高出约 25%，是朴素 PyTorch 循环的 3-4 倍。本课程将以可绘制架构图的粒度解析调度器与注意力内核，并以 `code/main.py` 中的简易 Continuous Batcher 收尾，其调度 prefill 和 decode 的方式与 vLLM 一致。

**类型：** Learn
**语言：** Python（stdlib、简易 Continuous Batching 调度器）
**前置知识：** Phase 17 · 01 (Model Serving)、Phase 11 (LLM Engineering)
**耗时：** ~75 分钟

## 学习目标

- 将 PagedAttention 解释为 KV cache 分配器：说明 block、block table 的概念，以及为何在生产负载下碎片率保持在 4% 以下。
- 从迭代层面图解 Continuous Batching：说明已完成序列如何离开批次、新序列如何加入而无需清空批次。
- 用一句话描述 Chunked Prefill，并指出它保护的是哪项延迟指标（提示：是 TTFT 尾部延迟，而非平均吞吐量）。
- 指出 2026 版 vLLM v0.18.0 中一个容易让团队一次性开启所有优化时踩坑的陷阱。

## 问题所在

朴素的 PyTorch serve 循环一次只运行一个请求：tokenize、prefill、decode 直到 EOS，然后返回。面对单个用户时这没问题，但面对一百个用户时，就成了排队的耐心人群。显而易见的修复方案——static batching——会将窗口内的每个请求都 pad 到最长 prompt 的长度，将每次 decode 都 pad 到最长预期输出长度，并让整个批次卡在进度最慢的序列上。你为从未使用的 padding 买单，快请求必须等待慢请求。

vLLM 同时解决了三个问题。PagedAttention 阻止了 KV cache 碎片像经典连续分配那样吞噬 60-80% 的 GPU 显存。Continuous Batching 允许请求在每次 decode 迭代之间加入或离开批次，使批次始终充满真实计算任务。Chunked Prefill 将 32k token 的 prompt 拆分为约 512 token 的切片并与 decode 交错执行，从而避免长 prompt 冻结 GPU 上的所有 decode token。

2026 年的生产环境默认配置是全部开启这三项。你需要理解每项的作用，因为故障模式全出在调度器上，而非模型本身。

## 核心概念

### PagedAttention 作为虚拟内存系统

每个序列的 KV cache 占用 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 8192 token 的 Llama 3.3 70B，BF16 精度下每个序列约占 1.25 GB。如果你为每个请求预分配 8192 个 slot，但平均请求仅使用 1500 token，你将浪费约 82% 已预留的 HBM。Classic batching 就要承担这种浪费。

PagedAttention 借鉴了操作系统虚拟内存的思想。KV cache 在序列内并非连续存储。它以固定大小的 block 进行分配（默认 16 token）。每个序列拥有一个 block table，将其逻辑 token 位置映射到物理 block ID。当序列增长超出已分配的 block 时，会追加一个新的 block。当序列完成时，其占用的 block 会归还至内存池。

碎片率从 60-80%（经典方法）降至 4% 以下（PagedAttention）。你无需通过 flag 启用 PagedAttention——它是 vLLM 唯一提供的分配器。调节旋钮是 `--gpu-memory-utilization`（默认 0.9），用于告知 vLLM 在加载权重和激活值后，应为 KV block 预留多少 HBM。

### 迭代层面的 Continuous Batching

旧的“dynamic batching”会等待一个时间窗口（例如 10 ms）来填满批次，然后执行 prefill + decode + decode + decode，直到所有序列结束。快序列会提前退出并闲置，等待 GPU 完成慢序列的计算。

Continuous Batching 在每次 decode 步骤之间运行。将正在运行的序列集合称为 `RUNNING` 列表。在每次迭代中：

1. 任何刚达到 EOS 或 max_tokens 的 `RUNNING` 序列都会被移除。
2. 调度器查看等待队列。如果有空闲的 KV block，它会接纳新序列（prefill 或恢复执行）。
3. forward pass 在当前 `RUNNING` 中的所有序列上运行，为每个序列生成一个新 token。

批次大小永远不会 pad 到固定数值。处于输出不同位置的序列共享同一个 fused forward。在 2026 版 vLLM 中，这被称为 `V1 scheduler`。关键不变量：调度器按每次 decode 迭代运行一次，而非按每个请求运行一次。

### Chunked Prefill 保护 TTFT 尾部

Prefill 是计算密集型任务。在单张 H100 上，Llama 3.3 70B 处理 32k token 的 prompt 需要约 800 ms 的纯 prefill 时间。在 prefill 运行时，批次中其他所有序列的 decode token 都在等待。在 serve 循环中，一个长 prompt 的首 token 延迟（TTFT）会变成数十个其他用户的 inter-token 延迟（ITL）尖峰。

Chunked Prefill 将 prefill 拆分为固定大小的 chunk（默认 512 token），并将每个 chunk 作为独立单元进行调度。在 chunk 之间，调度器可以让 decode 序列推进一个 token。你以微小的绝对 prefill 延迟损失（每个 chunk 几毫秒）换取大幅降低的 decode 时间抖动。公开基准测试显示，混合负载下的 P99 ITL 从约 50 ms 降至约 15 ms。

### 三项默认配置的交互作用

这三项功能互为前提。PagedAttention 为调度器提供了细粒度的 KV 资源以供权衡。Continuous Batching 需要这种细粒度资源，以便接纳新序列时无需触发全局重排。Chunked Prefill 是调度器在同一 `RUNNING` 列表上做出的决策——它是另一项调度策略，而非独立系统。

你不需要记住每一个 flag。你需要知道调度器优化的是什么：在 KV block 预算约束下，结合 Chunked Prefill 切片策略的 goodput。

### 2026 v0.18.0 的陷阱

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与 draft model speculative decoding（`--speculative-model`）组合使用。文档记录的例外情况是 V1 scheduler 中的 N-gram GPU speculative decoding。未阅读发布说明就盲目开启所有 flag 的团队会在启动时遇到运行时错误，而非软性退化。如果你的 speculative 收益值得为此启用 Chunked Prefill，请重新评估该选择——2026 年的正确解法通常是关闭 Chunked Prefill 的 EAGLE-3，而非无法编译的 draft model 加 Chunked Prefill。

### 需要记住的关键数据

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三项全开：2,200-2,400 tok/s。
- 同模型，vLLM 默认配置（无 Chunked Prefill）：~1,800 tok/s。
- 同模型，朴素 PyTorch forward 循环：~600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片浪费：<4%。
- 混合负载下的 P99 ITL：开启 Chunked Prefill 约 15 ms，未开启约 50 ms。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是使用 stdlib Python 实现的上述循环，包含伪造的 token 计数与伪造的 forward 延迟。运行它可以直观展示 Chunked Prefill 如何在长 prefill 期间保持 decode 序列持续运行。

## 实践应用

`code/main.py` 模拟了一个支持开关特性的 vLLM 风格调度器。运行它以观察：

- `NAIVE` 模式：逐请求处理，无 batching。
- `STATIC` 模式：pad 并等待，classic batching。
- `CONTINUOUS` 模式：迭代级别的接纳与释放。
- `CONTINUOUS + CHUNKED` 模式：prefill 切片与 decode 交错执行。

输出结果包含总吞吐量（虚拟秒/token）、TTFT 均值与 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 行应表现最优。

## 部署交付

本课程将产出 `outputs/skill-vllm-scheduler-reader.md`。给定 serve 配置（batch size、KV 内存利用率、Chunked Prefill 大小、speculative 配置），它将生成一份调度器诊断报告，明确指出三项默认配置中哪一项是瓶颈，并提供调优建议。

## 练习

1. 运行 `code/main.py`。在包含短请求与长请求混合的工作负载下，对比 `STATIC` 与 `CONTINUOUS`。吞吐量差距源于何处——prefill 效率、decode 效率还是尾部延迟？
2. 修改简易调度器以添加 `--max-num-batched-tokens`。在运行 Llama 3.3 70B FP8 的 H100 上，该值的合理设定是多少？（提示：它是 KV block 大小与空闲 block 数量的函数，而非原始 HBM。）
3. 重读 vLLM v0.18.0 发布说明。哪些 flag 组合是互斥的？请列出它们。
4. 针对 1,000 个请求的追踪记录（平均输出 1,500 token，标准差 600 token），计算以下两种情况下的 KV cache 碎片浪费：(a) 最大 8192 的单请求连续分配；(b) 使用 16-token block 的 PagedAttention。
5. 用一段话解释为何 Chunked Prefill 能改善 P99 ITL 却孤立来看无法提升吞吐量。实际应用中吞吐量的提升究竟来自哪里？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|------------------------|
| PagedAttention | “KV 技巧” | 用于 KV cache 的固定大小 block 分配器；碎片率 <4% |
| Block table | “页表” | 按序列划分，将逻辑 token 位置映射到物理 KV block 的表格 |
| Continuous Batching | “正确的 dynamic batching” | 在每次 decode 迭代时做出接纳/释放决策 |
| Chunked Prefill | “prefill 拆分” | 将长 prefill 拆分为 512-token 切片并与 decode 交错执行 |
| TTFT | “首 token 时间” | Prefill + 排队 + 网络；长 prompt 下由 prefill 主导 |
| ITL | “词元间延迟” | 连续 decode token 之间的时间；受 batch size 主导 |
| Goodput | “符合 SLO 的吞吐量” | 满足所有请求 TTFT 与 ITL 目标的 token/sec |
| V1 Scheduler | “新调度器” | vLLM 2026 版调度器；N-gram spec decode 是与 Chunked Prefill 兼容的路径 |
| `--gpu-memory-utilization` | “内存旋钮” | 加载权重与激活值后，为 KV block 预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) —— 关于 Chunked Prefill 与 speculative decoding 兼容性的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) —— 2026 版本发布节奏与特定版本行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) —— 定义该分配器思考方式的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) —— 碎片分析与调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) —— 附带 flame graphs 的 V1 Scheduler 详细拆解。
