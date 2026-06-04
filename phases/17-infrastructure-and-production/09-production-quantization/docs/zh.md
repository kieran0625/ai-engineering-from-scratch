# 生产环境量化 —— AWQ、GPTQ、GGUF K-quant、FP8、MXFP4/NVFP4

> 量化格式并非万能选择——它取决于硬件、推理引擎和工作负载。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘端，通过 llama.cpp 和 Ollama 交付。在 vLLM 中，当你需要在同一基础模型上部署多个 LoRA 时，GPTQ 是首选。配合 Marlin-AWQ 内核的 AWQ 在 7B 级别模型上可实现约 741 tok/s 的吞吐量，并在 INT4 格式中获得最佳的 Pass@1 表现——这是 2026 年数据中心生产环境的默认选择。FP8 在 Hopper、Ada 和 Blackwell 架构上占据中间地带——近乎无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）更为激进，需要逐块验证。团队常踩的两个坑：校准数据集必须与部署领域匹配；KV cache 独立于权重量化之外——AWQ 的教训“我的模型现在只有 4 GB”往往忽略了在生产批次大小下高达 10-30 GB 的 KV cache。

**类型：** 学习
**语言：** Python（标准库，用于跨格式的简易内存与吞吐量对比）
**前置知识：** 阶段 10 · 13（量化基础），阶段 17 · 04（vLLM 推理内部机制）
**预计时间：** 约 75 分钟

## 学习目标

- 说出六种生产环境量化格式及其在 2026 年的适用场景。
- 根据硬件（CPU 与 GPU、Hopper 与 Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规对话、推理、多 LoRA）选择合适的格式。
- 计算选定格式所节省的权重内存以及未受影响的 KV cache 大小。
- 指出会导致量化模型在领域流量上性能下降的校准数据集陷阱。

## 问题背景

量化能减少内存占用和 HBM 带宽需求，这正是解码（decode）阶段所需要的。一个 FP16 精度的 70B 模型权重为 140 GB。将权重量化为 INT4（AWQ 或 GPTQ）后，模型大小降至 35 GB——可放入单张 H100 显卡，并为 KV cache 留出空间。这很重要，因为在 128 个并发序列、2k 上下文的场景下，仅 KV cache 就需要 20-30 GB。

但量化并非没有代价。激进的量化会降低模型质量，尤其是在重度依赖推理的任务上。不同格式适配不同的引擎，不同硬件原生支持的精度也不同。2026 年的格式生态确实繁杂，你不能直接照搬别人的选择——必须根据你的技术栈来决策。

## 核心概念

### 六种格式

| 格式 | 位数 | 适用场景 | 推理引擎 |
|------|------|----------|----------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA 部署 | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产环境 | vLLM (Marlin-AWQ)、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户场景 | TRT-LLM |

### GGUF —— CPU/边缘端默认选择

GGUF 是一种文件格式，而非纯粹的量化方案——它将多种 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产环境默认选项——在 4-5 位精度下提供接近 BF16 的质量。由于 llama.cpp 是目前最快的 CPU 推理引擎，因此它是 CPU 或边缘端部署的最佳选择。

在 vLLM 中的吞吐量损耗：7B 模型仅约 93 tok/s——该格式并未针对 GPU 内核进行优化。仅在部署目标为 CPU/边缘端时使用 GGUF，否则不建议。

### GPTQ —— vLLM 中的多 LoRA 部署

GPTQ 是一种带有校准步骤的训练后量化算法。Marlin 内核使其在 GPU 上运行极快（相比非 Marlin 版本的 GPTQ 提速 2.6 倍）。7B 模型下约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA 适配器。如果你需要部署一个基础模型加上 10-50 个微调变体（每个均以 LoRA 形式存在），GPTQ 是你的正确路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ —— 数据中心 GPU 默认选择

激活感知权重量化（Activation-aware Weight Quantization）。在量化过程中保护约 1% 的最显著权重。Marlin-AWQ 内核：相比朴素实现提速 10.9 倍。7B 模型下约 741 tok/s，在 INT4 格式中拥有最佳的 Pass@1 表现。

除非你需要多 LoRA 部署（选 GPTQ）或激进的 Blackwell FP4（选 NVFP4），否则新建 GPU 服务时应优先选择 AWQ。

### FP8 —— 可靠的中间方案

8 位浮点数。近乎无损。广泛支持。Hopper Tensor Core 原生加速 FP8，Blackwell 架构继承此特性。当质量要求不可妥协时（如推理、医疗、代码生成），FP8 是 2026 年安全的首选默认值。其内存节省幅度约为 INT4 的一半，但质量风险要低得多。

### MXFP4 / NVFP4 —— Blackwell 激进方案

微缩放 FP4（Microscaling FP4）。每组权重拥有独立的缩放因子。方案较为激进，但在 Blackwell Tensor Core 上享有硬件加速。相较于 FP8，每 token 字节数减半——这是阶段 17 · 07 中提到的经济效益所在。

注意事项：
- 目前尚不支持 LoRA（截至 2026 年初）。
- 在重度推理工作负载上会出现可见的质量下降。
- 需针对每个模型在你的评估集上进行验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常为 C4 或 WikiText。对于领域模型（代码、医疗、法律），若使用通用网页文本进行校准，算法可能会错误判断哪些权重需要保护。这可能导致 HumanEval 上的 Pass@1 指标下降数个点位。

解决方案：使用领域内数据进行校准。通常数百条领域样本即可。上线前务必在评估集上进行测试。

### KV cache 陷阱

AWQ 将权重量化至 4 位。KV cache 是独立存储的，仍保持 FP16/FP8 精度。以配备 AWQ 的 70B 模型为例：

- 权重：约 35 GB（从 140 GB 压缩至 INT4）。
- KV cache（128 并发 × 2k 上下文）：约 20 GB。
- 激活值：约 5 GB。
- 总计：约 60 GB——可容纳于单张 80GB H100。

盲目地认为“我把模型量化到了 4 GB”会忽略掉另外 30-50 GB 的开销。必须整体规划 HBM 预算。

此外，KV cache 量化（FP8 KV 或 INT8 KV）是一个独立的决策，有其自身的权衡——它会直接影响注意力计算的精度，并非免费午餐。

### AWQ INT4 对推理任务存在风险

思维链（Chain-of-thought）、数学计算、长上下文代码生成——这些任务会因激进的量化而明显受损。AWQ INT4 在 MATH 基准测试上会损失约 3-5 个百分点。对于重度推理工作负载，请交付 FP8 或 BF16 版本；接受相应的内存成本。

### 2026 选型指南

- CPU/边缘端部署：GGUF Q4_K_M。搞定。
- GPU 部署、常规对话、无需 LoRA：AWQ。
- GPU 部署、多 LoRA：带 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心、已验证质量：NVFP4 + FP8 KV。
- 不确定时：对每种候选格式运行 1,000 条样本的评估。

## 实践应用

`code/main.py` 计算了不同模型规模下六种格式的内存占用（权重 + KV + 激活值）及相对吞吐量。展示了 KV cache 何时占主导地位、权重压缩何时带来收益，以及何时 FP8 是安全的选择。

## 交付成果

本课程将产出 `outputs/skill-quantization-picker.md`。根据硬件配置、模型规模、工作负载类型和质量容忍度，选择一种格式并制定校准/验证计划。

## 练习

1. 运行 `code/main.py`。针对 128 并发、2k 上下文的 70B 模型，计算每种格式的总 HBM 占用。哪种格式能让你将其塞进单张 80GB H100？
2. 你有一个 7B 的代码模型。选择一种格式并说明理由。如果你对质量容忍度的判断有误，回退路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么数据越多并不总是越好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为何 AWQ 在 7B 模型上能达到 741 tok/s，而原始 GPTQ 仅为 ~712。
5. 在什么情况下，将 AWQ 权重与 FP8 KV cache 结合比将 KV 保持为 BF16 更合理？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GGUF | “llama.cpp 格式” | 打包 K-quant 变体的文件格式；CPU/边缘端默认格式 |
| Q4_K_M | “Q4 K M” | 4 位 K-quant 中等精度；生产环境 GGUF 默认选项 |
| GPTQ | “gee pee tee q” | 带校准的训练后 INT4 量化；在 vLLM 中支持 LoRA |
| AWQ | “a w q” | 激活感知 INT4 量化；Marlin 内核；INT4 中 Pass@1 最佳 |
| Marlin kernels | “fast INT4 kernels” | 专为 Hopper 架构设计的 INT4 自定义 CUDA 内核；提速 10 倍 |
| FP8 | “eight-bit float” | Hopper/Ada/Blackwell 上的安全精度默认值 |
| MXFP4 / NVFP4 | “microscaling four” | Blackwell 4 位浮点，带逐块缩放因子 |
| Calibration dataset | “cal data” | 用于确定量化参数的输入文本；必须与领域匹配 |
| KV cache quantization | “KV INT8” | 独立于权重的选择；直接影响注意力精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) —— 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) —— 各格式的吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) —— 逐格式选型指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) —— 支持的格式与命令行参数。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) —— AWQ 原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) —— GPTQ 原始论文。
