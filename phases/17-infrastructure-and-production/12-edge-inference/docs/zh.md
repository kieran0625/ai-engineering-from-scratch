# 边缘推理 — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> 边缘计算的核心瓶颈是内存带宽，而非算力。移动端 DRAM 带宽约为 50-90 GB/s；数据中心 HBM3 可达 2-3 TB/s —— 差距达 30-50 倍。解码过程受内存带宽限制，因此这一差距具有决定性影响。到 2026 年，技术格局将分为四大阵营。Apple M4/A18 Neural Engine 峰值算力达 38 TOPS，采用统一内存架构（无需 CPU↔NPU 间的数据拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 的 Hexagon 处理器可达 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B (Q4) 约 41 tok/s（约为原生性能的 70-80%）；GitHub Star 数达 1.76 万，提供 OpenAI 兼容 API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super (8GB) 可运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 约 40 tok/s；Jetson T4000 (JetPack 7.1) 性能为 AGX Orin 的两倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4 和分块预填充（chunked prefill）—— 博世、中科创达、联发科已在 CES 2026 上展示。

**类型：** 学习
**语言：** Python（标准库、简易带宽受限解码模拟器）
**前置知识：** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**预计时间：** 约 60 分钟

## 学习目标

- 解释为何移动端 LLM 推理受内存带宽限制，而算力处于次要地位。
- 列举四大边缘目标平台（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并说明各自适用的使用场景。
- 指出 2026 年 WebGPU 的覆盖缺口（Firefox Android 正在追赶）及 Safari iOS 26 的发布情况。
- 为各目标平台选择合适的量化格式（ANE 使用 Core ML INT4 + FP16，Hexagon 使用 QNN INT8/INT4，浏览器使用 WebGPU Q4，Jetson Thor 使用 NVFP4）。

## 问题背景

某客户希望开发一款端侧聊天机器人：语音优先、默认隐私保护、支持离线运行。在搭载 M3 Max 的 MacBook Pro 上，Llama 3.1 8B Q4 运行速度约为 55 tok/s —— 表现良好。在 iPhone 16 Pro 上，同一模型仅能跑 3 tok/s —— 难以接受。在搭载 Snapdragon 8 Gen 3 的中端 Android 设备上，约为 7 tok/s。在 Chrome Android v121+ 浏览器中通过 WebGPU 运行，根据设备不同约为 4-8 tok/s。

吞吐量的差异并非移植问题所致，而是由带宽差距、量化格式以及用户态是否可访问 NPU 共同决定的。2026 年的边缘推理实际上是四个不同的问题，对应四种不同的解决方案。

## 核心概念

### 带宽才是真正的性能天花板

解码阶段每次生成 token 都需要读取完整的权重集。一个 Q4 格式的 7B 模型大小约为 3.5 GB。以 50 GB/s 的带宽读取 3.5 GB 需要 70 ms —— 理论上限约为 14 tok/s。若达到 90 GB/s（高端移动 DRAM），上限则提升至约 25 tok/s。低于此数值时，增加算力毫无帮助。

数据中心 HBM3 带宽达 3 TB/s，读取同样的 3.5 GB 仅需 1.2 ms —— 上限高达 830 tok/s。相同的模型，相同的权重，但内存子系统截然不同。

### Apple Neural Engine (M4 / A18)

- 峰值算力达 38 TOPS。采用统一内存架构（CPU 与 ANE 共享同一内存池）—— 无数据拷贝开销。
- 可通过 Core ML + `.mlmodel` 编译后的模型进行调用，或通过 PyTorch 借助 Metal Performance Shaders (MPS) 访问。
- Llama.cpp 的 Metal 后端使用的是 MPS 而非直接调用 ANE；原生 ANE 需经过 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：使用 Core ML，配合 INT4 权重 + FP16 激活值。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 峰值算力达 45 TOPS。集成于 SoC 的 CPU 和 GPU 之中，但拥有独立的内存域。
- QNN (Qualcomm Neural Network) SDK 和 AI Hub 提供从 PyTorch/ONNX 的模型转换支持。
- Chat templates、Llama 3.2、Phi-3 均在 AI Hub 上作为一等公民工件提供。

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 算力 40-50 TOPS。软件生态落后于 Apple/Qualcomm；OpenVINO 正在改善但受众较窄。
- 最适合 Windows ARM 平台的 Copilot 类应用；在 AMD/Intel 桌面端原生运行适合本地优先（local-first）场景。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- 在 M3 Max 上运行 Llama 3.1 8B Q4 约 41 tok/s —— 通过相同后端实现，约为原生性能的 70-80%。
- WebLLM 获 1.76 万 GitHub Stars；提供 OpenAI 兼容的 JS API；采用 Apache 2.0 协议。
- 2026 年覆盖情况：Chrome Android v121+、Safari iOS 26 正式版（GA）、Firefox Android 仍在追赶。整体移动端覆盖率约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super (8GB)：可流畅运行 Llama 3.2 3B、Phi-3。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- Thor / T4000 (JetPack 7.1)：性能为 AGX Orin 的两倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM (2026) 支持 EAGLE-3 推测解码、NVFP4 权重、分块预填充 —— 将数据中心的优化技术移植到了边缘端。

### 各目标平台的量化格式选择

| 目标平台 | 格式 | 说明 |
|--------|--------|-------|
| Apple ANE | INT4 权重 + FP16 激活值 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 已编译的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 受内存带宽限制 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘端的长上下文陷阱

Llama 3.1 的 128K 上下文长度是面向数据中心的特性。在 8 GB 内存的手机上，4 GB 模型 + 32K token 的 2 GB KV Cache + 系统开销 = 内存溢出（OOM）。除非接受激进的 KV 量化（Q4 KV），否则边缘部署通常将上下文限制在 4K-8K。

### 语音是杀手级应用

语音代理对延迟敏感（首字延迟 < 500 ms）。本地推理可完全消除网络延迟。结合语音转文本（Whisper Turbo 变体可在边缘端运行），边缘推理即可构成生产级的语音交互闭环。

### 关键数据速记

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 下约 41 tok/s。
- AGX Orin：gpt-oss-20b 通过 vLLM 运行约 40 tok/s。
- 数据中心与边缘端的带宽差距：30-50 倍。
- WebGPU 移动端覆盖率：约 70-75%（Firefox Android 进度滞后）。

## 动手实践

`code/main.py` 基于边缘目标平台的带宽受限数学模型，计算理论解码吞吐量上限。将其与实际基准测试对比，突出显示瓶颈在于带宽而非算力。

## 交付应用

本课将产出 `outputs/skill-edge-target-picker.md`。根据指定平台（iOS/Android/浏览器/Jetson）、模型以及延迟/内存预算，自动选择量化格式与转换流水线。

## 练习

1. 运行 `code/main.py`。针对 Snapdragon 8 Gen 3（带宽约 77 GB/s）上的 Q4 格式 7B 模型，计算其解码上限。与实际观测到的 6-8 tok/s 对比——运行时效率如何？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计降级方案——通过服务端提供相同的 OpenAI 兼容 API。
3. 你的 iOS 应用需要支持 4K 上下文流式输出。在 iPhone 16 上，哪种模型/格式组合能将活跃内存占用控制在 4 GB 以内？
4. Jetson AGX Orin 能以 40 tok/s 运行 gpt-oss-20b，而 Jetson Nano 仅能运行 3B 模型。若产品同时面向两者，你如何统一推理栈？
5. 论证“WebLLM 在 2026 年是否已具备生产环境就绪条件”。请引用覆盖率、性能表现及 Firefox Android 的缺口作为依据。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | M 系列与 A 系列芯片中的端侧 NPU；采用统一内存架构 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 进行调用 |
| WebGPU | "browser GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 支持 |
| WebLLM | "browser LLM runtime" | MLC-LLM 项目；Apache 2.0 协议；提供 OpenAI 兼容 JS API |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 家族 |
| TRT Edge-LLM | "edge TensorRT" | 2026 年 TensorRT-LLM 的边缘移植版本；支持 EAGLE-3 + NVFP4 |
| Unified memory | "shared pool" | CPU 与 NPU 共享同一物理内存；无数据拷贝开销 |
| Bandwidth-bound | "memory limited" | 解码过程受限于每秒读取权重的字节数 |
| Core ML | "Apple conversion" | Apple 用于构建 ANE 原生模型的框架 |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## 延伸阅读

- [《2026 年端侧大模型现状报告》](https://v-chandra.github.io/on-device-llms/) — 技术全景与基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor 系列介绍。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘移植版本发布公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 架构设计与性能基准。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生模型转换指南。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 预转换模型仓库。
