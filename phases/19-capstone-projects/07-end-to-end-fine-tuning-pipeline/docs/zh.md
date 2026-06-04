# 毕业设计 07 —— 端到端微调流水线（数据到 SFT 到 DPO 到服务部署）

> 在自有数据上训练的 8B 模型，基于自有偏好进行 DPO 对齐，经过量化与推测解码，并以可测量的 $/1M tokens 成本提供服务。2026 年的开源技术栈为 Axolotl v0.8、TRL 0.15、用于快速迭代的 Unsloth、用于量化的 GPTQ/AWQ/GGUF，以及用于服务部署的带 EAGLE-3 的 vLLM 0.7。本毕业设计的目标是完整、可复现地运行整个流水线——输入 YAML 配置，输出可服务的接口——并在《2026 模型开放框架》下发布一份模型卡片。

**类型：** 毕业设计
**语言：** Python（流水线脚本）、YAML（配置文件）、Bash（Shell 脚本）
**前置要求：** 第 2 阶段（机器学习）、第 3 阶段（深度学习）、第 7 阶段（transformers）、第 10 阶段（从零构建 LLM）、第 11 阶段（LLM 工程）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**预计耗时：** 35 小时

## Problem

2026 年，每个严肃的 AI 团队都会随时准备一套微调流水线。并非因为他们会发布前沿的基础模型，而是下游适配工作——领域特定的 SFT、基于标注偏好的 DPO、用于推测解码的蒸馏草稿模型、配合 EAGLE-3 的服务部署——才是产生可衡量收益的关键所在。Axolotl v0.8 负责处理多 GPU 的 SFT 配置。TRL 0.15 负责 DPO 和 GRPO。Unsloth 能让你实现快速的单 GPU 迭代。vLLM 0.7 配合 EAGLE-3 可在不损失质量的前提下将解码吞吐量提升 2-3 倍。工具链已经成熟；真正的功夫在于 YAML 配置、数据清洗规范以及评估纪律。

你将使用任务特定数据，对一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）依次进行 SFT 和 DPO 训练，随后进行量化以用于服务部署，并使用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 来衡量性能提升。你还需要根据《2026 模型开放框架》编写一份模型卡片。核心目标是可复现性——一条命令即可从头到尾重新运行整个流水线。

## Concept

该流水线包含五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII（个人身份信息）清除、针对公开基准测试污染的数据划分规范性检查。**SFT**：Axolotl YAML 配置、8xH100 上的 ZeRO-3、余弦学习率调度、序列打包、2-3 个 epoch。**DPO 或 GRPO**：TRL 配置、1 个 epoch、偏好对（人工标注或模型评判）、beta 参数调优。**量化**：GPTQ + AWQ + GGUF 以提供部署灵活性。**服务部署**：vLLM 0.7 配合 EAGLE-3 推测头（或使用带 SpecForge 的 SGLang）、K8s 部署、基于队列等待时间的 HPA（水平自动扩缩容）。

消融实验是交付物的一部分：在三个任务特定基准上对比仅 SFT、SFT+DPO 与 SFT+GRPO。服务指标：batch size 为 1 / 8 / 32 时的 tokens/s、EAGLE-3 接受率、$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡片：偏见评估、可复现性随机种子、数据许可协议。

## Architecture

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Stack

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量过滤，Presidio 用于 PII 清除
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8（支持 ZeRO-3、Flash Attention 3、序列打包）
- 偏好对齐：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 快速迭代
- 量化：GPTQ (Marlin)、AWQ、GGUF（通过 llama.cpp）
- 服务部署：vLLM 0.7 配合 EAGLE-3 推测解码（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin、基于队列等待指标的 HPA
- 可观测性：W&B 用于训练监控，Langfuse 用于推理追踪

## Build It

1. **数据流水线**。在原始语料上运行 Datatrove 去重。应用 Nemotron-CC 风格的质量分类器。使用 Presidio 清除 PII。使用明确的随机种子划分训练集/验证集。
2. **污染检查**。针对每一个验证集划分，计算其与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的 MinHash 相似度。拒绝任何重叠数据。
3. **Axolotl SFT**。使用包含 ZeRO-3、FA3、序列打包的 YAML 配置。在 8xH100 上训练 2-3 个 epoch。日志记录至 W&B。
4. **TRL DPO / GRPO**。加载 SFT 检查点，在偏好对上运行一个 epoch 的 DPO（或在数学/代码任务上使用可验证奖励运行 GRPO）。进行 beta 参数扫描。
5. **量化**。生成三种量化版本：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M（用于 llama.cpp）。记录模型大小与标称吞吐量。
6. **配合推测解码的服务部署**。使用 vLLM 0.7 配置，搭载通过 Red Hat Speculators 训练的 EAGLE-3 草稿头。测量 batch size 为 1 / 8 / 32 时的接受率与尾部延迟。在同一评估标准下，报告与 Anthropic / OpenAI 相比的 $/1M tokens 成本。
7. **评估矩阵**。在基础模型、仅 SFT、SFT+DPO、SFT+GRPO 上运行 lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成对比表格。
8. **安全评估**。在开发集上测试 Llama Guard 4 通过率。使用 ShieldGemma-2 进行输出过滤。
9. **模型卡片**。遵循 MOF 2026 模板：包含数据、训练、评估、安全、许可部分，以及附带 YAML 配置和提交 SHA 的可复现性说明。

## Use It

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Ship It

`outputs/skill-finetuning-pipeline.md` 描述了交付物要求。一条命令即可让数据依次经过 SFT、DPO、量化、服务部署与评估全流程，最终输出一份模型卡片与服务接口。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 相对于基础模型的评估提升 | 目标任务上的性能增益（MMLU-Pro、MT-Bench-v2、任务特定基准） |
| 20 | 流水线可复现性 | 一条命令配合相同随机种子即可从头到尾重新运行 |
| 20 | 数据清洗规范 | 去重率、PII 清除覆盖率、污染检查通过 |
| 20 | 服务效率 | batch size 为 1/8/32 时的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡片与安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## Exercises

1. 在同一个任务特定基准上分别运行仅 SFT、SFT+DPO 与 SFT+GRPO。报告哪种偏好对齐方法表现最佳及其优势幅度。
2. 将 Llama 3.3 8B 替换为 Qwen3 14B。在质量匹配的情况下测量 $/1M tokens 成本。
3. 对比领域数据与通用 ShareGPT 数据上的 EAGLE-3 接受率。报告差异值及其对延迟预算的影响。
4. 注入 1% 的污染数据（将 MMLU-Pro 答案泄露至训练集中）并重新运行评估。观察 MMLU-Pro 准确率出现不合理的飙升。构建一个能拦截此类情况的污染检查 CI 门禁。
5. 添加 LoRA SFT 作为全量微调的替代方案。在内存占用降低 10 倍的情况下测量质量差距。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | “SFT 训练器” | 统一基于 YAML 驱动的 SFT、DPO 与知识蒸馏训练框架 |
| TRL | “偏好对齐工具” | Hugging Face 提供的用于 LLM 进行 DPO、GRPO、PPO 的库 |
| GRPO | “组相对策略优化” | DeepSeek R1 采用的强化学习配方，具备可验证奖励机制 |
| EAGLE-3 | “推测解码草稿模型” | 可提前预测 N 个 token 的草稿头；由 vLLM 使用目标模型进行验证 |
| MOF | “模型开放框架” | 2026 年用于从数据、代码、许可证维度对模型发布进行评级的标准 |
| 污染检查 | “划分规范性检查” | 基于 MinHash 检测测试集数据泄露至训练集中的情况 |
| 接受率 | “EAGLE / MTP 指标” | 目标模型接受的草稿 token 所占比例 |

## Further Reading

- [Axolotl 官方文档](https://axolotl-ai-cloud.github.io/axolotl/) —— 权威的 SFT / DPO 训练参考
- [TRL 官方文档](https://huggingface.co/docs/trl) —— DPO 和 GRPO 的参考实现
- [Unsloth](https://github.com/unslothai/unsloth) —— 单 GPU 快速迭代参考
- [DeepSeek R1 论文 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) —— GRPO 方法论
- [vLLM + EAGLE-3 官方文档](https://docs.vllm.ai) —— 权威的服务部署技术栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) —— 替代性的推测解码训练器
- [2026 模型开放框架 (MOF)](https://isocpp.org/) —— 开源发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) —— 标准的评估运行工具
