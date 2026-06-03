# 音频-语言模型 — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频-语言模型能够对语音、环境音和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 持平。Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开源与闭源之间的差距已基本消除——除了多音频任务，在该任务上所有模型都接近随机水平。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04 (ASR)、Phase 12 · 03 (视觉-语言模型)、Phase 7 · 10 (Audio Transformers)
**时间：** ~45 分钟

## 问题

你有 5 秒音频：狗叫、有人喊"stop!"，然后安静。有用的提问跨越多个维度：

- **转录。** "说了什么？"——ASR 领域。
- **语义推理。** "这个人有危险吗？"——需要联合理解狗叫 + 喊叫 + 安静。
- **音乐推理。** "什么乐器在演奏旋律？"
- **长音频检索。** "在这 90 分钟的讲座中，讲师在哪里讲解了梯度下降？"

一个能用单一 prompt 回答所有这些问题的模型就是**音频-语言模型**（LALM / ALM）。与纯 ASR 不同：LALM 生成自由形式的自然语言答案，而不仅仅是转录文本。

## 概念

![音频-语言模型：音频编码器 + 投影器 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年的 LALM 都有相同的骨架：

1. **音频编码器。** Whisper encoder · BEATs · CLAP · WavLM · 或各模型自定义的编码器。
2. **投影器。** 线性层或 MLP，将音频编码器特征桥接到 LLM 的 token 嵌入空间。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本 + 音频 token；生成文本。

训练：

- **阶段 1。** 冻结编码器 + LLM；仅在 ASR / 字幕数据上训练投影器。
- **阶段 2。** 在指令遵循型音频任务（QA、推理、音乐理解）上进行全量 / LoRA 微调。
- **阶段 3（可选）。** 语音输入 / 语音输出增加语音解码器。Qwen2.5-Omni 和 AF3-Chat 采用此方案。

### 2026 年模型地图

| 模型 | 骨干网络 | 音频编码器 | 输出模态 | 获取方式 |
|------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### 基准现实检验（2026）

**MMAU-Pro。** 1800 个 QA 对，覆盖语音 / 声音 / 音乐 / 混合。包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频列对所有人都是致命的。** 4 选 1 多选题的随机概率 = 25%；大多数模型得分就在附近。LALM 仍难以比较两个片段。

### LALM 在 2026 年的适用场景

- **呼叫中心录音合规审计。** "客服是否提到了必需的披露声明？"
- **无障碍辅助。** 向听障用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **播客 / 会议章节划分。** 语义摘要，而非仅仅是说话人轮次。
- **音乐目录分析。** "找出所有 B 段有转调的曲目。"

### 尚不适用（暂时）的场景

- 细粒度音乐理论（和弦级别以下）。
- 长对话中的说话人归因推理（超过 10 分钟后性能下降）。
- 多音频比较（22-26%  barely above random）。
- 实时流式推理（大多数是离线批处理推理）。

## 动手构建

### 步骤 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常只有 1-3 个线性层。在 ASR 对（音频 → 转录）上训练它是阶段 1 的代理任务。

### 步骤 3：MMAU / LongAudioBench 基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

按类别（语音 / 声音 / 音乐 / 多音频）分别报告。聚合数字会掩盖模型的失败点。

## 实际使用

| 任务 | 2026 年推荐 |
|------|-----------|
| 自由形式音频 QA（开源） | Qwen2.5-Omni-7B |
| 最佳开源长音频模型 | Audio Flamingo Next |
| 最佳闭源模型 | Gemini 2.5 Pro |
| 语音输入 / 语音输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | Gemini 2.5 Pro via API，配合 RAG 检索内部政策文档 |

## 常见陷阱

- **过度信任多音频能力。** 如果你的任务需要"哪个片段包含 X"，随机概率级别的性能是真实的。
- **长音频退化。** 超过 10 分钟后，大多数模型的说话人归因会失效。先进行说话人分离（第 6 课），再总结。
- **对静音产生幻觉。** 使用 Whisper 编码器的 LALM 继承的同类 Whisper 问题。使用 VAD 门控。
- **基准 cherry-picking。** 厂商博客只展示最佳类别。自己运行 MMAU-Pro 多音频子集。

## 交付

保存为 `outputs/skill-alm-picker.md`。为给定音频理解任务选择 LALM + 基准子集 + 输出模态（text vs speech）。

## 练习

1. **简单。** 运行 `code/main.py` 查看玩具投影器模式 + 假 LALM 路由 (audio-embedding, text-tokens) → output tokens。
2. **中等。** 在 100 个 MMAU-Pro 语音条目上评测 Qwen2.5-Omni-7B。与论文报告数字对比。
3. **困难。** 构建最小化音频字幕基线：BEATs 编码器 + 2 层投影器 + 冻结 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。与 SALMONN 在 Clotho-AQA 上对比。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| Projector | Adapter | 将音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | The benchmark | 10k 音频-QA 对，覆盖语音、声音、音乐。 |
| MMAU-Pro | Harder MMAU | 1800 个多音频 / 重推理问题。 |
| LongAudioBench | Long-form eval | 多分钟片段配合语义查询。 |
| Voice-in / voice-out | Speech-native | 模型接收语音并输出语音，无需经过文本中转。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频领先者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — 2026 实时排名。
