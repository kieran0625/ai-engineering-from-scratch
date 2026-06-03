# 流式语音到语音 — Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 推出了单一模型，能够以 200 毫秒延迟同时听和说。Hibiki 则逐块进行语音到语音翻译。两者都摒弃了 ASR → LLM → TTS 流水线，转而采用基于 Mimi 编解码器 token 的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 13（神经音频编解码器）、Phase 6 · 11（实时音频）、Phase 7 · 05（完整 Transformer）
**时间：** ~75 分钟

## 问题所在

从第 11 课和第 12 课构建的每个语音智能体都有一个约 300-500 毫秒的基本延迟下限：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有自己的最小延迟。你可以优化和并行化，但流水线的结构决定了上限。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有流水线会怎样？如果一个模型直接接收音频并持续输出音频，将文本作为中间的"内心独白"而非必需阶段，会怎样？

答案是**全双工语音到语音**。理论延迟 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）。在单张 L4 GPU 上实际延迟 200 毫秒。这是顶级流水线式语音智能体延迟的一半。

## 核心概念

![Moshi 架构：两条并行 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两条 Mimi 编解码器流，均为 12.5 Hz × 8 个码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自身音频（由 Moshi 生成）

**Transformer。** 一个 70 亿参数的 Temporal Transformer 处理两条流和一个文本"内心独白"流。在每个 80 毫秒步骤中，它：

1. 接收最新的用户 Mimi token（8 个码本）。
2. 接收最近的 Moshi Mimi token（8 个码本，已生成）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi token（通过小型 Depth Transformer，8 个码本）。

三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话时听到用户；可以在用户打断时自我打断；可以进行反馈性回应（"嗯哼"）而不打断其主要话语。

**Depth Transformer。** 在一个帧内，8 个码本不是并行预测的——它们存在码本间依赖关系。一个小型 2 层"depth transformer"在 80 毫秒内顺序预测它们。这是自回归编解码器语言模型的标准分解方式（VALL-E、VibeVoice 也采用此方法）。

### 内心独白文本为何有帮助

如果没有显式文本，模型必须在声学流中隐式建模语言。Moshi 的洞见是：强制模型在输出音频的同时发出文本 token。文本流本质上是 Moshi 所说内容的转录。这提升了语义连贯性，使更换语言模型头部更容易，并免费提供转录文本。

### Hibiki：流式语音到语音翻译

相同架构，在翻译对上训练。源语言音频输入，目标语言音频输出，持续进行。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句级数据 + GRPO 强化学习进行延迟优化。

最初支持四种语言对；可适应新语言，约需 1000 小时数据。

### Kyutai 技术栈全景（2026）

- **Moshi** — 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同步语音翻译
- **Kyutai STT** — 流式 ASR（500 毫秒或 2.5 秒前瞻）
- **Kyutai Pocket TTS** — 1 亿参数 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合上述技术的完整流水线

在 L40S GPU 上的吞吐量：64 个并发会话，速度为实时 3 倍。

### Sesame CSM — 近亲

Sesame CSM（2025）采用类似思路——Llama-3 骨干网络 + Mimi 编解码器头部。但 CSM 是单向的（接收上下文 + 文本，生成语音），而非全双工。它是市场上"语音存在感"最佳的 TTS；与 Moshi 的全双工能力不完全相同。

### 2026 年性能数据

| 模型 | 延迟 | 用途 | 许可证 |
|------|------|------|--------|
| Moshi | 200 毫秒（L4） | 全双工英语 / 法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 相同 | 5 种语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 毫秒 TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 毫秒 | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 毫秒 | 闭源，Google API | 商业 |

## 动手构建

### 步骤 1：接口

Moshi 暴露一个 WebSocket 服务器，接收 80 毫秒块的 Mimi 编码音频，并返回 80 毫秒块的 Mimi 编码音频。双向持续进行。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 步骤 3：训练目标（概念性）

对于每个 80 毫秒帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文本先于音频预测（内心独白）；音频在 depth transformer 内按码本顺序预测。

### 步骤 4：Moshi 的优势与局限

Moshi 的优势：

- 廉价硬件上端到端低于 250 毫秒。
- 自然的反馈性回应和打断。
- 无需流水线胶水代码。

Moshi 的局限：

- 工具调用（未针对此训练；需要单独的 LLM 路径）。
- 长推理（Moshi 是约 80 亿参数的对话模型，非 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数企业生产用例（2026 年仍在使用流水线）。

## 使用场景

| 场景 | 选择 |
|------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具的企业智能体 | 流水线（第 12 课），非 Moshi |
| 上下文中的定制语音 TTS | Sesame CSM |
| 任意语言的语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 常见陷阱

- **工具调用受限。** Moshi 是对话模型，非智能体框架。与流水线结合以实现工具调用。
- **特定语音条件。** Moshi 使用单一训练人格；克隆需要单独的训练运行。
- **语言覆盖。** 法语 + 英语表现优秀；其他语言有限。Hibiki-Zero 有帮助，但仍需训练数据。
- **资源成本。** 完整的 Moshi 会话占用一个 GPU 槽位；非廉价的共享租户部署模式。

## 交付实践

保存为 `outputs/skill-duplex-pipeline.md`。为语音智能体工作负载选择流水线与全双工架构，并说明理由。

## 练习题

1. **简单。** 运行 `code/main.py`。它以符号方式模拟双流 + 内心独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户话音结束到 Moshi 响应开始的实际延迟。
3. **困难。** 取第 12 课的流水线智能体，与 Moshi 在 20 组匹配测试话语上对比 P50 延迟。撰写分析，说明流水线在架构上仍然获胜的情况。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| 全双工（Full-duplex） | 同时听和说 | 同一模型上两条音频流同时活跃。 |
| 内心独白（Inner monologue） | 模型的文本流 | Moshi 在输出音频的同时发出文本 token。 |
| Depth Transformer | 码本间预测器 | 在 80 毫秒帧内预测 8 个码本的小型 transformer。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 码本；语义 + 声学；驱动 Moshi。 |
| 流式 S2S | 音频 → 音频实时 | 逐块翻译/对话，无流水线阶段。 |
| 反馈性回应（Back-channeling） | "嗯哼"式反应 | Moshi 可发出小型确认而不打断其话轮。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无需对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规格。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + 服务器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 闭源商业同类产品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS 框架。
