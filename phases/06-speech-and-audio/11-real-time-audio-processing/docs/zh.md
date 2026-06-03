# 实时音频处理

> 批处理流水线处理文件。实时流水线在下一个 20 毫秒到来之前处理当前的 20 毫秒。每一个对话式 AI、广播演播室和电话机器人，都取决于这一延迟预算的成败。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**时间：** ~75 分钟

## 问题

你想要一个感觉有生命力的语音助手。人类对话轮转的延迟约为 230 毫秒（从沉默到回应）。超过 500 毫秒就会感觉像机器人；超过 1500 毫秒就感觉坏了。2026 年完整的**听 → 理解 → 回应 → 说**循环的预算为：

| 阶段 | 预算 |
|------|------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首个 token） | 100 ms |
| TTS（首块音频） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai, 2024）实现了 200 毫秒全双工。GPT-4o-realtime（2024）约为 320 毫秒。2022 年的级联流水线延迟为 2500 毫秒。10 倍的提升来自三项技术：(1) 全流式处理，(2) 带部分结果的异步流水线，(3) 可中断生成。

## 概念

![流式音频流水线：环形缓冲区、VAD 门控、中断](../assets/real-time.svg)

**帧 / 块 / 窗口。** 实时音频以固定大小的块流动。常见选择：20 毫秒（16 kHz 下 320 个采样点）。下游所有环节必须跟上这一节奏。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。避免热路径上的内存分配。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 的环形缓冲区 = 32,000 个采样点。

**VAD（语音活动检测）。** 当无人说话时阻断下游工作。Silero VAD 4.0（2024）在 CPU 上每 30 毫秒帧运行时间 <1 毫秒。`webrtcvad` 是较早的替代方案。

**流式 ASR。** 音频到达时输出部分转录的模型。Parakeet-CTC-0.6B 流式模式（NeMo, 2024）在 320 毫秒延迟下达到 2–5% WER。Whisper-Streaming（Macháček 等, 2023）将 Whisper 分块实现近流式处理，延迟约 2 秒。

**中断。** 当用户在助手说话时插话，你必须 (a) 检测到打断，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。全部要在 100 毫秒内完成，否则用户会感知到助手"聋了"。

**WebRTC Opus 传输。** 20 毫秒帧，48 kHz，自适应码率 8–128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的技术栈。

**抖动缓冲区。** 网络包可能乱序或延迟到达。抖动缓冲区重新排序并平滑；太小 → 可闻间隙，太大 → 延迟增加。典型值 60–80 毫秒。

### 常见陷阱

- **线程争用。** Python 的 GIL + 重量级模型可能饿死音频线程。使用 C 回调音频库（sounddevices、PortAudio），让 Python 远离热路径。
- **采样率转换延迟。** 流水线内部重采样增加 5–20 毫秒。要么提前重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是 Kokoro 这样的快速 TTS，首次请求也有 100–200 毫秒预热。缓存模型 + 在首次真实对话前用虚拟运行预热。
- **回声消除。** 没有 AEC 时，TTS 输出会重新进入麦克风，触发 ASR 识别机器人自己的声音。WebRTC AEC3 是开源默认方案。

## 构建

### 步骤 1：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32,000 个采样点 = 2 秒。

### 步骤 2：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤 3：流式 ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤 4：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖于异步 I/O 和可取消的 TTS 流式传输。WebRTC peerconnection.stop() 作用于音频轨道是标准做法。

## 使用

2026 年技术栈：

| 层级 | 选择 |
|------|------|
| 传输 | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **缓冲 500 毫秒求稳妥。** 缓冲区*就是*你的延迟下限。缩小它。
- **不固定线程优先级。** 音频回调运行在低于 UI 线程的优先级上 = 负载高时出现爆音。
- **TTS 块太小。** 低于 200 毫秒的块会让声码器伪影可闻。320 毫秒块是最佳点。
- **没有抖动缓冲区。** 真实网络有抖动；没有平滑处理就会出现爆音。
- **单次错误处理。** 音频流水线必须能容错。一个异常就会终止整个会话。

## 交付

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，为每个阶段设定具体的延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；为模拟的 10 秒流打印各阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个直通循环，以 20 毫秒帧处理你的麦克风，并在每帧打印 VAD 状态。
3. **困难。** 使用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量端到端延迟。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Ring buffer | 循环队列 | 固定大小、无锁（或 SPSC 锁）的音频帧 FIFO。 |
| VAD | 静音门 | 标记语音与非语音的模型或启发式方法。 |
| Streaming ASR | 实时语音转文字 | 音频到达时输出部分文本；有界前瞻。 |
| Jitter buffer | 网络平滑器 | 对乱序包重新排序队列；典型 60–80 毫秒。 |
| AEC | 回声消除 | 消除扬声器到麦克风的反馈路径。 |
| Barge-in | 用户打断 | 系统在 TTS 过程中检测到用户语音；必须取消播放。 |
| Full duplex | 双向同时 | 用户和机器人可以同时说话；Moshi 是全双工。 |

## 延伸阅读

- [Macháček 等 (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 毫秒延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — 生产级音频智能体编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — 亚毫秒级 VAD，Apache 2.0 协议。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源回声消除方案。
