# 语音活动检测与话轮转换 — Silero、Cobra 与 Flush Trick

> 每个语音智能体的成败取决于两个判断：用户是否在说话，以及用户是否说完了？VAD 回答第一个问题。话轮检测（VAD + 静音悬停 + 语义端点模型）回答第二个。两个中任意一个出错，你的助手要么打断用户，要么永远停不下来。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 6 · 11（实时音频），阶段 6 · 12（语音助手）
**时间：** ~45 分钟

## 问题

语音智能体在每个 20 ms 音频块上需要做出三个不同的决策：

1. **这一帧是语音吗？** — VAD。逐帧二分类。
2. **用户开始新的发言了吗？** — 起始检测。
3. **用户说完了吗？** — 端点检测（话轮结束）。

朴素的解决方案（能量阈值）在任何噪声下都会失效 — 交通噪音、键盘声、人群嘈杂声。2026 年的答案是：Silero VAD（开源、深度学习）+ 话轮检测模型（语义端点检测）+ VAD 校准的静音悬停。

## 概念

![VAD 级联：能量 → Silero → 话轮检测器 → flush trick](../assets/vad-turn-taking.svg)

### 三级 VAD 级联

**第一层：能量门控。** 成本最低。RMS 阈值设为 -40 dBFS。过滤明显的静音，但任何高于阈值的噪声都会触发。

**第二层：Silero VAD**（2020-2026，MIT 许可证）。1M 参数。训练数据覆盖 6000+ 种语言。在单 CPU 线程上每 30 ms 块约 1 ms。5% FPR 下 TPR 为 87.7%。开源默认选择。

**第三层：语义话轮检测器。** LiveKit 的话轮检测模型（2024-2026）或你自己的小型分类器。区分"句中停顿"与"说完了"。使用语言上下文（语调 + 近期词语），而非仅依赖静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；默认 > 0.5 或敏感模式 > 0.3 判定为语音。阈值越低 = 首词截断越少，误报越多。
- **最小语音时长。** 拒绝短于 250 ms 的语音 — 通常是咳嗽或椅子噪音。
- **静音悬停（端点检测）。** VAD 返回 0 后，等待 500-800 ms 再声明话轮结束。太短 → 打断用户。太长 → 感觉迟钝。
- **前滚缓冲区。** VAD 触发前保留 300-500 ms 音频。防止"嘿"被截断。

### Flush Trick（Kyutai 2025）

流式 STT 模型存在前瞻延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常语音结束后需等待该时长才能获得转录。Flush trick：当 VAD 触发语音结束时，**向 STT 发送 flush 信号**强制立即输出。STT 以约 4× 实时速度处理，因此 500 ms 缓冲约 125 ms 完成。

端到端：125 ms VAD + flush STT = 对话级延迟。

### 2026 VAD 对比

| VAD | TPR @ 5% FPR | 延迟 | 许可证 |
|-----|-------------|------|--------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是合适的默认选择。Cobra 是合规/精度升级方案。纯能量 VAD 在 2026 年生产环境中已无立足之地。

## 构建

### 步骤 1：能量门控

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：话轮结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：flush trick 骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能生效。Whisper streaming 不支持 — 它是基于块的，始终等待分块完成。

## 使用

| 场景 | VAD 选择 |
|------|---------|
| 开源、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 端侧（手机） | Silero VAD ONNX |
| 研究 / 说话人分割 | pyannote segmentation |
| 零依赖降级方案 | WebRTC VAD（遗留） |
| 需要话轮结束质量 | Silero + LiveKit turn-detector 分层 |

经验法则：除非万不得已，永远不要上线纯能量 VAD。

## 陷阱

- **固定阈值。** 安静环境可用，嘈杂环境失效。要么在设备上校准，要么切换到 Silero。
- **静音悬停过短。** 智能体打断句中。500-800 ms 是对话语音的最佳平衡点。
- **静音悬停过长。** 感觉迟钝。与目标用户做 A/B 测试。
- **无前滚缓冲区。** 丢失用户前 200-300 ms 音频。始终保持滚动前滚。
- **忽略语义端点检测。** "嗯，让我想想..." 包含长停顿。用户讨厌思考被打断。使用 LiveKit 的 turn-detector 或类似方案。

## 交付

保存为 `outputs/skill-vad-tuner.md`。为工作负载选择 VAD 模型、阈值、悬停、前滚和话轮检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟语音 + 静音 + 语音 + 咳嗽序列，测试三级 VAD。
2. **中等。** 安装 `silero-vad`，处理 5 分钟录音，调优阈值以最小化首词截断和误触发。报告精确率/召回率。
3. **困难。** 构建迷你话轮检测器：Silero VAD + 基于最近 10 个词嵌入的 3 层 MLP（使用 sentence-transformers）。在手动标注的话轮结束数据集上训练。以 10% F1 优势超越纯 Silero。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| VAD | 语音检测器 | 逐帧二分类：这是语音吗？ |
| Turn detection | 端点检测 | VAD + 静音悬停 + 语义端点。 |
| Silence hangover | 语音后等待 | 声明话轮结束前的等待时间；500-800 ms。 |
| Pre-roll | 语音前缓冲 | VAD 触发前保留 300-500 ms 音频。 |
| Flush trick | Kyutai 技巧 | VAD → flush-STT → 125 ms 替代 500 ms 延迟。 |
| Semantic endpoint | "他们是想停吗？" | 看词语而非仅看静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 点 | 标准 VAD 基准；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 开源 VAD 参考实现。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业精度领先者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 低于 200 ms 的工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 生产环境语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 遗留基线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话人分割级分割。
