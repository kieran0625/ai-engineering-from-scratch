# 构建语音助手流水线 —— 第六阶段综合项目

> 将第 01-11 课的所有内容整合在一起。构建一个能听、能思考、能回应的语音助手。在 2026 年，这是一个已解决的工程问题，而非研究问题 —— 但集成细节决定了它能否交付。

**类型：** 构建
**语言：** Python
**前置条件：** 第六阶段 · 04、05、06、07、11；第十一阶段 · 09（函数调用）；第十四阶段 · 01（智能体循环）
**时间：** ~120 分钟

## 问题描述

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的起始/结束。
3. 流式转录。
4. 将转录文本传递给能调用工具的 LLM（计时器、天气、日历）。
5. 将 LLM 文本流式传输至 TTS。
6. 向用户播放音频。
7. 如果用户在响应过程中打断，则停止。

延迟目标：在笔记本电脑 CPU 上，用户说完话后 800 毫秒内输出第一个 TTS 音频字节。质量目标：不遗漏单词、静默时不产生幻觉字幕、不发生语音克隆泄漏、提示注入攻击无法成功。

## 核心概念

![语音助手流水线：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频捕获。** 麦克风 → 16 kHz 单声道 → 20 毫秒分块。Python 中通常使用 `sounddevice`，生产环境则使用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（第 11 课）。** Silero VAD，阈值 0.5，最短语音 250 毫秒，静音拖尾 500 毫秒。发出"开始"和"结束"信号。
3. **流式 STT（第 4-5 课）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。支持部分转录和最终转录。
4. **支持工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具使用 JSON 模式。流式输出 token。
5. **流式 TTS（第 7 课）。** Kokoro-82M（最快的开源方案）或 Cartesia Sonic（商业方案）。LLM 输出 20 个 token 后启动 TTS。
6. **播放。** 扬声器输出；opus 编码用于低带宽网络。
7. **打断处理。** 如果 TTS 播放期间 VAD 触发，停止播放、取消 LLM、重启 STT。

### 你会遇到的三种故障模式

1. **首词截断。** VAD 启动稍晚，用户的"嘿"被遗漏。将起始阈值设为 0.3，而非 0.5。
2. **响应中打断混乱。** 用户打断后 LLM 继续生成；助手与用户同时说话。将 VAD 接入取消 LLM 的逻辑。
3. **静默幻觉。** Whisper 在静音预热帧上输出"感谢观看"。始终用 VAD 进行门控。

### 2026 年生产参考技术栈

| 技术栈 | 延迟 | 许可证 | 说明 |
|--------|------|--------|------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 毫秒 | 商业 API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 毫秒 |  mostly open | 适合 DIY |
| Moshi（全双工） | 200-300 毫秒 | CC-BY 4.0 | 单模型；不同架构，见第 15 课 |
| Vapi / Retell（托管） | 300-500 毫秒 | 商业 | 最快上线；定制受限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私 / 边缘 |

## 动手构建

### 步骤 1：带分块的麦克风捕获（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤 2：VAD 门控的轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤 3：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤 4：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤 5：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用方式

参见 `code/main.py` 获取可运行的模拟程序，它用存根模型连接了全部七个组件，即使没有硬件也能看到流水线结构。如需真实实现，将存根替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 常见陷阱

- **永久记录 PII。** 完整轮次音频在大多数司法管辖区属于 PII。保留 30 天，静态加密。
- **不支持抢话。** 用户会打断。你的助手必须停止说话。
- **TTS 阻塞。** 同步 TTS 会阻塞事件循环。使用异步或单独线程。
- **缺少工具调用错误处理。** 工具会失败。LLM 必须收到错误信息并重试一次，然后优雅降级。
- **过度激进的幻觉过滤。** 过滤过度会导致助手重复"我无法帮助您"。过滤不足则会导致它随意发言。在保留数据集上进行校准。
- **没有唤醒词选项。** 始终监听是隐私风险。添加唤醒词门控（Porcupine 或 openWakeWord）。

## 交付

保存为 `outputs/skill-voice-assistant-architect.md`。给定预算 + 规模 + 语言 + 合规约束，产出完整的技术栈规格。

## 练习

1. **简单。** 运行 `code/main.py`。它用存根模块模拟一个完整轮次，并打印各阶段延迟。
2. **中等。** 将 STT 存根替换为真实 Whisper 模型，处理预录制的 `.wav`。测量 WER 和端到端延迟。
3. **困难。** 添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`。将 LLM 路由经过这些工具，并验证当用户说"设一个 5 分钟计时器"时，正确的函数被触发且语音回复确认此事。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Turn（轮次） | 用户 + 助手的一次往返 | 一个 VAD 限界的用户语音 + 一个 LLM-TTS 响应。 |
| Barge-in（抢话） | 打断 | 用户说话时助手正在说话；助手停止。 |
| Wake word（唤醒词） | "嘿助手" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| End-pointing（端点检测） | 轮次结束 | VAD + 最短静音判断，确定用户已说完。 |
| Pre-roll（预卷缓冲） | 语音前缓冲 | VAD 触发前保留 200-400 毫秒音频，避免首词截断。 |
| Tool call（工具调用） | 函数调用 | LLM 输出 JSON；运行时调度；结果反馈回循环。 |

## 延伸阅读

- [LiveKit — 语音助手快速入门](https://docs.livekit.io/agents/) — 生产级参考。
- [Pipecat — 语音助手示例](https://github.com/pipecat-ai/pipecat) — 适合 DIY 的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 托管式原生语音路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工参考（第 15 课）。
- [Porcupine 唤醒词](https://picovoice.ai/products/porcupine/) — 唤醒词门控。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM 函数调用。
