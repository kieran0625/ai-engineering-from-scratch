# 综合项目 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个体验良好的语音代理，其端到端延迟需低于 800 毫秒，能够准确判断用户何时停止说话、处理插话（barge-in），并能在不卡顿的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年均达到了这一标准。它们采用相同的架构形态：流式 ASR、话轮检测器（turn-detector）、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，并在每一跳都设置了激进的延迟预算。构建一个这样的系统，测量 WER（词错误率）和 MOS（平均意见得分）以及误切断率，并在丢包环境下运行它。

**类型：** 综合项目
**语言：** Python（代理与管道），TypeScript（Web 客户端）
**前置要求：** 第 6 阶段（语音与音频）、第 7 阶段（transformers）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（代理）、第 17 阶段（基础设施）
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17
**预计耗时：** 30 小时

## Problem

语音是 2025-2026 年发展最快的 AI 交互类别。技术门槛每季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 均使首音频出时间低于 800 毫秒成为可能。考核标准不仅仅是延迟。更重要的是交互体验：不打断用户、不被意外打断、能从句子中途的插话中恢复、在对话中途调用工具且不阻塞音频、在移动网络抖动下保持稳定。

仅靠拼接三个 REST 调用无法实现。架构必须是端到端的流水线流式传输。一旦开始构建，各种故障模式就会显现：为电话音频调优的 VAD 却因背景电视声触发、等待永远不会出现的标点符号的话轮检测器、发出前缓冲 400 毫秒的 TTS。本综合项目的目标是：在高负载下逐一修复这些问题，并发布一份延迟与质量报告。

## Concept

该管道包含五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录文本）、**话轮检测**（VAD 加上一个小型话轮检测模型，用于读取部分转录文本以获取完成提示）、**LLM**（一旦判定话轮完成即流式输出 token）、**TTS**（在首个 LLM token 出现后约 200 毫秒内流式输出音频）。

三个跨领域关注点。**插话（Barge-in）**：当用户在代理说话时开始发言，TTS 会立即取消，ASR 随即接管。**工具调用**：对话中途的功能调用（天气、日历）必须在侧信道运行且不阻塞音频；如果延迟超过 300 毫秒，代理会预先填充确认 token（“请稍等……”）。**背压（Backpressure）**：在丢包情况下，部分转录文本会被暂存，VAD 会提高语音门限阈值，且代理应避免在未收到确认的消息上重叠发声。

衡量标准是量化的。在 15 dB SNR 的 Hamming VAD 基准测试中，WER 低于 8%。在 100 次实测通话中，首音频出时间的 p50 低于 800 毫秒。误切断率低于 3%。TTS 的 MOS 高于 4.2。单台 g5.xlarge 实例支持 50 路并发通话。这些指标即为交付成果。

## Architecture

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Stack

- 传输层：LiveKit Agents 1.0（WebRTC）配合 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首条部分转录低于 300 毫秒）或自托管 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 配合 LiveKit 话轮检测器（读取部分转录文本的小型 transformer）
- LLM：OpenAI GPT-4o-realtime（紧密集成）、Gemini 2.5 Flash Live，或级联 Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（首字节延迟最低）、ElevenLabs Flash v3，或开源 Orpheus（自托管）
- 工具：FastMCP 侧信道用于天气/日历/预订；若工具调用耗时 >300 毫秒，代理会预播填充语
- 可观测性：OpenTelemetry 语音 span、Langfuse 语音 trace（含音频回放）
- 部署：单台 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；使用托管 API 以获得最低延迟

## Build It

1. **WebRTC 会话。** 搭建一个 LiveKit 房间和一个流式传输麦克风音频的 Web 客户端。在服务器端，附加一个加入该房间的 agent worker。

2. **ASR 流式传输。** 将 20 毫秒的 PCM 帧输入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录文本。记录每条部分转录的延迟。

3. **VAD 与话轮检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件触发时，将最新的部分转录文本送入 LiveKit 话轮检测器。仅当 VAD 检测到 500 毫秒静音且话轮检测器的完成度评分 > 0.6 时，才判定“话轮完成”。

4. **LLM 流式传输。** 话轮完成后，使用当前对话历史加最终转录文本发起 LLM 调用。流式输出 token。在首个 token 处，移交至 TTS。

5. **TTS 流式传输。** Cartesia Sonic-2 流式返回音频块。首个音频块必须在首个 LLM token 出现后 200 毫秒内离开服务器。将音频块发送至 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **插话处理。** 当 TTS 播放期间 VAD 检测到新用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新激活 ASR。发布一个 `tts_canceled` span。

7. **工具侧信道。** 注册天气和日历为函数调用工具。被调用时，并发执行请求；若在 300 毫秒内未返回结果，让 LLM 输出“请稍等，我查一下”作为填充语；待工具返回后继续。

8. **评估框架。** 录制 100 次通话。计算 WER（对照预留转录文本）、误切断率（用户说话中途 TTS 被取消）、首音频出时间 p50、TTS MOS（人工或 NISQA 自动评估），以及抖动丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 使用模拟呼叫者在单台 g5.xlarge 上驱动 50 路并发通话。测量持续的 p95 首音频出时间。

## Use It

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Ship It

`outputs/skill-voice-agent.md` 为交付成果。针对特定领域（客户支持、日程安排或自助终端），它将启动一个 LiveKit agent，并将 ASR/VAD/LLM/TTS 管道调优至符合上述衡量标准。评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录音通话中，p50 首音频出时间低于 800 毫秒 |
| 20 | 话轮切换质量 | 在 Hamming VAD 基准测试中，误切断率低于 3% |
| 20 | 工具调用正确性 | 对话中途的工具调用能返回正确数据且不阻塞音频 |
| 20 | 丢包环境下的可靠性 | 注入 3% 丢包时，WER 和话轮切换的稳定性 |
| 15 | 评估框架完整性 | 提供公开配置的可复现测量结果 |
| **100** | | |

## Exercises

1. 在 g5.xlarge 上将 Deepgram Nova-3 替换为 faster-whisper v3 turbo。测量延迟和 WER 的差异。明确 CPU 与 GPU 决策的关键影响点。

2. 添加中断仲裁策略：当用户在工具调用期间插话时，代理应如何处理？对比三种策略（硬取消、完成工具后停止、排队下一话轮）。

3. 运行对抗性话轮检测器测试：在用户句子中途设置长停顿。调优 VAD 静音阈值和话轮检测器评分阈值，以在不超过 900 毫秒的前提下实现最低误切断率。

4. 通过 Twilio 在同一 PSTN 网络上部署该 agent。对比 PSTN 与 WebRTC 的首音频出时间。解释抖动缓冲区和编解码器的差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。对比 Silero VAD v5 的误触发率与针对特定语言微调后的效果。

## Key Terms

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| 话轮检测 (Turn detection) | “语句结束” | 结合 VAD 静音状态和部分转录文本，判断用户是否已说完的 classifier |
| 插话 (Barge-in) | “中断处理” | 当 VAD 检测到新用户语音时，在中途取消 TTS 播放 |
| 首音频出 (First-audio-out) | “延迟” | 从用户停止说话到首个音频包离开服务器的时间 |
| VAD | “语音门控” | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| 抖动缓冲区 (Jitter buffer) | “音频平滑” | 客户端缓冲区，短暂持有数据包以吸收网络波动 |
| 填充语 (Filler) | “确认 token” | 当工具响应缓慢时，代理发出的短短语以避免沉默 |
| MOS | “平均意见得分” | 主观语音质量评级；NISQA 是其自动化替代方案 |

## Further Reading

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC agent 框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选 Python 优先的流式 agent 框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型的参考文档
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI architecture](https://docs.retellai.com) — 生产级语音 agent 架构
- [Vapi.ai production stack](https://docs.vapi.ai) — 备选生产环境参考
