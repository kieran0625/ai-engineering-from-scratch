# 语音智能体：Pipecat 与 LiveKit

> 语音智能体在 2026 年已成为一等生产级类别。Pipecat 提供基于帧的 Python 管道（VAD → STT → LLM → TTS → 传输层）。LiveKit Agents 通过 WebRTC 将 AI 模型连接至用户。对于高端技术栈，端到端的生产延迟目标为 450–600 毫秒。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 12（工作流模式）
**预计时间：** 约 60 分钟

## 学习目标

- 描述 Pipecat 的基于帧的管道：DOWNSTREAM（源到接收器）和 UPSTREAM（控制）。
- 列举标准语音管道的各个阶段以及 Pipecat 支持的传输协议。
- 解释 LiveKit Agents 的两类语音智能体（MultimodalAgent、VoicePipelineAgent）及其适用场景。
- 总结 2026 年的生产环境延迟预期，以及这些预期如何驱动架构选型。

## 问题所在

语音智能体并非简单地在文本循环上外挂一个 TTS。延迟预算极其严苛（约 600 毫秒），部分音频输出是常态，话轮检测依赖模型，传输协议涵盖从传统电话 SIP 到 WebRTC 不等。你要么自己构建基于帧的管道（Pipecat），要么依托平台方案（LiveKit）。

## 核心概念

### Pipecat (pipecat-ai/pipecat)

- 基于帧的 Python 管道框架。
- `Frame` → `FrameProcessor` 链。
- 两种数据流向：
  - **DOWNSTREAM**（下游）—— 源 → 接收器（音频输入，TTS 输出）。
  - **UPSTREAM**（上游）—— 反馈与控制（取消、指标、插话打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命周期，并提供用于指标/追踪/RTVI 的观察者机制。

典型管道流程：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

支持的传输协议：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话能力（状态机）。Pipecat Cloud 是托管运行时环境。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 将 AI 模型桥接至用户端。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类语音智能体：
  - **MultimodalAgent** —— 通过 OpenAI Realtime 或同等服务直接处理音频。
  - **VoicePipelineAgent** —— STT → LLM → TTS 级联；提供文本级别的精细控制。
- 基于 Transformer 模型实现语义话轮检测。
- 原生支持 MCP 集成。
- 支持通过 SIP 进行电话通信。
- 通过 LiveKit Inference 提供 50 多种免 API Key 的模型；通过插件还可扩展 200 多种。

### 商业平台

Vapi（优化后的高端技术栈约 450–600 毫秒）和 Retell（180 次测试调用平均端到端约 600 毫秒）均构建于上述技术之上。当你希望获得托管式语音技术栈且无需专门维护 WebRTC 团队时，可选择此类平台。

### 常见陷阱

- **未处理插话打断（Barge-in）。** 用户打断时，智能体仍在说话。需在 Pipecat 中发送 UPSTREAM 取消帧，LiveKit 中有对应机制。
- **忽略 STT 置信度。** 低置信度的转录文本被当作绝对真理喂给 LLM。应设置置信度阈值拦截或要求用户确认。
- **TTS 句子中途截断。** 当管道在 utterance 中途取消时，TTS 需要感知并切断音频输出。
- **忽视延迟预算。** 每个组件都会增加 50–200 毫秒延迟。上线前务必计算整个链路的总耗时。

### 2026 年典型延迟参考

- VAD：20–60 毫秒
- STT 部分输出：100–250 毫秒
- LLM 首 Token 生成：150–400 毫秒
- TTS 首段音频生成：100–200 毫秒
- 传输层往返延迟（RTT）：30–80 毫秒

端到端 450–600 毫秒属于高端水平。800–1200 毫秒较为常见。超过 1500 毫秒会明显感觉卡顿/不可用。

## 动手实践

`code/main.py` 是一个基于帧的示例管道，包含：

- `Frame` 类型（音频、转录文本、纯文本、TTS 音频、控制指令）。
- `Processor` 接口及 `process(frame)` 方法。
- 五个阶段的管道（VAD → STT → LLM → TTS → 传输层）以脚本化处理器形式实现。
- 一个 UPSTREAM 取消帧，用于演示插话打断功能。

运行方式：

```
python3 code/main.py
```

日志追踪展示了正常流转过程，以及一次在 utterance 中途触发 TTS 停止的插话打断操作。

## 选型建议

- **Pipecat**：适合需要完全控制的场景——自定义处理器、Python 优先、可插拔提供商。
- **LiveKit Agents**：适合以 WebRTC 为核心的部署及电话通信场景。
- **Vapi / Retell**：适合无需 WebRTC 团队即可快速托管语音智能体的场景。
- **OpenAI Realtime / Gemini Live**：适合直接音频输入/输出的场景（配合 MultimodalAgent）。

## 生产部署

`outputs/skill-voice-pipeline.md` 提供了一个符合 Pipecat 架构的语音管道脚手架，包含 VAD + STT + LLM + TTS + 传输层，并内置插话打断处理逻辑。

## 练习

1. 为你的示例管道添加一个指标观察者：统计每秒各阶段处理的帧数。延迟主要累积在哪个环节？
2. 实现基于置信度阈值的 STT：低于阈值时，提示“能否请您重复一遍？”
3. 添加语义话轮检测：使用简单规则——如果转录文本以“？”结尾，则判定为用户发言结束。
4. 阅读 Pipecat 的传输层文档。将标准库传输替换为 SmallWebRTCTransport 配置（存根）。
5. 对比同一查询下 OpenAI Realtime 与 STT+LLM+TTS 级联方案的延迟。文本级别的控制会带来多少额外的延迟成本？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Frame | “事件” | 管道中类型化的数据单元（音频、转录文本、纯文本、控制指令） |
| Processor | “管道阶段” | 带有 process(frame) 方法的处理器 |
| DOWNSTREAM | “正向流转” | 源到接收器：音频输入，语音输出 |
| UPSTREAM | “反馈流转” | 控制信号：取消、指标、插话打断 |
| VAD | “语音活动检测” | 检测用户何时正在说话 |
| 语义话轮检测 | “智能话轮结束判断” | 基于模型的决策，判断用户是否已说完 |
| MultimodalAgent | “直连音频智能体” | 音频输入，音频输出；中间不经过文本 |
| VoicePipelineAgent | “级联智能体” | STT + LLM + TTS；提供文本级别的控制 |

## 延伸阅读

- [Pipecat 官方文档](https://docs.pipecat.ai/getting-started/introduction) —— 基于帧的管道、处理器、传输协议
- [LiveKit Agents 官方文档](https://docs.livekit.io/agents/) —— WebRTC 与语音基础组件
- [Vapi](https://vapi.ai/) —— 托管式语音平台
- [Retell AI](https://www.retellai.com/) —— 托管式语音，附带延迟基准测试
