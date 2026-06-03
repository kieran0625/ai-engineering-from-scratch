# 音频生成

> 音频是 16-48 kHz 的一维信号。五秒的片段就有 80-240k 个采样点。没有任何 transformer 能直接处理这样的序列长度。2026 年所有生产级音频模型的解决方案都是一样的：神经编解码器（Encodec、SoundStream、DAC）将音频压缩为 50-75 Hz 的离散 token，然后由 transformer 或扩散模型生成 token。

**类型：** Build
**语言：** Python
**前置知识：** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**时间：** ~45 分钟

## 问题背景

三类音频生成任务：

1. **文本转语音（TTS）。** 给定文本，生成语音。干净语音是窄带的，且有强烈的语音结构——基于 token 的 transformer 能很好地解决。代表：VALL-E (Microsoft)、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、风格），生成音乐。分布更宽广。代表：MusicGen (Meta)、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 声音设计。** 给定提示，生成环境音或拟音（Foley）。代表：AudioGen、AudioLDM 2、Stable Audio Open。

这三类任务都基于相同的底层架构：神经音频编解码器 + token 自回归或扩散生成器。

## 核心概念

![音频生成：编解码器 token + transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec (Meta, 2022)、SoundStream (Google, 2021)、Descript Audio Codec (DAC, 2023)。卷积编码器将波形压缩为逐时间步的向量；残差向量量化（RVQ）将每个向量转换为一组 K 个码本索引的级联。解码器则反向还原。24 kHz 音频以 2 kbps 压缩，使用 8 层 RVQ 码本，75 Hz 的帧率 = 每秒 600 个 token。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 之上的两种生成范式

**Token 自回归。** 将 RVQ token 展平为序列，运行 decoder-only transformer。MusicGen 使用"延迟并行"（delayed parallel）方式，以逐流偏移的方式并行发射 K 个码本流。VALL-E 从文本提示 + 3 秒语音样本生成语音 token。

**隐空间扩散。** 将编解码器 token 打包为连续隐变量，或用分类扩散建模。Stable Audio 2.5 在连续音频隐变量上使用流匹配（flow matching）。AudioLDM 2 使用文本到 mel 再到音频的扩散。

2024-2026 年的趋势：流匹配在音乐生成领域占据优势（推理更快、样本更干净），而 token-AR 仍在语音领域占主导，因为它天然因果且易于流式传输。

## 生产环境概览

| 系统 | 任务 | 骨干网络 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | ~300ms 首 token |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | ~200ms |
| NaturalSpeech 3 | TTS | 隐空间流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频隐变量流匹配 | ~10s 生成 1 分钟片段 |
| Suno v4 | 完整歌曲 | 未公开；疑似 token-AR | ~30s 每首歌 |
| Udio v1.5 | 完整歌曲 | 未公开 | ~30s 每首歌 |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | ~5s 生成 5s 片段 |
| Riffusion v2 | 音乐 | 频谱图扩散 | ~10s |

## 动手实现

`code/main.py` 模拟核心思想：在由两种不同"风格"生成的合成"音频 token"序列上训练一个微型 next-token transformer（风格 A 为交替的低高 token，风格 B 为单调递增）。基于风格进行条件化并采样。

### 步骤 1：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤 2：训练微型 token 预测器

基于风格条件化的 bigram 风格预测器。核心模式是：编解码器 token → 交叉熵训练 → 自回归采样。

### 步骤 3：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。持续进行 20-40 个 token。

## 常见陷阱

- **编解码器质量决定输出上限。** 如果编解码器无法忠实还原某种声音，生成器再好也无济于事。DAC 是目前开源最佳选择。
- **RVQ 误差累积。** 每层 RVQ 建模前一层的残差。第 1 层的误差会传播。在更高层使用 temperature 0 采样有助于缓解。
- **音乐结构。** 30 秒的 token 在 75 Hz 下超过 20k 个 token，对 transformer 构成挑战。MusicGen 使用滑动窗口 + 提示续写；Stable Audio 使用更短片段 + 交叉淡入淡出。
- **边界伪影。** 生成片段之间的交叉淡入淡出需要小心的叠接相加（overlap-add）处理。
- **对干净数据的渴求。** 音乐生成器需要数万小时的授权音乐。Suno / Udio 的 RIAA 诉讼（2024 年）将这一问题暴露于公众视野。
- **语音克隆伦理。** VALL-E / XTTS / ElevenLabs 仅需 3 秒样本加文本提示即可克隆声音。每个生产模型都需要滥用检测 + 退出名单。

## 实际应用

| 任务 | 2026 年技术栈 |
|------|------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（已验证同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速生成 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音助手 | GPT-4o realtime 或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付实践

保存 `outputs/skill-audio-brief.md`。该技能接收音频需求简报（任务、时长、风格、音色、授权）并输出：模型 + 托管方案、提示格式（风格标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链路、种子协议，以及评估计划（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B）。

## 练习

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成序列是否符合该风格的模式。
2. **中等。** 添加延迟并行解码：模拟 2 个 token 流，它们必须保持 1 步的偏移。训练联合预测器。
3. **困难。** 使用 HuggingFace transformers 本地运行 MusicGen-small。用三个不同提示生成 10 秒片段；对风格一致性进行 A/B 测试。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| Codec | "神经压缩" | 音频的编码器/解码器；典型输出为 50-75 Hz 的 token。 |
| RVQ | "残差 VQ" | K 个量化器的级联；每个建模前一层的残差。 |
| Token | "一个编解码器符号" | 码本中的离散索引；典型大小为 1024 或 2048。 |
| Delayed parallel | "偏移码本" | 以交错偏移的方式发射 K 个 token 流，以减少序列长度。 |
| Flow matching | "2024 年音频领域的突破" | 扩散的直线路径替代方案；采样更快。 |
| Voice prompt | "3 秒样本" | 说话人嵌入或 token 前缀，用于引导克隆音色。 |
| Mel spectrogram | "那个视觉图" | 对数幅度感知频谱图；许多 TTS 系统使用。 |
| Vocoder | "Mel 到波形" | 将 mel 频谱图还原为音频的神经组件。 |

## 生产注记：音频是一个流式问题

音频是唯一用户期望*边生成边播放*的输出模态，而非一次性全部输出。在生产中，这意味着 TPOT（Time Per Output Token）至关重要，因为用户的收听速度就是目标吞吐量——而非阅读速度。对于 16kHz 音频以 ~75 token/秒（Encodec）编码，服务器必须每用户每秒生成 ≥75 个 token 才能保持播放流畅。

两个架构层面的影响：

- **流匹配音频模型无法轻易流式化。** Stable Audio 2.5 和 AudioCraft 2 一次性渲染固定长度的片段。要流式化，需将片段分块并重叠边界——类似滑动窗口扩散——这相比编解码器 AR 模型会增加 100-300ms 的延迟开销。

如果产品是"实时语音聊天"或"实时音乐续写"，选择编解码器 AR 路径。如果是"提交后渲染 30 秒片段"，流匹配在质量和总延迟上更优。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 首个广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年基于流匹配的文本到音乐生成。
