# 文本转语音（TTS) —— 从 Tacotron 到 F5 和 Kokoro

> ASR 将语音转换为文本；TTS 将文本转换为语音。2026 年的技术栈分为三部分：文本 → token、token → mel、mel → 波形。每个部分都有一个能在笔记本电脑上运行的默认模型。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02 (频谱图与 Mel)、Phase 5 · 09 (Seq2Seq)、Phase 7 · 05 (完整 Transformer)
**时间：** ~75 分钟

## 问题

你有一个字符串："Please remind me to water the plants at 6 pm." 你需要一段 3 秒的自然音频，具有正确的韵律（停顿、重音），"plants" 的发音元音正确，并且在 CPU 上运行时间低于 300 毫秒，用于实时语音助手。你还需要能够切换声音、处理语码混合输入（"remind me at 6 pm, daijoubu?"），并且在人名上不出错。

现代 TTS 流水线如下：

1. **文本前端。** 规范化文本（日期、数字、邮箱），转换为音素或子词 token，预测韵律特征。
2. **声学模型。** 文本 → mel 频谱图。Tacotron 2 (2017)、FastSpeech 2 (2020)、VITS (2021)、F5-TTS (2024)、Kokoro (2024)。
3. **声码器。** Mel → 波形。WaveNet (2016)、WaveRNN、HiFi-GAN (2020)、BigVGAN (2022)、2024+ 的神经编解码声码器。

到 2026 年，声学模型 + 声码器的界限因端到端扩散和流匹配模型而模糊。但三部分的思维模型对于调试仍然有效。

## 概念

![Tacotron、FastSpeech、VITS、F5/Kokoro 对比](../assets/tts.svg)

**Tacotron 2 (2017)。** Seq2seq：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器输出 mel 帧。慢（AR），长文本不稳定。仍被引用为基线。

**FastSpeech 2 (2020)。** 非自回归。时长预测器输出每个音素获得多少 mel 帧。单遍，比 Tacotron 快 10 倍。损失一些自然度（单调对齐），但到处都在用。

**VITS (2021)。** 联合训练编码器 + 基于流的时长 + HiFi-GAN 声码器，端到端变分推断。高质量，单模型。2022–2024 年主导开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2 (2024, Coqui)。

**F5-TTS (2024)。** 流匹配上的扩散 Transformer。自然韵律，5 秒参考音频即可零样本语音克隆。2026 年开源 TTS 排行榜榜首。335M 参数。

**Kokoro (2024)。** 小型（82M）、CPU 可运行、实时英语 TTS 最佳。封闭词汇、仅英语、apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业最先进。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色声音在 2026 年主导有声书制作。

### 声码器演进

| 年代 | 声码器 | 延迟 | 质量 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅离线 | 发布时 SOTA |
| 2018 | WaveRNN | ~实时 | 良好 |
| 2020 | HiFi-GAN | 100× 实时 | 接近人类 |
| 2022 | BigVGAN | 50× 实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与 AR 模型集成 | 离散 token，比特高效 |

到 2026 年，大多数 "TTS" 模型都是文本到波形的端到端；mel 频谱图只是内部表示。

### 评估

- **MOS（平均意见分）。** 1–5 分，众包。仍是黄金标准； painfully slow。
- **CMOS（比较 MOS）。** A 对 B 偏好。每次注释的置信区间更紧。
- **UTMOS、DNSMOS。** 无参考神经 MOS 预测器。用于排行榜。
- **通过 ASR 的 CER（字符错误率）。** 将 TTS 输出通过 Whisper，与输入文本计算 CER。可理解性的代理指标。
- **SECS（说话人嵌入余弦相似度）。** 语音克隆质量。

2026 年 LibriTTS test-clean 上的数据：

| 模型 | UTMOS | CER（通过 Whisper） | 大小 |
|-------|-------|-------------------|------|
| 真实音频 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建

### 步骤 1：将输入音素化

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。避免将原始文本直接输入到低于 VITS 质量的模型。

### 步骤 2：运行 Kokoro（2026 年 CPU 默认）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M 参数。

### 步骤 3：使用 F5-TTS 进行语音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考片段 + 其转录文本；F5 克隆韵律和音色。

### 步骤 4：从头实现 HiFi-GAN 声码器

太大，无法放入教程脚本，但结构是：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：对抗（短窗口判别器）+ mel 频谱图重建损失 + 特征匹配损失。已商品化 —— 从 `hifi-gan` 仓库或 nvidia-NeMo 使用预训练检查点。

### 步骤 5：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 实时英语语音助手 | Kokoro (CPU) 或 XTTS v2 (GPU) |
| 从 5 秒参考克隆语音 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 表现力 / 情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年的开源领先者：**F5-TTS 用于质量，Kokoro 用于效率**。除非你是历史学家，否则不要碰 Tacotron。

## 陷阱

- **没有文本规范化器。** "Dr. Smith" 读作 "Doctor" 还是 "Drive"？"2026" 读作 "twenty twenty six" 还是 "two zero two six"？在音素化之前先规范化。
- **OOV 专有名词。** "Ghumare" → "ghyu-mair"？为未知 token 配备备用的字素到音素模型。
- **削波。** 声码器输出很少削波，但推理时 mel 缩放不匹配可能导致超出 ±1.0。务必 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；你的下游流水线期望 16 kHz → 重采样，否则会出现混叠。

## 交付

保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计 TTS 流水线。

## 练习

1. **简单。** 运行 `code/main.py`。从玩具词汇构建音素词典，估计每个音素的时长，并打印一个假 "mel" 时间表。
2. **中等。** 安装 Kokoro，在声音 `af_bella` 和 `am_adam` 下合成同一句子。比较音频时长和主观质量。
3. **困难。** 录制一段你自己的 5 秒参考片段。使用 F5-TTS 克隆它。报告参考与克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| Phoneme | 声音单位 | 抽象声音类别；英语中有 39 个（ARPABet）。 |
| Duration predictor | 每个音素持续多久 | 非 AR 模型输出；每个音素的整数帧数。 |
| Vocoder | Mel → 波形 | 将 mel 频谱图映射到原始样本的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；2020–2024 年主导。 |
| MOS | 主观质量 | 人类评分者的 1–5 平均意见分。 |
| SECS | 语音克隆指标 | 目标与输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英语领先者 | 82M 参数模型，Apache 2.0。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端基于流。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在使用的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年 CPU 友好型英语 TTS。
