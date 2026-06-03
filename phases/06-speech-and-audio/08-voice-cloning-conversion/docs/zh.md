# 语音克隆与语音转换

> 语音克隆用别人的声音朗读你的文本。语音转换把你的声音改写成别人的声音，同时保留你说的内容。两者都基于同一种分解：将说话人身份与内容分离。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 6 · 06（说话人识别），阶段 6 · 07（TTS）
**时间：** ~75 分钟

## 问题背景

2026 年，一段 5 秒的音频片段就足以在消费级 GPU 上生成任何人的高质量语音克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 等都支持零样本或少样本克隆。这项技术既是福音（无障碍 TTS、配音、辅助语音），也是武器（诈骗电话、政治深度伪造、知识产权盗窃）。

两个密切相关的任务：

- **语音克隆（TTS 侧）：** 文本 + 5 秒参考语音 → 该语音的音频。
- **语音转换（语音侧）：** 源音频（A 说 X）+ B 的参考语音 → B 说 X 的音频。

两者都将波形分解为（内容、说话人、韵律），然后将一个来源的内容与另一个来源的说话人重新组合。

2026 年你必须遵守的关键约束：**水印和同意机制在欧盟（AI 法案，2026 年 8 月生效）和加利福尼亚州（AB 2905，2025 年生效）是法律强制要求的**。你的流水线必须输出不可听水印，并拒绝非自愿克隆。

## 核心概念

![语音克隆与转换：分解、交换说话人、重新组合](../assets/voice-cloning.svg)

**零样本克隆。** 将 5 秒片段传递给一个在数千个说话人上训练过的模型。说话人编码器将片段映射为说话人嵌入；TTS 解码器以该嵌入加文本为条件进行生成。

使用方：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**少样本微调。** 录制 5-30 分钟目标语音。用 LoRA 微调基础模型一小时。质量从"还行"跃升到"无法区分"。Coqui 和 ElevenLabs 都支持这种模式；社区也将其用于 F5-TTS。

**语音转换（VC）。** 两个家族：

- **识别-合成。** 运行类 ASR 模型提取内容表示（例如，软音素后验概率，PPGs），然后用目标说话人嵌入重新合成。对语言和口音鲁棒。使用方：KNN-VC (2023)、Diff-HierVC (2023)。
- **解耦。** 训练一个自编码器，在瓶颈处的潜在空间中将内容、说话人和韵律分离。在推理时交换说话人嵌入。质量较低但速度更快。使用方：AutoVC (2019)、VITS-VC 变体。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox —— 将音频视为 SoundStream / EnCodec 的离散 token，在 codec token 上训练大型自回归或流匹配模型。短提示下的质量可与 ElevenLabs 媲美。

### 伦理部分，不是附加组件

**水印。** PerTh (Perth) 和 SilentCipher (2024) 在音频中不可感知地嵌入 ~16-32 位 ID。 survives 重新编码、流媒体和常见编辑。生产就绪的开源方案。

**同意机制。** 每个克隆输出必须配对可验证的同意记录。"我，Rohit，于 2026-04-22，授权此语音用于 X 目的。" 存储在防篡改日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发布。ASVspoof 2025 挑战赛公布了针对 ElevenLabs、VALL-E 2 和 Bark 输出的最先进检测器的 EER 为 0.8–2.3%。

### 数据（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|------|---------|------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众来说通常与目标无法区分。

## 动手构建

### 步骤 1：用识别-合成方法进行分解（main.py 中的纯代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念简单；实现复杂度主要在 `tts_model` 和说话人编码器中。

### 步骤 2：用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须完全匹配音频；不匹配会破坏对齐。

### 步骤 3：用 KNN-VC 进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM 提取源语音和目标池的逐帧嵌入，然后将每个源帧替换为池中的最近邻。非参数方法，仅需一分钟目标语音即可工作。

### 步骤 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 位有效载荷，MP3 重新编码和轻度噪声后仍可检测。

### 步骤 5：同意机制

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 实际应用

2026 年技术栈：

| 场景 | 选择 |
|------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（改写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 常见陷阱

- **参考转录不对齐。** F5-TTS 等要求参考文本与参考音频完全匹配，包括标点符号。
- **混响参考。** 回声会毁掉克隆。录制干声，近距离拾音。
- **情感不匹配。** 用"欢快"的参考训练会产生所有内容的欢快克隆。将参考情感与目标用途匹配。
- **语言泄漏。** 克隆英语说话人后让模型说法语，通常仍带有英语口音；使用跨语言模型（XTTS、VALL-E X）。
- **没有水印。** 2026 年 8 月起在欧盟法律上不可发布。

## 发布上线

保存为 `outputs/skill-voice-cloner.md`。设计一个包含同意机制 + 水印 + 质量目标的克隆或转换流水线。

## 练习

1. **简单。** 运行 `code/main.py`。演示通过计算交换前后两个"说话人"之间的余弦相似度来展示说话人嵌入交换。
2. **中等。** 使用 OpenVoice v2 克隆你自己的声音。测量参考与克隆之间的 SECS。通过 Whisper 测量 CER。
3. **困难。** 对 20 个克隆应用 SilentCipher 水印，经过 128 kbps MP3 编码+解码后检测有效载荷。报告位准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 零样本克隆 | 5 秒就够了 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | 音素后验图 | 用作语言无关内容表示的逐帧 ASR 后验。 |
| KNN-VC | 最近邻转换 | 将每个源帧替换为目标池中最近的帧。 |
| 神经编解码器 TTS | VALL-E 风格 | 基于 EnCodec/SoundStream token 的自回归模型。 |
| 水印 | 不可听签名 | 嵌入音频中的比特，survives 重新编码。 |
| SECS | 克隆保真度 | 目标与克隆说话人嵌入之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen 等 (2024). F5-TTS](https://arxiv.org/abs/2410.06885) —— 开源 SOTA 零样本克隆。
- [Baevski 等 / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) —— 神经编解码器 TTS。
- [Qian 等 (2019). AutoVC](https://arxiv.org/abs/1905.05879) —— 基于解耦的语音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) —— 基于检索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) —— 生产就绪的 32 位音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) —— 检测器与合成器的军备竞赛，2026 年更新。
