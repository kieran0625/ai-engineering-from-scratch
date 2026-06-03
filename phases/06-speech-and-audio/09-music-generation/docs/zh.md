# 音乐生成 — MusicGen、Stable Audio、Suno 与授权地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 引领开源领域。技术问题已基本解决。法律问题（华纳音乐 5 亿美元和解、环球音乐集团和解）在 2025-2026 年重塑了整个领域。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图），Phase 4 · 10（扩散模型）
**时间：** ~75 分钟

## 问题

文本 → 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **器乐生成。** 如"lo-fi hip-hop drums with warm keys" → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人声 + 歌词）。** "Country song about rainy Texas nights" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件 / 可控生成。** 扩展现有片段、重新生成桥段、切换风格、分轨分离或修复。Udio 的修复 + 分轨分离是 2026 年需要匹配的功能。

## 概念

![音乐生成：token-LM 与扩散，2026 年模型图谱](../assets/music-generation.svg)

### 神经编解码器 token 上的 Token LM

Meta 的 **MusicGen**（2023，MIT 协议）及众多衍生模型：基于文本/旋律嵌入进行条件化，自回归预测 EnCodec token（32 kHz，4 个码本），再用 EnCodec 解码。3 亿 - 33 亿参数。强力基线；超过 30 秒表现下降。

**ACE-Step**（开源，40 亿 XL 版本 2026 年 4 月发布）扩展了全曲歌词条件生成。开源社区最接近 Suno 的方案。

### 梅尔频谱或潜变量上的扩散

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：压缩音频上的潜变量扩散。擅长循环、声音设计、环境纹理。结构化完整歌曲表现不佳。

**AudioLDM / AudioLDM2**：通过类 T2I 潜变量扩散实现文本到音频，泛化到音乐、音效、语音。

### 混合（生产级）— Suno、Udio、Lyria

闭源权重。可能是 AR 编解码器 LM + 基于扩散的声码器，配合专门的语音/鼓点/旋律头。Suno v5（2026）是 ELO 1293 的质量领导者。Udio v4 新增修复 + 分轨分离（贝斯、鼓点、人声可分别下载）。

### 评估

- **FAD（Fréchet Audio Distance）。** 使用 VGGish 或 PANNs 特征，计算生成音频与真实音频分布之间的嵌入级距离。越低越好。MusicGen small 在 MusicCaps 上为 4.5 FAD；SOTA ~3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 ELO 1293 领先。
- **文本-音频对齐。** 提示词与输出之间的 CLAP 分数。
- **音乐性瑕疵。** 节拍错位、乐句漂移、30 秒后结构丢失。

## 2026 年模型图谱

| 模型 | 参数量 | 时长 | 人声 | 协议 |
|------|--------|------|------|------|
| MusicGen-large | 3.3B | 30 秒 | 无 | MIT |
| Stable Audio Open | 1.2B | 47 秒 | 无 | Stability 非商业 |
| ACE-Step XL（2026 年 4 月） | 4B | &gt; 2 分钟 | 有 | Apache-2.0 |
| YuE | 7B | &gt; 2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4 分钟 | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | ? | 4 分钟 | 有 + 分轨 | 商业 |
| Google Lyria 3（闭源） | ? | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | ? | 4 分钟 | 有 | 商业 API |

## 法律环境（2025-2026）

- **华纳音乐诉 Suno 和解。** 5 亿美元。WMG 现对 Suno 上的 AI  likeness、音乐版权和用户生成曲目拥有监督权。Udio 上也有类似的 UMG 和解。
- **欧盟 AI 法案** + **加利福尼亚 SB 942**：AI 生成音乐必须披露。
- **Riffusion / MusicGen** 采用 MIT 协议，无合规负担，但也无商业人声。

安全发布模式：

1. 仅生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），按生成付费授权。
3. 在自有或授权曲库上训练（大多数企业最终选择此方案）。
4. 为生成内容添加水印 + 元数据。

## 动手构建

### 步骤 1：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种尺寸：`small`（3 亿，快速）、`medium`（15 亿）、`large`（33 亿）。Small 足以验证"想法是否成立"。

### 步骤 2：旋律条件化

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收色度图并在保留曲调的同时更换音色。适用于"把这个旋律变成弦乐四重奏"。

### 步骤 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于风格级回归测试；不能替代人工听审。

### 步骤 4：集成到 LLM-音乐工作流

结合第 7-8 课的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用场景

| 目标 | 技术栈 |
|------|--------|
| 器乐声音设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime（闭源） |
| 完整歌曲含人声（商业） | Suno v5 或 Udio v4，需明确授权 |
| 完整歌曲含人声（开源） | ACE-Step XL 或 YuE |
| 短广告 jingle | MusicGen 基于哼唱参考进行旋律条件化 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍在犯的陷阱

- **版权洗白提示词。** "Song in the style of Taylor Swift" — 商业 Suno/Udio 现已过滤这些，开源模型不会。需自行添加过滤列表。
- **30 秒后的重复 / 漂移。** AR 模型会循环。对多段生成进行交叉淡入淡出，或使用 ACE-Step 保持结构连贯性。
- **节拍漂移。** 模型会偏离 BPM。在提示词中使用 BPM 标签，并用 librosa 的 `beat_track` 进行后过滤。
- **人声可懂度。** Suno 表现优秀；开源模型往往吐字不清。若歌词重要，使用商业 API 或微调。
- **单声道输出。** 开源模型生成单声道或伪立体声。使用 proper stereo reconstruction（ezst、Cartesia 的立体声扩散）升级。

## 发布上线

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、授权策略、时长/结构计划和披露元数据。

## 练习

1. **简单。** 运行 `code/main.py`。它生成一个"生成式"和弦进行 + 鼓点模式，以 ASCII 符号表示 —— 一个音乐生成的卡通演示。如需播放，可通过任何 MIDI 渲染器回放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 基于 4 个风格提示词生成 10 秒片段，对照参考风格集测量 FAD。
3. **困难。** 使用 ACE-Step（或 MusicGen-melody），用不同音色提示词生成同一曲调的三种变体。计算与提示词的 CLAP 相似度以验证对齐。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| FAD | Audio FID | 真实与生成音频嵌入分布之间的 Fréchet 距离。 |
| Chromagram | 旋律作为音高 | 12 维每帧向量；旋律条件化的输入。 |
| Stems | 乐器音轨 | 分离的贝斯/鼓点/人声/旋律 WAV。 |
| Inpainting | 重新生成一段 | 遮罩时间窗口；模型仅重新生成该部分。 |
| CLAP | 文本-音频 CLIP | 对比式音频-文本嵌入；评估文本-音频对齐。 |
| EnCodec | 音乐编解码器 | Meta 的神经编解码器，MusicGen 使用；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 声音设计默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源 40 亿全曲生成器，2026 年 4 月。
- [Suno v5 platform docs](https://suno.com) — 商业质量领导者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 面向音乐 + 音效的潜变量扩散。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月先例。
