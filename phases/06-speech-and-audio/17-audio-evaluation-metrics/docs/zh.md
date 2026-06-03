# 音频评估 — WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 无法衡量，便无法交付。本课列出 2026 年各类音频任务的核心指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、音频语言模型（MMAU、LongAudioBench）、音乐（FAD、CLAP）以及说话人（EER）。还有可供对比的排行榜。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09（模型评估）
**时间：** ~60 分钟

## 问题

每项音频任务都有多个指标，各自衡量不同的维度。用错指标会导致模型在仪表板上看起来很好，但在生产环境中表现糟糕。2026 年标准清单：

| 任务 | 主要指标 | 次要指标 |
|------|---------|---------|
| ASR | WER | CER · RTFx · 首 token 延迟 |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| 语音克隆 | SECS（ECAPA 余弦） | MOS · CER |
| 说话人验证 | EER | minDCF · 工作点处的 FAR / FRR |
| 说话人分割 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | macro F1 · 逐类别召回率 |
| 音乐生成 | FAD | CLAP · 听测小组 MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 概念

![音频评估矩阵 — 指标 vs 任务 vs 2026 排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（词错误率）。** `(S + D + I) / N`。评分前转小写、去除标点、归一化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。&lt; 5% = 人类水平的朗读语音。

**CER（字错误率）。** 公式相同，但基于字符级别。用于词边界模糊的有声调语言（普通话、粤语）。

**RTFx（实时因子倒数）。** 每 wall-clock 秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约为 30×。

**首 token 延迟。** 从音频输入到首个转录 token 的 wall-clock 时间。对流式应用至关重要。Deepgram Nova-3：~150 ms。

### TTS 指标

**MOS（平均意见分）。** 1-5 分的人工评分。金标准但速度慢。每个样本需 20+ 名听众，每个模型需 100+ 样本。

**UTMOS（2022-2026）。** 基于学习的 MOS 预测器。在标准基准上与人类 MOS 的相关性约 0.9。F5-TTS：UTMOS 3.95；真值：4.08。

**SECS（说话人编码器余弦相似度）。** 用于语音克隆。参考音频与克隆输出之间的 ECAPA 嵌入余弦相似度。&gt; 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip。** 对 TTS 输出运行 Whisper，计算与输入文本的 WER。用于发现可懂度退化。2026 SOTA：&lt; 2% CER。

**TTFA（首音频时间）。** Wall-clock 延迟。Kokoro-82M：~100 ms；F5-TTS：~1 s。

### 语音克隆专用

**SECS + MOS + CER** 三联指标。SECS 高但 MOS 低意味着音色对但不自然；反之则意味着声音自然但说话人不对。

### 说话人验证

**EER（等错误率）。** 错误接受率等于错误拒绝率时的阈值。ECAPA on VoxCeleb1-O：0.87%。

**minDCF（最小检测代价）。** 选定工作点（通常为 FAR=0.01）的加权代价。比 EER 更贴近生产实际。

### 说话人分割

**DER（说话人分割错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误检语音 + 说话人混淆，各自占比。AMI 会议：DER ~10-20% 为现实水平。pyannote 3.1 + Precision-2 商业方案：录制良好的音频上 &lt;10% DER。

**JER（Jaccard 错误率）。** DER 的替代指标，对短片段偏置更鲁棒。

### 音频分类

多标签：**mAP（平均精度均值）** 跨所有类别。AudioSet：BEATs-iter3 为 0.548 mAP。

多类别互斥：**top-1、top-5 准确率**。Speech Commands v2：99.0% top-1（Audio-MAE）。

类别不平衡：**macro F1** + **逐类别召回率**。逐类别报告 — 聚合准确率会掩盖哪些类别失败。

### 音乐生成

**FAD（Fréchet 音频距离）。** 真实与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small on MusicCaps：4.5。MusicLM：4.0。越低越好。

**CLAP Score。** 使用 CLAP 嵌入的文本-音频对齐分数。&gt; 0.3 = 合理对齐。

**听测小组 MOS。** 消费级音乐仍以此为准。Suno v5 在 TTS Arena 上 ELO 1293（来自成对人类偏好）。

### 音频语言基准

**MMAU（Massive Multi-Audio Understanding）。** 10k 音频-QA 对。

**MMAU-Pro。** 1800 道难题，四个类别：语音 / 声音 / 音乐 / 多音频。四选一随机猜 25%。Gemini 2.5 Pro 整体约 60%；多音频约 22%，所有模型皆然。

**LongAudioBench。** 多分钟的片段，带语义查询。Audio Flamingo Next 击败 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 字幕生成基准。SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户语音结束到首个可听响应的 wall-clock 时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

输出端的 **WER / MOS**。

**插话响应性。** 从用户打断到助手静音的时间。目标 &lt; 150 ms。

### 2026 年排行榜

| 排行榜 | 追踪内容 | URL |
|--------|---------|-----|
| Open ASR Leaderboard (HF) | 英语 + 多语言 + 长音频 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，ELO 来自成对投票 | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU music subset | 音乐 LALM | （在 MMAU 内） |
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 动手实现

### 步骤 1：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤 2：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：说话人验证的 EER（与第 6 课代码相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 应用

每次部署都要搭配固定的评估套件，在每次模型更新时运行。三条基本原则：

1. **评分前归一化。** 转小写、去标点、展开数字。报告归一化规则。
2. **报告分布，而非平均值。** 延迟用 P50/P95/P99。分类用逐类别召回率。MMAU 用逐类别。
3. **运行一个标准公开基准。** 即使生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告也能让评审者进行同类比较。

## 陷阱

- **UTMOS 外推。** 在 VCTK 风格干净语音上训练；对嘈杂 / 克隆 / 情感化音频评分较差。
- **MOS 小组偏置。** 20 名 Amazon Mechanical Turk 工人 ≠ 20 名目标用户。如果 stakes 高，花钱请领域专家小组。
- **FAD 依赖参考集。** 跨模型比较时使用相同的参考分布。
- **聚合 WER。** 整体 5% WER 可能掩盖口音语音上 30% WER。按人口统计切片报告。
- **公开基准饱和。** 大多数前沿模型在标准基准上已接近天花板。构建反映自身流量的内部保留集。

## 交付

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在 toy 输入上计算 WER / CER / EER / SECS / 类 FAD / 类 MMAU。
2. **中等。** 构建 TTS 往返 WER 套件。将 Kokoro 或 F5-TTS 输出通过 Whisper 运行。在 50 个 prompt 上计算 WER。标记 WER &gt; 10% 的 prompt。
3. **困难。** 在 MMAU-Pro 语音 + 多音频子集（各 50 题）上评估你在第 10 课选择的 LALM。报告逐类别准确率并与公开数字比较。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| WER | ASR 分数 | `(S+D+I)/N`，归一化后的词级别。 |
| CER | 字符 WER | 用于有声调语言或字符级系统。 |
| MOS | 人工意见 | 1-5 分；20+ 听众 × 100 样本。 |
| UTMOS | 机器学习 MOS 预测器 | 学习得到的模型；与人类 MOS 相关性约 0.9。 |
| SECS | 语音克隆相似度 | 参考与克隆之间的 ECAPA 余弦相似度。 |
| EER | 说话人验证分数 | FAR = FRR 时的阈值。 |
| DER | 说话人分割分数 | (FA + 漏检 + 混淆) / 总计。 |
| FAD | 音乐生成质量 | VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每 wall-clock 秒的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带归一化工具的 WER/CER 库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 基于学习的 MOS 预测器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 音乐生成标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 人工投票 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) — 音频自监督学习基准。
