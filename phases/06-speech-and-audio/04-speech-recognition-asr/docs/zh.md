# 语音识别 (ASR) — CTC、RNN-T、Attention

> 语音识别是在每个时间步进行音频分类，再由一个懂得英语和静音的序列模型将其粘合起来。CTC、RNN-T 和 attention 是实现它的三种方式。选择一种并理解其原理。

**类型：** Build
**语言：** Python
**前置知识：** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**时间：** ~45 分钟

## 问题

你有一段 10 秒、16 kHz 的音频片段。你想要一个字符串："turn on the kitchen lights"。挑战在于结构性问题：音频帧与字符并非一一对应。单词 "okay" 可能耗时 200 毫秒或 1200 毫秒。静音穿插在话语中。某些音素比其他音素更长。输出 token 的数量事先无法确定。

三种方法解决了这个问题：

1. **CTC (Connectionist Temporal Classification)。** 逐帧输出 token 概率，包括一个特殊的 *blank*。在解码时折叠重复项和 blank。非自回归，速度快。用于 wav2vec 2.0、MMS。
2. **RNN-T (Recurrent Neural Network Transducer)。** 联合网络在给定编码器帧和先前 token 的情况下预测下一个 token。可流式处理。用于 Google 的端侧 ASR、NVIDIA Parakeet。
3. **Attention encoder-decoder。** 编码器将音频压缩为隐藏状态，解码器通过交叉注意力自回归地生成 token。用于 Whisper、SeamlessM4T。

2026 年，LibriSpeech test-clean 上的 SOTA WER 为 1.4% (Parakeet-TDT-1.1B, NVIDIA) 和 1.58% (Whisper-Large-v3-turbo)。差距很小；部署差异却很大。

## 概念

![三种 ASR 方法：CTC、RNN-T、attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 帧级别的 `V+1` token 分布（V 个字符 + blank）。对于长度为 `U < T` 的目标字符串 `y`，任何折叠后能得到 `y` 的帧对齐都有效。CTC 损失对所有此类对齐求和。推理：逐帧 argmax，折叠重复项，移除 blank。

优势：非自回归、可流式处理、零前瞻。缺点：*条件独立性假设* — 每帧预测独立于其他帧，因此没有内部语言模型。通过外部 LM 经束搜索或浅层融合来修正。

**RNN-T 直觉。** 增加一个 *predictor* 网络来嵌入 token 历史，以及一个 *joiner* 将 predictor 状态与编码器帧结合，生成 `V+1` 上的联合分布（`+1` 为空/不发射）。显式建模了 CTC 忽略的条件依赖。可流式处理，因为每一步仅依赖于过去的帧和过去的 token。

优势：可流式处理 + 内部 LM。缺点：训练更复杂且内存消耗大（3D 损失网格）；RNN-T 损失核本身就是一个完整的库类别。

**Attention encoder-decoder。** 编码器（6-32 层 transformer）处理 log-mel 帧。解码器（6-32 层 transformer）通过交叉注意力关注编码器输出来自回归地生成 token。无对齐约束 — attention 可以查看音频中的任意位置。除非限制 attention（分块 Whisper-Streaming，2024），否则不可流式处理。

优势：离线 ASR 质量最高，使用标准 seq2seq 工具易于训练。缺点：自回归延迟与输出长度成正比；不经过工程处理无法流式处理。

### WER：唯一指标

**Word Error Rate** = `(S + D + I) / N`，其中 S=替换，D=删除，I=插入，N=参考词数。与词级别的 Levenshtein 编辑距离一致。越低越好。WER 高于 20% 通常不可用；低于 5% 对于朗读语音达到人类水平。2026 年标准基准上的数据：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 大小 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是基于 encoder-decoder 或 RNN-T 的。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 动手实现

### 步骤 1：greedy CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复项，丢弃 blank。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带 LM 融合的前缀树束搜索；这是概念骨架。

### 步骤 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20 倍实时速度运行。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 进行流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和延续状态；使用支持它的库（NeMo 用于 Parakeet，`transformers` pipeline 配合 `chunk_length_s`）。

## 使用建议

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒性 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘、移动端、<500 ms 延迟 | Whisper-Tiny 量化版或 Moonshine (2024) |
| 长音频 | 基于 VAD 分块的 Whisper (WhisperX) |
| 领域特定（医疗、法律） | 微调 wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍在犯的陷阱

- **没有 VAD。** 对静音运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用 VAD 把关。
- **字符 vs 词 vs 子词 WER。** 报告规范化后（小写、去除标点）的词级别 WER。
- **语言 ID 漂移。** Whisper 的自动语言识别会将嘈杂片段错误路由到日语或威尔士语；在已知时强制指定 `language="en"`。
- **长片段不分块。** Whisper 有 30 秒窗口。对更长的内容使用 `chunk_length_s=30, stride=5`。

## 交付

保存为 `outputs/skill-asr-picker.md`。为给定的部署目标选择模型、解码策略、分块方式和 LM 融合方案。

## 练习

1. **简单。** 运行 `code/main.py`。它对手工构造的 CTC 输出进行 greedy 解码，并计算与参考的 WER。
2. **中等。** 在步骤 2 中正确实现前缀树束搜索（考虑 blank 合并规则）。在 10 个示例的合成数据集上与 greedy 方法比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 个话语的 WER。与已发表数据比较。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | blank-token 损失 | 对所有帧到 token 对齐的边际化；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一 token 预测器；处理词序。 |
| Attention enc-dec | Whisper 风格 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 你报告的数字 | 词级别的 `(S+D+I)/N`。 |
| Blank | 空 | CTC 中的特殊 token，表示"此帧无发射"。 |
| LM fusion | 外部语言模型 | 在束搜索期间加入加权 LM 对数概率。 |
| VAD | 静音门 | 语音活动检测器；修剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年经典论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 年 Open ASR Leaderboard 榜首。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25+ 模型的实时基准。
