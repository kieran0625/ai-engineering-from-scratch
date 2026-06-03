# 说话人识别与验证

> ASR 问的是“他们说了什么？”说话人识别问的是“谁在说话？”数学公式看起来一样——嵌入向量加余弦相似度——但每一个生产决策都取决于一个 EER 数字。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图与梅尔滤波器组），Phase 5 · 22（嵌入模型）
**时间：** ~45 分钟

## 问题

用户说出一个通行短语。你想知道：这是否是其所声称的人（*验证*，1:1），或是注册库中的某个人（*识别*，1:N）？或者都不是——这是否是一个未知说话人（*开集*）？

2018 年之前：GMM-UBM + i-vector。EER 尚可，但对信道变化（手机 vs 笔记本）和情绪敏感。2018–2022：x-vector（用角度边际训练的 TDNN 骨干网络）。2022 年之后：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年，该领域由三个模型和一个指标主导。

这个指标就是 **EER**——等错误率。设定决策阈值使得误接受率 = 误拒绝率。交叉点即为 EER。每篇论文、每个排行榜、每次采购都会用到。

## 概念

![注册 + 验证流程：嵌入 + 余弦 + EER](../assets/speaker-verification.svg)

**流程。** 注册：录制目标说话人 5–30 秒音频；计算固定维度嵌入（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试语音嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020 年提出，2026 年仍占主导）。** 强调信道注意力、传播与聚合——时延神经网络。一维卷积块配合 Squeeze-Excitation、多头注意力池化，随后线性层映射到 192 维。在 VoxCeleb 1+2（2,700 位说话人，110 万条语音）上用加性角度边际损失（AAM-softmax）训练。

**WavLM-SV（2022 年后）。** 用 AAM 损失微调预训练的 WavLM-large SSL 骨干网络。质量更高但更慢——300+ MB 对比 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典模型；在 CPU / 边缘设备上仍然有用。

**AAM-softmax。** 标准 softmax 在角度空间中加入边际 `m`：`cos(θ + m)` 用于正确类别。强制类间角度分离。典型 `m=0.2`，尺度 `s=30`。

### 打分

- **余弦** 相似度，计算注册与测试嵌入之间。基于阈值的决策。
- **PLDA（概率 LDA）。** 将嵌入投影到潜在空间，使得同说话人 vs 不同说话人具有闭式似然比。在余弦基础上叠加，可降低 10–20% EER。2020 年前为标准方法；现仅用于闭集场景。
- **分数归一化。** `S-norm` 或 `AS-norm`：用一组冒名顶替者的均值和标准差对每个分数进行归一化。跨域评估必不可少。

### 你应该知道的数字（2026）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量（A100） |
|------|-----------------|--------|---------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 说话人分割

“谁在什么时候说话”——多说话人片段中的问题。流程：VAD → 分段 → 每段计算嵌入 → 聚类（凝聚式或谱聚类）→ 平滑边界。现代方案：`pyannote.audio` 3.1，将说话人分割 + 嵌入 + 聚类封装在一个调用中。2026 年在 AMI 上的 SOTA DER 约为 ~15%（从 2022 年的 23% 下降）。

## 动手实现

### 步骤 1：基于 MFCC 统计量的简易嵌入

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

离 SOTA 差得远——仅用于教学。`code/main.py` 在合成说话人数据上使用此方案作为概念验证。

### 步骤 2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤 3：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。两者都需报告。

### 步骤 4：使用 SpeechBrain 进行生产部署

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 步骤 5：使用 pyannote 进行说话人分割

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用场景

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 闭集 1:1 验证，边缘设备 | ECAPA-TDNN + 余弦阈值 |
| 开集验证，云端 | WavLM-SV + AS-norm |
| 说话人分割（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放 / 深度伪造检测） | AASIST 或 RawNet2 |
| 极小嵌入式（KWS + 注册） | Titanet-Small (NeMo) |

## 陷阱

- **信道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。务必在目标信道上评估。
- **语音过短。** 测试音频低于 3 秒时，EER 急剧恶化。
- **带噪注册。** 一条带噪注册会污染锚点。使用 ≥3 条干净样本并取平均。
- **跨条件固定阈值。** 务必在目标域的留出开发集上调整阈值。
- **对未归一化嵌入使用余弦。** 先进行 L2 归一化；否则幅度会占主导。

## 交付

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调整计划以及欺诈防护措施。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成“说话人”（不同音调轮廓），注册，在 100 对试验列表上计算 EER。
2. **中等。** 在 30 条 VoxCeleb1 语音（5 位说话人 × 每人 6 条）上使用 SpeechBrain ECAPA。用余弦 vs PLDA 计算 EER。
3. **困难。** 使用 `pyannote.audio` 构建完整的注册 → 分割 → 验证流程。在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| EER |  headline 指标 | 误接受率 = 误拒绝率时的阈值。 |
| Verification | 1:1 | “这是 Alice 吗？” |
| Identification | 1:N | “谁在说话？” |
| Open-set | 可能存在未知说话人 | 测试集可包含未注册说话人。 |
| Enrollment | 注册 | 计算说话人的参考嵌入。 |
| AAM-softmax | 损失函数 | 带加性角度边际的 Softmax；强制簇分离。 |
| PLDA | 经典打分方法 | 概率 LDA；嵌入之上的似然比打分。 |
| DER | 分割指标 | 说话人分割错误率——漏检 + 虚警 + 混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 经典深度嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020–2026 年主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) —— 用于 SV 和分割的 SSL 骨干网络。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— 生产级分割 + 嵌入方案。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 各模型当前 EER 排名。
