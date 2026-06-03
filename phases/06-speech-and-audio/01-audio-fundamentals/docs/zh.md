# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号，频谱图是其表示形式，Mel 特征则是适合机器学习的形式。每一个现代 ASR（自动语音识别）和 TTS（文本转语音）流水线都遵循这一阶梯，而第一级就是理解采样和傅里叶变换。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 1 · 06（向量与矩阵）、Phase 1 · 14（概率分布）
**时间：** ~45 分钟

## 问题

麦克风产生的是压力-时间信号，而神经网络消费的是张量。两者之间存在一整套约定，一旦违反就会产生静默型 bug：模型训练看似正常但 WER（词错误率）翻倍，TTS 输出出现嘶嘶声，或者语音克隆系统记住了麦克风而非说话人。

语音系统中的每一个 bug 都可追溯到以下三个问题之一：

1. 数据录制时的采样率是多少？模型期望的采样率又是多少？
2. 信号是否存在混叠？
3. 你是在原始样本上操作，还是在频域表示上操作？

把这些问题搞对，Phase 6 的其余内容就会迎刃而解。搞错了，即使是 Whisper-Large-v4 也会输出垃圾。

## 概念

![波形、采样、DFT 和频率分桶的可视化](../assets/audio-fundamentals.svg)

**波形（Waveform）。** `[-1.0, 1.0]` 中的一维浮点数组。以样本序号索引。转换为秒需除以采样率：`t = n / sr`。一段 10 秒、16 kHz 的音频是一个包含 160,000 个浮点数的数组。

**采样率（sr）。** 每秒的样本数。2026 年的常见采样率：

| 采样率 | 用途 |
|--------|------|
| 8 kHz | 电话、传统 VOIP。4 kHz 的奈奎斯特频率会抹掉辅音，ASR 应避免使用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 均使用 16 kHz。 |
| 22.05 kHz | 旧版 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 影视、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理。** 采样率 `sr` 可以无歧义地表示最高至 `sr/2` 的频率。`sr/2` 这一边界称为*奈奎斯特频率*。高于奈奎斯特的能量会发生*混叠*——被折叠到较低的频率中，从而污染信号。降采样前务必先进行低通滤波。

**位深度。** 16-bit PCM（有符号 int16，范围 ±32,767）是通用的交换格式。24-bit 用于音乐，32-bit float 用于内部 DSP。像 `soundfile` 这样的库读取 int16，但在 `[-1, 1]` 中暴露为 float32 数组。

**傅里叶变换。** 任何有限信号都是不同频率正弦波的叠加。离散傅里叶变换（DFT）对 `N` 个样本计算 `N` 个复系数——每个频率分桶一个。`bin k` 对应频率 `k · sr / N` Hz。幅度表示该频率的振幅，角度表示相位。

**FFT。** 快速傅里叶变换：当 `N` 为 2 的幂时，DFT 的 `O(N log N)` 算法。每个音频库底层都使用 FFT。1024 点 FFT 在 16 kHz 下产生 512 个可用频率分桶，覆盖 0–8 kHz，分辨率为 15.6 Hz。

**分帧 + 加窗。** 我们不会对整个片段做 FFT。而是将其切分为重叠的*帧*（通常 25 ms，跳步 10 ms），每帧乘以窗函数（Hann、Hamming）以消除边缘不连续，然后对每个帧做 FFT。这就是短时傅里叶变换（STFT）。第 02 课将在此基础上继续。

## 动手实现

### 步骤 1：读取音频片段并绘制波形

`code/main.py` 仅使用标准库中的 `wave` 模块，以保持演示无额外依赖。生产环境中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤 2：从零开始合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz 正弦波（音乐会 A）在 16 kHz 下持续 1 秒，共 16,000 个浮点数。使用 `wave.open(..., "wb")` 以 16-bit PCM 编码写入。

### 步骤 3：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` —— 用于 `N=256` 验证正确性尚可，真实音频中完全派不上用场。实际代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：找出主导频率

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行，应在分桶 `440 * N / sr` 处返回峰值。

### 步骤 5：演示混叠

以 10 kHz 采样 7 kHz 正弦波（奈奎斯特 = 5 kHz）。7 kHz 音调高于奈奎斯特频率，折叠至 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都配备砖墙式低通滤波器的原因。

## 实际应用

2026 年你会实际部署的技术栈：

| 任务 | 库 | 原因 |
|------|-----|------|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 文件检查 | `ffprobe` 或 `soxi` | 命令行工具，快速报告采样率/声道/编码。 |

决策规则：**先匹配采样率，再考虑其他**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾输出。

## 交付

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否符合下游模型的预期，并在不匹配时正确重采样。

## 练习

1. **简单。** 合成 1 秒 220 Hz + 440 Hz + 880 Hz 的混合信号，采样率 16 kHz。运行 DFT，确认三个峰值出现在预期的分桶。
2. **中等。** 以 48 kHz 录制 3 秒你自己的语音 WAV。先使用 `torchaudio.transforms.Resample`（带抗混叠）降采样到 16 kHz，再用朴素抽取（每三个样本取一个）降采样到 16 kHz。对两者分别做 FFT。混叠出现在哪里？
3. **困难。** 仅使用 `math` 和步骤 3 中的 DFT 从零实现 STFT。帧长 400，跳步 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度图。这就是第 02 课的频谱图。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| 采样率 | 每秒多少样本 | ADC 测量信号的频率，单位为 Hz。 |
| 奈奎斯特 | 你能表示的最大频率 | `sr/2`；高于它的能量会混叠回低频。 |
| 位深度 | 每个样本的分辨率 | `int16` = 65,536 级；`float32` = `[-1, 1]` 中的 24-bit 精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，要求 `N` = 2 的幂。 |
| 分桶（Bin） | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层实现 | 随时间变化的加窗分帧 FFT。 |
| 混叠 | 奇怪的频率幽灵 | 高于奈奎斯特的能量镜像到较低分桶。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— 采样定理的原始论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) —— 免费、权威的 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) —— 带代码的实用入门。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) —— 参考：为什么真实世界的音频不是干净的正弦波。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) —— 10 分钟理清频率分桶直觉。
