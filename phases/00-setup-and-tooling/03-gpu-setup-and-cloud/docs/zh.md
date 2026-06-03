# GPU 配置与云服务

> 使用 CPU 进行训练对学习来说已经足够。但真正的训练需要 GPU。

**类型：** Build
**语言：** Python
**前置要求：** Phase 0, Lesson 01
**时间：** ~45 分钟

## 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 可用性
- 配置 Google Colab 的 T4 GPU 进行免费云端实验
- 在 CPU 与 GPU 上执行矩阵乘法基准测试并测量加速比
- 使用 fp16 经验法则估算 VRAM 能容纳的最大模型

## 问题背景

第 1-3 阶段的大部分课程在 CPU 上运行良好。但一旦开始训练 CNN、transformer 或 LLM（第 4 阶段及以后），就需要 GPU 加速。一个在 CPU 上需要 8 小时的训练任务，在 GPU 上只需 10 分钟。

你有三种选择：本地 GPU、云端 GPU 或 Google Colab（免费）。

## 核心概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 动手实践

### 选项 1：本地 NVIDIA GPU

检查你是否拥有 NVIDIA GPU：

```bash
nvidia-smi
```

安装带 CUDA 支持的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 访问 [colab.research.google.com](https://colab.research.google.com)
2. 点击 运行时 > 更改运行时类型 > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

可直接将本课程的 notebook 上传到 Colab 使用。

### 选项 3：云端 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没问题。

大部分课程在 CPU 上都能运行。需要 GPU 的课程会明确说明，并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手实践：GPU 与 CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习题

1. 运行上述基准测试，比较 CPU 与 GPU 的运行时间
2. 如果你没有 GPU，在 Google Colab 上运行并进行比较
3. 查看你的 GPU 显存容量，估算能容纳的最大模型（经验法则：fp16 下每个参数占 2 字节）

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| CUDA | "GPU 编程" | NVIDIA 的并行计算平台，允许在 GPU 上运行代码 |
| VRAM | "GPU 显存" | GPU 上的视频内存，与系统内存分离。限制模型大小。 |
| fp16 | "半精度" | 16 位浮点数，占用 fp32 一半的内存，精度损失极小 |
| Tensor Core | "快速矩阵硬件" | GPU 中专用于矩阵乘法的特殊核心，比普通核心快 4-8 倍 |
