# 实时视觉 —— 边缘部署

> 边缘推理是一门让 90% 准确率的模型在 2 GB RAM 设备上以 30 fps 运行的学问。每一个百分点的准确率都要与毫秒级的延迟做权衡。

**类型：** 学习 + 构建
**语言：** Python
**前置知识：** 第 4 阶段第 04 课（图像分类），第 10 阶段第 11 课（量化）
**时间：** ~75 分钟

## 学习目标

- 测量任意 PyTorch 模型的推理延迟、峰值内存和吞吐量，并理解 FLOPs / 参数量 / 延迟的权衡关系
- 使用 PyTorch 训练后量化将视觉模型量化为 INT8，并验证准确率损失 < 1%
- 导出为 ONNX 并使用 ONNX Runtime 或 TensorRT 编译；说出三种最常见的导出失败及其修复方法
- 解释在边缘约束下何时选择 MobileNetV3、EfficientNet-Lite、ConvNeXt-Tiny 或 MobileViT

## 问题所在

训练时的视觉模型是一个浮点数怪兽。1 亿参数，每次前向传播 10 GFLOPs，2 GB VRAM。这些都无法塞进手机、车载信息娱乐系统、工业相机或无人机。部署视觉系统意味着将同样的预测能力塞进小 100 倍的预算里。

三个旋钮承担了大部分工作：模型选择（更小的架构，相同的训练方法）、量化（INT8 替代 FP32）和推理运行时（ONNX Runtime、TensorRT、Core ML、TFLite）。把它们调好，决定了你的是只能在工位上跑的 demo，还是能部署在 30 美元相机模块上的产品。

本课首先建立测量规范（无法测量就无法优化），然后逐一讲解这三个旋钮。目标不是学会每一个边缘运行时，而是知道有哪些杠杆可用，以及如何验证每一个都如你所想。

## 核心概念

### 三项预算

```mermaid
flowchart LR
    M["Model"] --> LAT["Latency<br/>ms per image"]
    M --> MEM["Memory<br/>peak MB"]
    M --> PWR["Power<br/>mJ per inference"]

    LAT --> SHIP["Ship / no-ship<br/>decision"]
    MEM --> SHIP
    PWR --> SHIP

    style LAT fill:#fecaca,stroke:#dc2626
    style MEM fill:#fef3c7,stroke:#d97706
    style PWR fill:#dbeafe,stroke:#2563eb
```

- **延迟**：p50、p95、p99。只取平均 p50 会隐藏对实时系统至关重要的尾部行为。
- **峰值内存**：设备实际看到的最大值，而非稳态平均值。重要是因为在嵌入式目标上 OOM 是致命的。
- **功耗 / 能量**：电池供电设备上每次推理的毫焦数。通常用 CPU/GPU 利用率 × 时间来近似。

一张（模型、延迟、内存、准确率）表格就是边缘决策的依据。每个单元格都在目标设备上测量，而非工位机。

### 测量规范

每条边缘性能分析都应遵循的三条规则：

1. **预热**模型，测量前进行 5-10 次虚拟前向传播。冷缓存和 JIT 编译会产生不具代表性的首次数据。
2. 在计时块前后用 `torch.cuda.synchronize()` **同步** GPU 工作负载。不这样做的话，你测量的是内核调度而非内核执行。
3. **固定输入尺寸**为生产分辨率。224×224 的延迟不是 512×512 的延迟。

### FLOPs 作为代理指标

FLOPs（每次推理的浮点运算次数）是一个廉价、与设备无关的延迟代理指标。适用于架构比较，但作为绝对 wall-clock 时间则具有误导性。FLOPs 多 10% 的模型实际可能快 2 倍，因为它使用了硬件友好的算子（深度可分离卷积编译得好，大卷积核 7×7 则不然）。

规则：用 FLOPs 做架构搜索，用设备上延迟做部署决策。

### 量化一句话

将 FP32 的权重和激活替换为 INT8。模型大小降为 1/4，内存带宽降为 1/4，在有 INT8 内核的硬件上计算量降为 1/2 到 1/4（每台现代移动 SoC、每块带 Tensor Core 的 NVIDIA GPU 都支持）。视觉任务上训练后静态量化的准确率损失通常为 0.1-1 个百分点。

类型：

- **动态** —— 权重量化为 INT8，激活用 FP 计算。简单，加速有限。
- **静态（训练后）** —— 权重量化 + 在小规模校准集上校准激活范围。比动态快得多。
- **量化感知训练（QAT）** —— 训练期间模拟量化，让模型学会适应。准确率最高，需要标注数据。

对于视觉任务，训练后静态量化能带来 95% 的收益，却只花 5% 的力气。仅当 PTQ 的准确率损失不可接受时才用 QAT。

### 剪枝与蒸馏

- **剪枝** —— 移除不重要的权重（基于幅度）或通道（结构化）。在过参数化模型上效果好；对已经紧凑的架构用处不大。
- **蒸馏** —— 训练小模型（学生）模仿大模型（教师）的 logits。通常能恢复模型缩小后损失的大部分准确率。生产级边缘模型的标准做法。

### 推理运行时

- **PyTorch eager** —— 慢，不用于部署。仅开发时使用。
- **TorchScript** —— 遗留方案。已被 `torch.compile` 和 ONNX 导出取代。
- **ONNX Runtime** —— 通用运行时。CPU、CUDA、CoreML、TensorRT、OpenVINO 都有 ONNX 后端。从这里开始。
- **TensorRT** —— NVIDIA 的编译器。NVIDIA GPU（工位机和 Jetson）上延迟最低。可与 ONNX Runtime 集成或独立使用。
- **Core ML** —— Apple 的 iOS/macOS 运行时。需要 `.mlmodel` 或 `.mlpackage`。
- **TFLite** —— Google 的 Android/ARM 运行时。需要 `.tflite`。
- **OpenVINO** —— Intel 的 CPU/VPU 运行时。需要 `.xml` + `.bin`。

实践中：导出 PyTorch -> ONNX -> 为目标选择运行时。ONNX 是通用语言。

### 边缘架构选择器

| 预算 | 模型 | 理由 |
|--------|-------|-----|
| < 3M 参数 | MobileNetV3-Small | 到处都能编译，良好的基线 |
| 3-10M | EfficientNet-Lite-B0 | TFLite 上每参数准确率最高 |
| 10-20M | ConvNeXt-Tiny | 每参数准确率最高，CPU 友好 |
| 20-30M | MobileViT-S 或 EfficientViT | 带 ImageNet 准确率的 Transformer |
| 30-80M | Swin-V2-Tiny | 如果栈支持窗口注意力 |

除非有特定理由，否则全部量化为 INT8。

## 动手构建

### 步骤 1：正确测量延迟

```python
import time
import torch

def measure_latency(model, input_shape, device="cpu", warmup=10, iters=50):
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "mean_ms": sum(times) / len(times),
    }
```

预热、同步、使用 `time.perf_counter()`。报告百分位数，而非仅平均值。

### 步骤 2：参数量和 FLOP 统计

```python
def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def flops_estimate(model, input_shape):
    """
    Rough FLOP count for a conv/linear-only model. For production use `fvcore` or `ptflops`.
    """
    total = 0
    def conv_hook(m, inp, out):
        nonlocal total
        c_out, c_in, kh, kw = m.weight.shape
        h, w = out.shape[-2:]
        total += 2 * c_in * c_out * kh * kw * h * w
    def linear_hook(m, inp, out):
        nonlocal total
        total += 2 * m.in_features * m.out_features
    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_shape))
    for h in hooks:
        h.remove()
    return total
```

真实项目使用 `fvcore.nn.FlopCountAnalysis` 或 `ptflops`；它们能正确处理每种模块类型。

### 步骤 3：训练后静态量化

```python
def quantise_ptq(model, calibration_loader, backend="x86"):
    import torch.ao.quantization as tq
    model = model.eval().cpu()
    model.qconfig = tq.get_default_qconfig(backend)
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for x, _ in calibration_loader:
            model(x)
    tq.convert(model, inplace=True)
    return model
```

三步：配置、准备（插入观察器）、用真实数据校准、转换（融合 + 量化）。要求模型已融合（`Conv -> BN -> ReLU` -> `ConvBnReLU`），`torch.ao.quantization.fuse_modules` 会处理这个。

### 步骤 4：导出为 ONNX

```python
def export_onnx(model, sample_input, path="model.onnx"):
    model = model.eval()
    torch.onnx.export(
        model,
        sample_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return path
```

`opset_version=17` 是 2026 年的安全默认。`dynamic_axes` 让你能用任意 batch size 运行 ONNX 模型。

### 步骤 5：基准测试与对比方案

```python
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def compare_regimes():
    model = mobilenet_v3_small(weights=None, num_classes=10)
    params = parameter_count(model)
    flops = flops_estimate(model, (1, 3, 224, 224))
    lat_fp32 = measure_latency(model, (1, 3, 224, 224), device="cpu")
    print(f"FP32 MobileNetV3-Small: {params:,} params  {flops/1e9:.2f} GFLOPs  "
          f"p50={lat_fp32['p50_ms']:.2f}ms  p95={lat_fp32['p95_ms']:.2f}ms")
```

对 `resnet50`、`efficientnet_v2_s` 和 `convnext_tiny` 运行相同函数，你就得到了部署决策所需的对比表格。

## 实际应用

生产栈最终收敛为三条路径之一：

- **Web / 无服务器**：PyTorch -> ONNX -> ONNX Runtime（CPU 或 CUDA 后端）。最简单，对大多数场景足够好。
- **NVIDIA 边缘（Jetson、GPU 服务器）**：PyTorch -> ONNX -> TensorRT。延迟最低，工程投入最大。
- **移动端**：PyTorch -> ONNX -> Core ML（iOS）或 TFLite（Android）。导出前量化。

测量方面，`torch-tb-profiler`、`nvprof` / `nsys` 以及 macOS 上的 Instruments 提供逐层分解。`benchmark_app`（OpenVINO）和 `trtexec`（TensorRT）提供独立 CLI 数值。

## 交付成果

本课产出：

- `outputs/prompt-edge-deployment-planner.md` —— 一个根据目标设备和延迟 SLA 选择骨干网络、量化策略和运行时的提示词。
- `outputs/skill-latency-profiler.md` —— 一个编写完整延迟基准测试脚本的能力，包含预热、同步、百分位数和内存追踪。

## 练习

1. **（简单）** 在 CPU 上测量 `resnet18`、`mobilenet_v3_small`、`efficientnet_v2_s` 和 `convnext_tiny` 在 224×224 下的 p50 延迟。报告表格并找出每毫秒准确率最高的架构。
2. **（中等）** 对 `mobilenet_v3_small` 应用训练后静态量化。报告 FP32 与 INT8 的延迟以及 CIFAR-10 或类似数据集留出子集上的准确率损失。
3. **（困难）** 将 `convnext_tiny` 导出为 ONNX，通过 `onnxruntime` 用 `CPUExecutionProvider` 运行，并与 PyTorch eager 基线对比延迟。找出 ONNX Runtime 更快的第一个层并解释原因。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Latency | "多快" | 从输入到输出的时间；p50/p95/p99 百分位数，不是平均值 |
| FLOPs | "模型大小" | 每次前向传播的浮点运算次数；计算成本的粗略代理 |
| INT8 quantisation | "8-bit" | 将 FP32 权重/激活替换为 8 位整数；~4 倍更小，2-4 倍更快 |
| PTQ | "训练后量化" | 不重新训练就量化已训练模型；简单，通常够用 |
| QAT | "量化感知训练" | 训练期间模拟量化；准确率最高，需要标注数据 |
| ONNX | "通用格式" | 每个主流推理运行时都支持的模型交换格式 |
| TensorRT | "NVIDIA 编译器" | 将 ONNX 编译为 NVIDIA GPU 优化引擎 |
| Distillation | "教师 -> 学生" | 训练小模型模仿大模型的 logits；恢复大部分损失的准确率 |

## 延伸阅读

- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946) —— 高效架构的复合缩放
- [MobileNetV3 (Howard et al., 2019)](https://arxiv.org/abs/1905.02244) —— 带 h-swish 和 squeeze-excite 的移动优先架构
- [A Practical Guide to TensorRT Optimization (NVIDIA)](https://developer.nvidia.com/blog/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users/) —— 如何实际获得论文中的吞吐量数字
- [ONNX Runtime docs](https://onnxruntime.ai/docs/) —— 量化、图优化、后端选择
