# 构建完整的视觉流水线 — 综合项目

> 生产级视觉系统是由数据契约串联起来的模型与规则链。本阶段已经涵盖了所有组成部分；本综合项目将它们端到端地连接起来。

**类型：** 构建
**语言：** Python
**前置条件：** 第4阶段第01-15课
**时间：** ~120分钟

## 学习目标

- 设计一个生产级视觉流水线，实现目标检测、分类，并输出结构化 JSON —— 同时处理所有失败路径
- 将检测器（Mask R-CNN 或 YOLO）、分类器（ConvNeXt-Tiny）和数据契约（Pydantic）整合到一个服务中
- 对端到端流水线进行基准测试，识别首个瓶颈（通常是预处理，然后是检测器）
- 交付一个最小化的 FastAPI 服务，接收图像上传，运行流水线，返回带分类的检测结果

## 问题背景

单独的视觉模型有其用途；视觉产品则是由它们组成的链条。零售货架审计是检测器加产品分类器加价格 OCR 流水线。自动驾驶是 2D 检测器加 3D 检测器加分割器加跟踪器加规划器。医学预筛查是分割器加区域分类器加临床医生界面。

将这些链条连接起来，是将 ML 原型与产品区分开的关键。模型之间的每个接口都是新的 bug 滋生地。每个坐标变换、每个归一化、每个掩码调整大小都是潜在的静默失败点。流水线的强度取决于其最弱的接口。

本综合项目搭建最小可行流水线：检测 + 分类 + 结构化输出 + 服务层。第4阶段的其他内容都可以插入这个骨架：将 Mask R-CNN 替换为 YOLOv8，添加 OCR 头，添加分割分支，添加跟踪器。架构稳定；组件可插拔。

## 核心概念

### 流水线

```mermaid
flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detections + classes"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a
```

七个阶段。两个模型阶段计算昂贵；其余五个阶段是 bug 的藏身之处。

### 使用 Pydantic 的数据契约

每个模型边界都变成类型化对象。这将静默失败转化为显式失败。

```
Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)
```

当检测器以 `(cx, cy, w, h)` 而非 `(x1, y1, x2, y2)` 返回边界框时，Pydantic 的验证会在边界处失败，你能立即发现问题，而不是去调试一个静默返回空区域的下游裁剪操作。

### 延迟分布

几乎每个视觉流水线都遵循三条规律：

1. **预处理通常是最大的单一耗时块。** 解码 JPEG、转换色彩空间、调整大小 —— 这些是 CPU 密集型且容易被忽视的。
2. **检测器主导 GPU 时间。** 70-90% 的 GPU 时间花在检测前向传播上。
3. **后处理（NMS、RLE 编解码）在 GPU 上便宜，在 CPU 上昂贵。** 务必在实际目标设备上进行分析。

了解分布情况，才能将优化变成一份优先级清单。

### 失败模式

- **空检测** —— 返回空列表，不要崩溃。记录日志。
- **越界边界框** —— 裁剪前钳制到图像尺寸。
- **过小的裁剪区域** —— 跳过小于分类器最小输入尺寸的边界框的分类。
- **损坏的上传** —— 返回 400 响应并附带特定错误码，而非 500。
- **模型加载失败** —— 在服务启动时失败，而非在首个请求时。

生产级流水线处理上述每种情况时，都不会编写掩盖失败的通用 `try/except`。每种失败都获得命名代码和响应。

### 批处理

生产服务面向多个客户端。跨请求批量处理检测和分类可以成倍提升吞吐量。权衡：等待批次填满带来的额外延迟。典型配置：收集请求最多 20ms，合并批次，处理，分发响应。`torchserve` 和 `triton` 原生支持此功能；负载可预测的小型服务则自行实现微批处理器。

## 动手构建

### 步骤 1：数据契约

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

五秒钟的代码节省任何严肃流水线上数小时的调试时间。

### 步骤 2：最小化的 Pipeline 类

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

每个接口都有类型。每个失败路径都有特定的处理决策。

### 步骤 3：连接检测器和分类器

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# Use ImageNet-pretrained weights for a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### 步骤 4：FastAPI 服务

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialised on startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

使用 `uvicorn main:app --host 0.0.0.0 --port 8000` 运行。使用 `curl -F 'file=@dog.jpg' http://localhost:8000/detect` 测试。

### 步骤 5：流水线基准测试

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warm up

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

CPU 上的典型输出：预处理 ~3 ms，检测 300-500 ms，分类 20-40 ms，总计 350-550 ms。在 GPU 上，检测为 20-40 ms，预处理和分类在相对意义上开始变得更加重要。

## 实际应用

生产模板最终都会收敛到相同的结构，外加：

- **模型版本控制** —— 始终在响应中记录模型名称和权重哈希。
- **逐请求追踪 ID** —— 记录每个请求的每个阶段耗时，以便将慢响应与具体阶段关联。
- **降级路径** —— 如果分类器超时，返回不带分类的检测，而非让整个请求失败。
- **安全过滤器** —— NSFW / PII 过滤器在分类之后、响应离开服务之前运行。
- **批量端点** —— 接受图像 URL 列表进行批量处理的 `/detect_batch`。

对于生产级服务，`torchserve`、`Triton Inference Server` 和 `BentoML` 开箱即用地处理批处理、版本控制、指标和健康检查。直接运行 `FastAPI` 适用于原型和小规模产品。

## 交付成果

本课程产出：

- `outputs/prompt-vision-service-shape-reviewer.md` —— 一个审查视觉服务代码中契约/响应形状违规并指出首个破坏性 bug 的提示词。
- `outputs/skill-pipeline-budget-planner.md` —— 一项技能，给定目标延迟和吞吐量后，为每个流水线阶段分配时间预算，并标记哪个阶段将首先超出预算。

## 练习题

1. **（简单）** 在任意开放数据集的 10 张图像上运行流水线。报告每阶段的平均时间和每张图像的检测数量分布。
2. **（中等）** 向 `Detection` 添加掩码输出字段，并将其编码为 RLE。验证即使对于 10 个目标的图像，JSON 也能保持在 1MB 以下。
3. **（困难）** 在分类器前添加微批处理器：收集裁剪区域最多 10 ms，在一次 GPU 调用中分类所有区域，按请求返回结果。测量在每秒 5 个并发请求时的吞吐量提升和增加的延迟。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Pipeline（流水线） | "系统" | 预处理、推理和后处理步骤的有序链条，每对步骤之间有类型化接口 |
| Data contract（数据契约） | "模式" | Pydantic / dataclass 定义，每个阶段的输入和输出都必须符合；在边界处捕获集成 bug |
| Preprocessing（预处理） | "模型之前" | 解码、色彩转换、调整大小、归一化；通常是最大的 CPU 时间消耗 |
| Postprocessing（后处理） | "模型之后" | NMS、掩码调整大小、阈值、RLE 编码；GPU 上便宜，CPU 上昂贵 |
| Microbatcher（微批处理器） | "收集然后转发" | 等待固定窗口期聚合多个请求，运行单次批处理前向传播的聚合器 |
| Trace ID（追踪 ID） | "请求 id" | 在每个阶段记录的逐请求标识符，以便端到端追踪慢请求 |
| Failure code（失败代码） | "命名错误" | 每类失败对应特定错误码，而非通用 500；支持客户端重试逻辑 |
| Health check（健康检查） | "就绪探针" | 报告服务是否可响应的廉价端点；负载均衡器依赖此机制 |

## 延伸阅读

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) —— 生产级 ML 部署的经典概述
- [BentoML docs](https://docs.bentoml.com) —— 支持批处理、版本控制和指标的 serving 框架
- [torchserve docs](https://pytorch.org/serve/) —— PyTorch 官方 serving 库
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) —— 支持批处理和多模型的高吞吐量 serving
