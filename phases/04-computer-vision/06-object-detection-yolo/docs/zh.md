# 目标检测 —— 从零实现 YOLO

> 检测就是分类加回归，在特征图的每个位置运行，然后用非极大值抑制进行清理。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 4 第 03 课（CNN）、阶段 4 第 04 课（图像分类）、阶段 4 第 05 课（迁移学习）
**时间：** ~75 分钟

## 学习目标

- 解释将检测转化为密集预测问题的网格-锚点设计，并说明输出张量中每个数字的含义
- 计算框之间的交并比（IoU），并从零实现非极大值抑制
- 在预训练骨干网络之上构建一个最简的 YOLO 风格检测头，包括分类、目标置信度和框回归损失
- 读取检测指标行（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95）并判断下一步该调整哪个参数

## 问题

分类说"这张图是狗"。检测说"像素位置 (112, 40, 280, 210) 有一只狗，(400, 180, 560, 310) 有一只猫，画面中没有其他东西"。这一个结构性变化——预测可变数量的带标签框，而不是每张图一个标签——正是每个自动驾驶系统、每个安防产品、每个文档布局解析器和每条工厂视觉产线所依赖的核心。

检测也是视觉领域中所有工程权衡同时涌现的地方。你需要框足够精确（回归头），需要每个框的类别正确（分类头），需要模型知道何时没有目标可检测（目标置信度分数），还需要每个真实目标只有一个预测（非极大值抑制）。遗漏任何一个，流水线就会漏检、产生幻觉框，或者把同一个目标在稍微不同的位置预测十五次。

YOLO（You Only Look Once，Redmon 等，2016）通过单次卷积网络前向传播实现了这一切的实时运行，而相同的结构决策至今仍是现代检测器（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）的骨干。掌握核心原理，每个变体都只是相同部件的重新排列。

## 概念

### 检测作为密集预测

分类器每张图像输出 C 个数字。YOLO 风格的检测器每张图像输出 `(S x S x (5 + C))` 个数字，其中 S 是空间网格大小。

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

`S * S` 个网格单元中的每一个都预测 `B` 个框。对于每个框：

- 4 个数字描述几何信息：`tx, ty, tw, th`。
- 1 个数字是目标置信度分数："这个单元格中心是否有目标？"
- C 个数字是类别概率。

每个单元格总计：`B * (5 + C)`。对于 VOC 数据集，`S=13, B=2, C=20`，即每个单元格 50 个数字。

### 为什么使用网格和锚点

简单回归会为每个目标预测 `(x, y, w, h)` 作为绝对坐标。这对卷积网络来说很困难，因为平移图像不应该让所有预测平移相同的量——每个目标都有空间锚定。网格通过将每个真实框分配给其中心落入的网格单元来解决这个问题；只有该单元格负责该目标。

锚点解决第二个问题。3×3 卷积无法轻易地从 16 像素感受野的特征单元回归出 500 像素宽的框。因此，我们为每个单元格预定义 `B` 个先验框形状（锚点），并预测每个锚点的小幅偏移。模型学习选择合适的锚点并进行微调，而不是从零开始回归。

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

现代检测器通常使用 FPN，在不同分辨率上使用不同的锚点集——浅层高分辨率图上的小锚点，深层低分辨率图上的大锚点。相同思路，更多尺度。

### 解码预测

原始的 `tx, ty, tw, th` 不是框坐标；它们是需要转换后才能绘制的回归目标：

```
centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)
```

`sigmoid` 将中心偏移限制在单元格内。`exp` 让宽度可以从锚点自由缩放而不出现符号翻转。`stride` 将网格坐标缩放回像素。这个解码步骤自 YOLO v2 以来每个版本都相同。

### IoU

检测中两个框之间的通用相似度度量：

```
IoU(A, B) = area(A intersect B) / area(A union B)
```

IoU = 1 表示完全相同；IoU = 0 表示无重叠。预测框与真实框之间的 IoU 决定了预测是否计为真正例（通常 IoU >= 0.5）。两个预测框之间的 IoU 是 NMS 用于去重的依据。

### 非极大值抑制

在相邻锚点上训练的卷积网络经常会对同一目标预测出重叠的框。NMS 保留置信度最高的预测，删除 IoU 超过阈值的其他预测。

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

典型阈值：目标检测中为 0.45。近期检测器用 `soft-NMS`、`DIoU-NMS` 替代标准 NMS，或直接学习抑制（RT-DETR），但结构目的相同。

### 损失

YOLO 损失是三个损失加权相加：

```
L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)
```

只有包含目标的单元格对框回归和分类损失有贡献。没有目标的单元格只对目标置信度损失有贡献（教会模型保持沉默）。`lambda_noobj` 通常较小（~0.5），因为绝大多数单元格是空的，否则会主导总损失。

现代变体将 MSE 框损失替换为 CIoU / DIoU（直接优化 IoU），使用 focal loss 处理类别不平衡，并用 quality focal loss 平衡目标置信度。三组件结构不变。

### 检测指标

准确率不适用于检测。四个有效的指标：

- **Precision@IoU=0.5** —— 在计为正例的预测中，有多少真正正确。
- **Recall@IoU=0.5** —— 在真实目标中，我们找到了多少。
- **AP@0.5** —— IoU 阈值 0.5 时的精确率-召回率曲线下面积；每个类别一个数字。
- **mAP@0.5:0.95** —— IoU 阈值 0.5, 0.55, ..., 0.95 上 AP 的平均值。COCO 指标；最严格、信息量最大。

四个都要报告。mAP@0.5 强但 mAP@0.5:0.95 弱的检测器定位大致正确但不紧密；用更好的框回归损失修复。精确率高但召回率低的检测器过于保守；降低置信度阈值或增加目标置信度权重。

## 动手实现

### 步骤 1：IoU

本课的核心工具。处理 `(x1, y1, x2, y2)` 格式的两组框数组。

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回成对 IoU 的 `(N_a, N_b)` 矩阵。对单个真实框使用时，让其中一个数组的形状为 `(1, 4)`。

### 步骤 2：非极大值抑制

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

确定性的，`O(N log N)` 来自排序，在相同输入下与 `torchvision.ops.nms` 的行为一致。

### 步骤 3：框编码与解码

在像素坐标与网络实际回归的 `(tx, ty, tw, th)` 目标之间转换。

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试：编码一个框然后解码——应该能恢复出与原始非常接近的结果（由于 sigmoid 逆函数在 `tx` 不在后 sigmoid 范围内时不完全可逆，会有微小差异）。

### 步骤 4：最简 YOLO 检测头

在特征图上做一个 1×1 卷积，重塑为 `(B, S, S, num_anchors, 5 + C)`。

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

输出形状：`(N, H, W, num_anchors, 5 + C)`。最后一维保存 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### 步骤 5：真实框分配

对于每个真实框，决定哪个 `(cell, anchor)` 负责。

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

锚点选择采用"与真实框形状最佳 IoU"——一个廉价的代理方法，与 YOLOv2/v3 的分配方式一致。v5 及以后使用更复杂的策略（任务对齐匹配、动态 k）来细化相同思路。

### 步骤 6：三个损失

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

五个超参数，每个 YOLO 教程要么硬编码要么扫描。比例很重要：`lambda_coord=5, lambda_noobj=0.5` 参考原始 YOLOv1 论文，仍可作为合理的默认值。

### 步骤 7：推理流水线

解码原始检测头输出，应用 sigmoid/exp，按目标置信度阈值筛选，然后 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这就是完整的评估路径：检测头 -> 解码 -> 阈值筛选 -> NMS。

## 使用它

`torchvision.models.detection` 以相同的概念结构交付生产检测器。加载预训练模型只需三行。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对于实时推理流水线，`ultralytics`（YOLOv8/v9）是标准选择：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。模型内部处理解码和 NMS，返回与你上面构建的相同的 `boxes / scores / labels` 三元组。

## 交付

本课产出：

- `outputs/prompt-detection-metric-reader.md` —— 一个提示词，将 `precision, recall, AP, mAP@0.5:0.95` 行转化为一行诊断和最有用的下一个实验。
- `outputs/skill-anchor-designer.md` —— 一项技能，给定真实框数据集，对 `(w, h)` 运行 k-means，返回每个 FPN 级别的锚点集以及选择合适锚点数量所需的覆盖统计信息。

## 练习

1. **（简单）** 实现 `box_iou`，并在 1,000 对随机框上与 `torchvision.ops.box_iou` 对比运行。验证最大绝对差低于 `1e-6`。
2. **（中等）** 将 `yolo_loss` 移植为使用 `CIoU` 框损失的版本，替代 MSE。在 100 张图像的合成数据集上展示，在相同 epoch 数下 CIoU 比 MSE 收敛到更好的最终 mAP@0.5:0.95。
3. **（困难）** 实现多尺度推理：将同一张图像以三种分辨率输入模型，合并框预测，最后运行单次 NMS。在留出集上测量相对于单尺度推理的 mAP 提升。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Anchor | "框先验" | 每个网格单元上预定义的框形状，网络从中预测偏移量而非绝对坐标 |
| IoU | "重叠" | 两个框的交并比；检测中的通用相似度度量 |
| NMS | "去重" | 贪心算法，保留最高分数的预测，删除超过阈值的重叠预测 |
| Objectness | "这里有东西吗" | 每个锚点、每个单元格上的标量，预测该单元格中心是否有目标 |
| Grid stride | "下采样因子" | 每个网格单元的像素数；416 像素输入配 13 网格检测头时 stride 为 32 |
| mAP | "平均精度均值" | 精确率-召回率曲线下面积的平均值，按类别平均（COCO 中还按 IoU 阈值平均） |
| AP@0.5 | "PASCAL VOC AP" | IoU 阈值 0.5 时的平均精度；该指标的宽松版本 |
| mAP@0.5:0.95 | "COCO AP" | IoU 阈值 0.5..0.95 步长 0.05 的平均；严格版本，当前社区标准 |

## 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) —— 奠基论文；此后每个 YOLO 都是对该结构的细化
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) —— 引入多尺度 FPN 风格检测头的论文；仍有最清晰的图示
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com) —— 当前生产参考；涵盖数据集格式、增强、训练配方
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) —— 检测器全景的最佳通俗讲解；对理解 DETR、RetinaNet、FCOS 和 YOLO 的关系 invaluable
