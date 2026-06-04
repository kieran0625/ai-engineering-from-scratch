# 毕业设计 12 —— 视频理解流水线（场景、问答、搜索）

> Twelve Labs 将 Marengo 和 Pegasus 产品化。VideoDB 推出了面向视频的 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 权重。Gemini 长上下文原生支持数小时视频。TimeLens-100K 定义了大规模时间定位。2026 年的流水线已定型：场景分割、逐场景描述与嵌入、字幕对齐、多向量索引，以及能够返回（开始，结束）时间戳加帧预览的查询。本次毕业设计的核心是处理 100 小时视频数据，跑通公开基准测试，并测量计数与动作类问题的幻觉率。

**类型：** 毕业设计
**语言：** Python（流水线），TypeScript（UI）
**前置要求：** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**耗时：** 30 小时

## 问题背景

在 2026 年的规模下，长视频问答是最耗带宽的多模态任务。Gemini 2.5 Pro 可以原生解析 2 小时的视频，但要将 100 小时视频摄入可查询语料库，仍需基于场景级别的索引。生产环境架构结合以下模块：场景分割（TransNetV2 或 PySceneDetect）、使用 VLM 进行逐场景描述生成（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、字幕对齐（带词级时间戳的 Whisper-v3-turbo），以及并排存储描述、帧嵌入和字幕的多向量索引。查询流水线将返回（开始，结束）时间戳及帧预览作为答案。

基准测试采用公开数据集（ActivityNet-QA、NeXT-GQA）加上你自行构建的 100 条自定义查询集。计数与动作类问题的幻觉是已知的高难度失败类别；本毕业设计将对其进行明确测量。

## 核心概念

摄入阶段并行运行三条流水线。**场景分割**将视频切分为独立场景。**VLM 描述生成**为每个场景生成描述文本，并从关键帧提取帧嵌入。**ASR 对齐**生成词级时间戳。这三路数据流通过 `(scene_id, time_range)` 进行关联。每个场景在多向量索引（Qdrant）中对应三种向量类型：描述嵌入、关键帧嵌入、字幕嵌入。

查询时，自然语言问题同时向三个向量发起检索；结果通过 RRF（倒数排名融合）合并；随后由时间定位适配器（TimeLens 风格）在最优场景中细化 `(start, end)` 窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询内容、Top 场景及裁剪后的帧，最终输出带有引用时间戳和帧预览的答案。

幻觉测量至关重要。计数类（“有多少人进入房间？”）和动作类（“厨师是否在搅拌前倒入了液体？”）问题以难以准确回答著称。需将此类问题的准确率与描述类问题分开报告。

## 架构设计

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 年 SOTA）或 PySceneDetect
- 语音识别 (ASR)：faster-whisper 封装的 Whisper-v3-turbo（支持词级时间戳）
- VLM 描述生成与回答：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：基于 TimeLens-100K 训练的适配器或 VideoITG
- 索引：支持多向量的 Qdrant（描述 / 帧 / 字幕）
- UI：Next.js 15，集成 HTML5 视频播放器与场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 题人工标注集
- 幻觉基准：含人工标注的计数与动作类子集

## 实现步骤

1. **数据摄取器**。接受 YouTube URL 或本地 MP4 文件。必要时将分辨率降至 720p。持久化存储至 `{video_id, file_path}`。

2. **场景分割**。运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标处理 100 小时视频：约 6k-8k 个场景。

3. **ASR 处理**。对音频运行 Whisper-v3-turbo；导出词级时间戳；按场景拆分子句字幕切片。

4. **VLM 描述生成**。针对每个场景，传入关键帧与简短描述模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。输出描述文本与帧嵌入。

5. **多向量索引**。创建包含三个命名向量的 Qdrant 集合。Payload 结构：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询**。自然语言问题触发三次密集检索；使用倒数排名融合（RRF）合并结果；取 Top-k=5 的场景。

7. **时间定位**。在最优场景上运行 TimeLens 风格适配器，以细化该场景内的 `(start, end)` 窗口。

8. **VLM 合成**。传入查询内容、Top-3 场景片段（图像或短视频）及字幕调用 Gemini 2.5 Pro。强制要求输出 `(video_id, start_ms, end_ms)` 引用格式。

9. **评估**。运行 ActivityNet-QA 与 NeXT-GQA。构建 100 条自定义查询集。报告整体准确率及各分类细分指标（计数、动作、描述）。

## 使用方式

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付标准

`outputs/skill-video-qa.md` 为最终交付物。输入 YouTube URL 或上传视频后，流水线将对场景建立索引，并返回带有时间戳引用的问答结果。

| 权重 | 评估标准 | 测量方式 |
|:-:|---|---|
| 25 | 时间定位 IoU | 在预留测试集上的交并比（Intersection-over-Union） |
| 20 | 问答准确率 | NeXT-GQA 与自定义 100 条查询集 |
| 20 | 摄入吞吐量 | 每美元消耗处理的视频时长 |
| 20 | UI 与引用交互体验 | 时间戳链接、缩略图条、跳转至指定帧 |
| 15 | 幻觉率 | 分别统计计数与动作类问题的准确率 |
| **100** | | |

## 练习

1. 将描述生成阶段的 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在人工评分的 50 个场景样本上报告描述质量差异。

2. 将每场景的帧嵌入从多向量简化为单个池化向量。测量检索性能的下降幅度。

3. 开发“严格计数”模式：合成器提取每个被计数实例的时间戳，用户点击进行验证。测量用户验证是否能降低幻觉率。

4. 基准测试摄入成本：对比三种 VLM 选择的“每美元处理视频时长”。找出性价比最优解。

5. 添加说话人分离字幕：对音频运行 pyannote 说话人分离，并为每位说话人生成独立字幕嵌入。演示“Alice 关于 X 说了什么？”类查询。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Scene segmentation | “镜头检测” | 在镜头切换边界处将视频切割为独立场景 |
| Multi-vector index | “描述+帧+字幕” | 为每种表示形式分配命名向量的 Qdrant 集合 |
| Temporal grounding | “具体发生在何时” | 为查询答案细化 `(start, end)` 时间窗口 |
| Frame embedding | “视觉表征” | 关键帧的向量嵌入；用于场景视觉相似度匹配 |
| RRF fusion | “倒数排名融合” | 跨多个排序列表的合并策略；经典混合检索技巧 |
| Counting hallucination | “数错” | VLM 在“有多少个 X”类问题上的已知失效模式 |
| ActivityNet-QA | “视频问答基准” | 长视频问答准确率基准测试 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 权重
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时间定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 托管版参考实现
- [VideoDB](https://videodb.io) — 面向视频的 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业化参考方案
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准
