# 对话状态追踪

> "我想要一家北边便宜的餐厅……其实改成中等价位……再加个意大利菜。"三个回合，三次状态更新。DST 保持槽位-值字典同步，这样预订才能正常进行。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**时间：** ~75 分钟

## 问题

在任务导向型对话系统中，用户的目标被编码为一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每个用户回合都可能添加、修改或删除槽位。系统必须读取整个对话并正确输出当前状态。

一个槽位出错，系统就会预订错误的餐厅、安排错误的航班，或者扣错卡。DST 是用户所说的话与后端执行之间的关键枢纽。

为什么到 2026 年 LLM 时代它仍然重要：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽位值，而非自由文本生成。
- 工具使用型智能体在调用 API 前仍需要槽位解析。
- 多轮修正比看起来更难："其实不，改成周四。"

现代流程：经典 DST 概念 + LLM 提取器 + 结构化输出护栏。

## 概念

![DST: 对话历史 → 槽位-值状态](../assets/dst.svg)

**任务结构。** 模式定义了领域（餐厅、酒店、出租车）及其槽位（菜系、区域、价格、人数）。每个槽位可以是空的、填充来自封闭集合的值（价格：{便宜, 中等, 昂贵}），或是自由文本值（名称："The Copper Kettle"）。

**两种 DST 形式。**

- **分类。** 对每个（槽位, 候选值）对，预测是/否。适用于封闭词汇槽位。2020 年前的标准做法。
- **生成。** 给定对话，生成槽位值作为自由文本。适用于开放词汇槽位。现代默认方法。

**指标。** 联合目标准确率（JGA）—— 所有槽位都正确的回合比例。全对或全错。MultiWOZ 2.4 排行榜在 2026 年最高约 83%。

**架构。**

1. **基于规则（槽位正则 + 关键词）。** 窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** 基于 BERT 编码的复制生成。LLM 出现前的标准。
3. **LDST（LLaMA + LoRA）。** 指令微调 LLM，使用领域-槽位提示。在 MultiWOZ 2.4 上达到 ChatGPT 级别质量。
4. **无本体（2024–26）。** 跳过模式；直接生成槽位名称和值。处理开放领域。
5. **提示 + 结构化输出（2024–26）。** LLM + Pydantic 模式 + 约束解码。5 行代码，生产就绪。

### 经典失败模式

- **跨轮共指。** "我们就选第一个选项吧。" 需要解析指的是哪个选项。
- **覆盖 vs 追加。** 用户说"加意大利菜。"是替换菜系还是追加？
- **隐式确认。** "好的酷"——这是接受了提供的预订吗？
- **修正。** "其实改成 7 点。"必须更新时间而不清除其他槽位。
- **指向前序系统话语的共指。** "是的，那个。"哪个"那个"？

## 构建

### 步骤 1：基于规则的槽位提取器

见 `code/main.py`。正则 + 同义词词典覆盖窄领域中 70% 的规范表达：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

超出规范词汇则脆弱。适用于确定性槽位确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 绝不重置用户未触碰的槽位。
- 明确否定（"算了，不要菜系了"）必须清除。
- 用户修正（"其实……"）必须覆盖，而非追加。

### 步骤 3：带结构化输出的 LLM 驱动 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证有效的状态对象。无需正则，无模式不匹配，无幻觉槽位。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统有多少比例的回合能**全部**槽位正确？对于 MultiWOZ 2.4，2026 年顶尖系统：80-83%。你的领域内系统应在窄词汇上超过该值，否则 LLM 基线会击败你。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最近更新的槽位而非追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史中重新生成整个状态，而非增量更新——这自然能处理修正。

## 陷阱

- **全历史再生成本。** 让 LLM 每轮重新生成状态的总令牌成本为 O(n²)。限制历史长度或总结较早回合。
- **模式漂移。** 事后添加新槽位会破坏旧训练数据。给你的模式版本化。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN"——处处规范化。
- **隐式继承。** 如果用户之前指定了"4 个人"，新的不同时间请求不应清除人数。始终传递完整历史。
- **自由文本 vs 封闭集合。** 名称、时间和地址需要自由文本槽位；菜系和区域是封闭的。在模式中混合两者。

## 使用

2026 年技术栈：

| 场景 | 方法 |
|-----------|----------|
| 窄领域（一两个意图） | 基于规则 + 正则 |
| 宽领域，有标注数据可用 | LDST（LLaMA + LoRA 在 MultiWOZ 风格数据上） |
| 宽领域，无标注，生产就绪 | LLM + Instructor + Pydantic 模式 |
| 语音 / 口语 | ASR + 规范化器 + LLM-DST |
| 多领域预订流程 | 模式引导的 LLM，使用按领域的 Pydantic 模型 |
| 合规敏感 | 基于规则为主，LLM 兜底并带确认流程 |

## 交付

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中构建基于规则的状态追踪器，处理 3 个槽位（菜系、区域、价格）。在 10 段手工编写的对话上测试。测量 JGA。
2. **中等。** 相同数据集，使用 Instructor + Pydantic + 小型 LLM。比较 JGA。检查最难的回合。
3. **困难。** 实现两者并路由：基于规则为主，当基于规则的输出 <2 个槽位且置信度不足时 LLM 兜底。测量联合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | 跨对话回合维护槽位-值字典。 |
| Slot | 用户意图单元 | 后端需要的命名参数（菜系、日期）。 |
| Domain | 任务领域 | 餐厅、酒店、出租车——槽位的集合。 |
| JGA | Joint Goal Accuracy | 所有槽位都正确的回合比例。全对或全错。 |
| MultiWOZ | 基准测试 | 多领域 WOZ 数据集；标准 DST 评估。 |
| Ontology-free DST | 无模式 | 直接生成槽位名称和值，无固定列表。 |
| Correction | "其实……" | 覆盖先前已填充槽位的回合。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 经典基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 指令微调用于 DST。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 基于复制的 DST 主力。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 经典 DST 结果。
