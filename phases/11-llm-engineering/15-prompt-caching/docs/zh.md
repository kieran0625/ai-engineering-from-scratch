# 提示词缓存与上下文缓存

> 你的系统提示词为 4,000 个 token。你的 RAG 上下文为 20,000 个 token。你每次请求都会同时发送它们。你也要为两者付费——每次都是。提示词缓存让提供商在服务器端保持该前缀处于“预热”状态，并在重复使用时按正常费率的 10% 计费。使用得当，可将推理成本降低 50–90%，首字延迟降低 40–85%。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 minutes

## The Problem

一个编码代理在对话的每一步都向 Claude 发送相同的 15,000 token 系统提示词。以每百万输入 token 3 美元计算，仅 20 步的输入成本就高达 0.90 美元——这还是在用户实际消息之前。如果每天处理 10,000 次这样的对话，这笔从未变化的文本费用就会达到每天 9,000 美元。

在不损害质量的前提下无法缩减提示词。你也无法避免发送它——模型在每一步都需要它。唯一的办法是停止为提供商已经见过的固定前缀支付全价。

这一方案就是提示词缓存。Anthropic 于 2024 年 8 月推出该功能（2025 年增加了支持 1 小时延长 TTL 的变体），OpenAI 同年晚些时候实现了自动化，Google 随 Gemini 1.5 推出了显式上下文缓存，目前这三家都已将其作为前沿模型的一等公民功能提供。

## The Concept

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**The mechanic.** 当请求的前缀与近期某次请求匹配时，提供商会直接复用上次运行的 KV-cache，而不是重新对 token 进行编码。首次写入需支付少量溢价，之后每次读取都能享受大幅折扣。

**Three provider flavors in 2026.**

| Provider | API style | Hit discount | Write premium | Default TTL | Min cacheable |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上显式使用 `cache_control` 标记 | 输入费用减免 90% | 加收 25% | 5 分钟（可扩展至 1 小时） | 1,024 token（Sonnet/Opus）、2,048（Haiku） |
| OpenAI | 自动检测前缀 | 输入费用减免 50% | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google (Gemini) | 显式 `CachedContent` API | 按存储计费；读取费用约为正常的 25% | 按 token·小时收取存储费 | 用户自定义（默认 1 小时） | 4,096 token（Flash）、32,768（Pro） |

**The invariant.** 三者均只缓存前缀。如果请求间存在任何 token 差异，从第一个差异 token 开始之后的所有内容都会导致缓存未命中。将*stable*部分放在顶部，*variable*部分放在底部。

### The cache-friendly layout

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反此顺序——例如将用户消息放在系统提示词上方，或在少样本示例之间穿插动态检索结果——缓存将永远无法命中。

### The break-even calculation

Anthropic 的 25% 写入溢价意味着，一个缓存块至少需要被读取两次才能净省成本。1 次写入 + 1 次读取平均每次请求成本为 0.675 倍（节省 32%）；1 次写入 + 10 次读取平均成本为 0.205 倍（节省 80%）。经验法则：预期在 TTL 内至少复用 3 次的任何内容都应加入缓存。

## Build It

### Step 1: Anthropic prompt caching with explicit markers

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记指示 Anthropic 将该块缓存 5 分钟。在此窗口期内复用可命中缓存；过期后再次复用则会导致缓存失效并重新写入。

**Response usage fields:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查这两个字段——如果 `cache_read_input_tokens` 在所有请求中始终为零，说明你的缓存键正在发生漂移。

### Step 2: one-hour extended TTL

对于长时间运行的批处理任务，默认的 5 分钟 TTL 会在任务间隔期间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是原来的 2 倍（比基准线高 50% 而非 25%），但对于任何复用该前缀超过 5 次的批处理任务，都能快速收回成本。

### Step 3: OpenAI automatic caching

OpenAI 无需你进行任何配置。任何长度超过 1,024 token 且与近期请求匹配的前缀都会自动获得 50% 的折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样适用缓存友好的布局规则。有两点会破坏 OpenAI 的缓存但不会影响 Anthropic：修改 `user` 字段（用作缓存键的一部分）以及调整工具顺序。

### Step 4: Gemini explicit context caching

Gemini 将缓存视为你需要创建并命名的头等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

只要缓存存在，Gemini 就会按 token·小时收取存储费，读取费用约为正常输入费用的 25%。当你需要在数天内跨多个会话反复使用同一个巨型提示词时，这是最合适的方案。

### Step 5: measuring hit rate in production

参见 `code/main.py`，其中包含一个模拟的三厂商计费器，用于跟踪写入/读取/未命中次数，并计算每千次请求的混合成本。部署应基于目标命中率进行门禁控制——大多数生产环境的 Anthropic 设置在预热后读取比例应超过 80%。

## Pitfalls that still ship in 2026

- **Dynamic timestamps at the top.** 在系统提示词顶部使用 `"Current time: 2026-04-22 15:30:02"`。会导致每次请求都缓存未命中。请将时间戳移至缓存断点下方。
- **Tool reordering.** 以稳定顺序序列化工具——部署间的字典顺序打乱会导致所有命中失效。
- **Free-text near-duplicates.** "You are helpful." vs "You are a helpful assistant." ——仅差一个字节就会导致完全未命中。
- **Too-small blocks.** Anthropic 强制执行 1,024 token 的下限（Haiku 为 2,048）。更小的块会被静默忽略而不进行缓存。
- **Blind cost dashboards.** 将“输入 token”拆分为已缓存与未缓存两类。否则流量下降会被误认为是缓存优化带来的收益。

## Use It

The 2026 caching stack:

| Situation | Pick |
|-----------|------|
| Agent with stable 10k+ system prompt, many turns | Anthropic `cache_control` with 5-min TTL |
| Batch job reusing a prefix for 30+ minutes | Anthropic with `ttl: "1h"` |
| Serverless endpoints on GPT-5, no custom infra | OpenAI automatic (just make your prefix stable and long) |
| Multi-day reuse of a giant code/doc corpus | Gemini explicit `CachedContent` |
| Cross-provider fallback | Keep the cacheable prefix layout identical across providers so any hit works |

结合语义缓存（Phase 11 · 11）用于用户消息层：提示词缓存处理*token-identical*复用，语义缓存处理*meaning-identical*复用。

## Ship It

Save `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Exercises

1. **Easy.** 针对 Claude 发起一段 10 轮对话，系统提示词为 5,000 token。分别在不启用 `cache_control` 和启用它的情况下运行，并报告各自的输入 token 账单。
2. **Medium.** 编写一个测试工具，给定提示词模板和请求日志，计算各厂商的预期命中率及美元节省额（Anthropic 5m, Anthropic 1h, OpenAI automatic, Gemini explicit）。
3. **Hard.** 构建一个布局优化器：给定提示词和一组标记为 `stable=True/False` 的字段列表，重写提示词，在不丢失信息的前提下将单个缓存断点放置在最符合缓存友好的位置。并在真实的 Anthropic 端点上验证。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Prompt caching | "Makes long prompts cheap" | 复用提供商侧的 KV-cache 以匹配前缀；重复输入 token 可享受 50-90% 折扣。 |
| `cache_control` | "The Anthropic marker" | 声明“此处之前所有内容均可缓存”的内容块属性；详见 `{"type": "ephemeral"}`。 |
| Cache write | "Paying the premium" | 首次填充缓存的请求；Anthropic 按约 1.25 倍输入费率计费，OpenAI 免费。 |
| Cache read | "The discount" | 后续匹配该前缀的请求；Anthropic 按 10% 计费，OpenAI 按 50% 计费，Gemini 按约 25% 计费。 |
| TTL | "How long it lives" | 缓存保持热状态的时间（秒）；Anthropic 默认 5 分钟（可扩展至 1 小时），OpenAI 尽力而为最长 1 小时，Gemini 由用户设定。 |
| Extended TTL | "1-hour Anthropic cache" | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价为 2 倍，但用于批处理复用非常划算。 |
| Prefix match | "Why my cache missed" | 仅当从开头到断点的每个 token 都字节完全相同时，缓存才会命中。 |
| Context caching (Gemini) | "The explicit one" | Google 的命名且按存储计费的缓存对象；最适合多日复用大型语料库的场景。 |

## Further Reading

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL 及盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配机制。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 附带延迟数据的原始发布博文。
- Phase 11 · 05 (Context Engineering) —— 如何切分提示词以便缓存能够生效。
- Phase 11 · 11 (Caching and Cost) —— 将提示词缓存与用户消息层的语义缓存结合使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) —— 提示词缓存向用户暴露的 KV-cache 内存模型；解释了为何读取缓存前缀的成本比重新计算低约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) —— prefill 正是提示词缓存所优化的阶段；本文解释了为何缓存命中时 TTFT 会大幅下降，而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) —— 提示词缓存与 speculative decoding、Flash Attention 以及 MQA/GQA 并列，是弯曲推理成本曲线的四大杠杆之一；阅读此文可了解其余三项。
