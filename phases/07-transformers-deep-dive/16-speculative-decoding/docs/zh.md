# 投机解码 — 起草、验证、重复

> 自回归解码是串行的。每个 token 都要等待前一个生成完毕。投机解码打破这一链条：用一个轻量模型起草 N 个 token，再用大模型在一次前向传播中验证全部 N 个。当起草正确时，你只需一次大前向传播就能换得 N 次生成。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**时间：** ~60 分钟

## 问题背景

一个 70B 的 LLM 在 H100 上采样一个 token 约需 30 ms。一个 3B 的起草模型只需约 3 ms。如果我们让 3B 模型提前起草 5 个 token，然后让 70B 模型*只运行一次*来验证这 5 个，总耗时为 `5×3 + 30 = 45 ms`（最多可接受 5 个 token）—— 而直接生成则需要 `5×30 = 150 ms`。这就是投机解码的完整卖点：用少量额外的 GPU 内存（起草模型）换取 2–4 倍的解码延迟降低。

这个技巧必须保持分布不变性。Leviathan 等人（2023）以及 Chen 等人同时提出的投机采样方法，保证输出序列与**大模型独立生成**的结果**完全同分布**。没有质量损失，只是更快。

2026 年推理领域有四类主流的起草-验证配对：

1. **经典投机解码（Leviathan 2023）。** 独立的起草模型（如 Llama 3 1B）+ 验证模型（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 在验证模型上添加多个解码头，并行预测位置 `t+1..t+k`。无需独立的起草模型。
3. **EAGLE 系列（Li 2024, 2025）。** 轻量起草模型复用验证模型的隐藏状态；接受率高于经典方法；典型加速 3–4 倍。
4. **Lookahead 解码（Fu 2024）。** Jacobi 迭代；完全不需要起草模型。自我投机。应用场景较窄，但无任何依赖。

2026 年每个生产级推理栈都默认内置投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持经典 + EAGLE-2。

## 核心概念

### 核心算法

给定验证模型 `M_q` 和更便宜的起草模型 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **起草**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，附带起草概率 `p_1..p_N`。
3. **并行验证**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上对 `M_q` 运行一次，得到位置 `k+1..k+N+1` 的验证概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个起草 token**：对于每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次被拒绝时：从归一化的"残差"分布 `(q_j - p_j)_+` 中采样 `t_j`。`j` 之后的所有起草均被丢弃。
6. 全部 `N` 被接受时：从 `q_{N+1}` 中多采样一个 token `t_{N+1}`（免费的奖励 token）。

残差分布技巧是数学上的关键洞见，它确保输出分布与 `M_q` 从头开始采样的结果完全一致。

### 什么决定了加速比

设 `α` = 每个起草 token 的期望接受率。设 `c` = 起草模型与验证模型的成本比。每步：

- 朴素生成每 token 需要 1 次大模型调用。
- 当 `α` 较高时，投机解码每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 只需 1 次大模型调用。

在 `α = 0.75` 和 `N = 5` 下的经验法则：大模型调用减少约 3 倍。起草成本便宜 5 倍。总 wall-clock 时间降低约 2.5 倍。

**α 取决于：**

- 起草模型对验证模型的近似程度。同一家族 / 相同训练数据能显著提升 α。
- 解码策略。贪心起草对贪心验证：α 高。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由创意写作接受较少。

### Medusa — 无需起草模型的起草

Medusa 用验证模型上的额外输出头替代起草模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logits。推理时从每个头采样得到候选序列，然后用树注意力方案一次前向验证，同时考虑所有候选延续。

优点：无需第二个模型。缺点：增加可训练参数；需要监督微调阶段（~1B token）；接受率略低于优质起草模型的经典投机解码。

### EAGLE — 通过复用隐藏状态获得更好的起草

EAGLE-1/2/3（Li 等，2024–2025）将起草模型设计为一个小型 transformer（通常 1 层），输入验证模型的最后一层隐藏状态。由于起草模型能看到验证模型的特征表示，其预测与验证模型的输出分布高度相关。接受率从经典方法的 ~0.6 提升到 0.85+。

EAGLE-3（2025）增加了候选延续的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机路径。

### KV 缓存的舞蹈

验证时将 `N` 个起草 token 一次性输入验证模型。这会将验证模型的 KV 缓存扩展 `N` 个条目。如果部分起草被拒绝，必须将缓存回滚到已接受前缀的长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）使用临时 KV 缓冲区处理。先写入，确认接受后再提交。概念上不复杂，但实现起来很繁琐。

## 动手实现

参见 `code/main.py`。我们实现核心投机采样算法（拒绝步骤 + 残差分布），使用：

- 一个"大模型"，即对手工编码分布做确定性 softmax（以便解析验证接受率的数学）。
- 一个"起草模型"，即对大模型的扰动。
- 一个接受/拒绝循环，产生与直接采样相同的边缘分布。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证模型对起草 token 的概率。`p_prob` 是起草模型的概率。Leviathan 定理指出：这个 Bernoulli 决策，加上拒绝时从残差分布采样，能精确保持验证模型的分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

将 `p` 从 `q` 中逐元素相减，负值截断为零，再归一化。任何拒绝时从此分布采样。

### 步骤 3：一次投机步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个被接受 → 一个奖励 → 一次验证前向产生六个 token。

### 步骤 4：测量接受率

在不同起草质量水平下运行 10,000 次投机步骤。绘制接受率与起草和验证分布之间 KL 散度的关系。应看到清晰的单调关系。

### 步骤 5：验证分布等价性

经验验证：投机循环产生的 token 直方图应与直接从验证模型采样产生的直方图匹配。这就是 Leviathan 定理的实际体现。卡方检验确认在采样误差范围内。

## 实际应用

生产环境：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 包装了投机解码，使用一个小型起草模型。

**选择起草策略：**

| 策略 | 适用场景 | 加速比 |
|----------|--------------|---------|
| 经典起草（1B/3B Llama 系列） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa 头 | 可以微调验证模型 | 2–3× |
| EAGLE-2 / 3 | 生产环境，追求最大速度 | 3–4× |
| Lookahead | 无起草模型，无训练，无额外参数 | 1.3–1.6× |

**何时不使用投机解码：**

- 单序列生成 1–5 个 token。开销占主导。
- 极度创意 / 高温采样（α 下降）。
- 内存受限部署（起草模型增加 VRAM）。

## 实战部署

参见 `outputs/skill-spec-decode-picker.md`。该技能要求为新推理负载选择投机解码策略（经典 / Medusa / EAGLE / lookahead）和调优参数（N、起草温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认在 50,000 个 token 上，投机 token 分布与验证模型直接采样分布匹配，卡方检验 p > 0.05。
2. **中等。** 绘制加速比（每大模型前向的 token 数）随 `N` 变化的函数，针对 `α = 0.5, 0.7, 0.85`。为每个 α 确定最优 `N`。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：取第 14 课的最终 GPT，添加 3 个额外的 LM 头预测位置 t+2, t+3, t+4。在 tinyshakespeare 上用联合多任务损失训练。与截断同一模型构建的经典起草比较接受率。
4. **困难。** 实现回滚：从 10 个 token 前缀的 KV 缓存开始，输入 5 个起草 token，模拟在位置 3 被拒绝。验证下一次迭代时缓存读取正确匹配"前缀 + 前 2 个被接受的起草"。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 起草模型 | "便宜的那个" | 提出候选 token 的较小模型；通常比验证模型便宜 10–50 倍。 |
| 验证模型 | "大的那个" | 我们要保持其分布的目标模型；每投机步骤运行一次。 |
| 接受率 (α) | "起草有多准" | 验证模型接受起草的每 token 概率。典型值 0.7–0.9。 |
| 残差分布 | "拒绝时的后备" | `(q - p)_+` 归一化；拒绝时从此采样以保持验证模型的分布。 |
| 奖励 token | "免费的那个" | 当全部 N 个起草被接受时，从验证模型的下一步分布多采样一个。 |
| Medusa | "无起草的投机" | 验证模型上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | "隐藏状态起草" | 以验证模型最后一层隐藏状态为条件的微型 transformer 起草模型。 |
| Lookahead 解码 | "Jacobi 迭代" | 使用不动点迭代的自我投机；无需起草模型。 |
| 树注意力 | "同时验证多个候选" | 分支验证，同时考虑多个起草延续。 |
| KV 回滚 | "撤销被拒绝的起草" | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法与等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期发表；简洁的 Bernoulli 拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；树注意力验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；隐藏状态条件起草。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，无起草方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 生产级标准参考，四种策略全部集成。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的参考代码。
