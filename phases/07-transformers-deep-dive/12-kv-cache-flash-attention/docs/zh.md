# KV 缓存、Flash Attention 与推理优化

> 训练是并行的、受 FLOP 限制的。推理是串行的、受内存限制的。瓶颈不同，技巧不同。

**类型：** Build
**语言：** Python
**前置知识：** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**时间：** ~75 分钟

## 问题所在

一个朴素的自回归解码器生成 `N` 个 token 需要做 `O(N²)` 的工作：每一步都重新计算整个前缀的 attention。对于 4K token 的回复，那就是 16M 次 attention 操作，其中大多数是冗余的。前缀 token 的每个隐藏状态一旦计算出来就是确定的——你只需要用新 token 的 query 去和之前所有 token 缓存下来的 key、value 做 attention。

此外，attention 本身会移动大量数据。标准 attention 会物化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 的读写次数太多。当 N≥2K 时，attention 在成为 FLOP 瓶颈之前就已经成为内存瓶颈。经典的 attention kernel 对现代 GPU 的利用率不足 4–10 倍。

Dao 等人提出的两项优化，将前沿推理从"慢"推到了"快"：

1. **KV 缓存。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的 attention 就是一个 query 去和缓存的 key 做计算。推理从每步 `O(N²)` 降低到 `O(N)`。
2. **Flash Attention。** 对 attention 计算做分块（tiling），让完整的 N×N 矩阵永远不会进入 HBM。softmax + matmul 全部在 SRAM 中完成。A100 上 2–4 倍 wall-clock 加速；H100 上 FP8 可达 5–10 倍。

到 2026 年，两者已成为标配。每个生产级推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认启用。每个前沿模型都自带 Flash Attention。

## 核心概念

![KV 缓存增长与 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV 缓存的数学

每个解码器层、每个 token、每个 head：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

以 7B 模型为例，32 层、32 个 head、d_head=128、fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Llama 3 70B（80 层、d_head=128、GQA 共 8 个 KV head）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文、batch size 为 1 时，仅 KV 缓存就需要近 40 GB A100 的大部分显存。

**GQA 是 KV 缓存的胜利。** 如果用 64 个 head 的 MHA，会是 32 GB。MLA 还能进一步压缩。

### Flash Attention —— 分块技巧

标准 attention：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM 往返。在 H100 上，HBM 带宽为 3 TB/s；SRAM 为 30 TB/s。每次 HBM 往返相比全部在片内计算都是 10 倍减速。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 一次 HBM 往返。总内存占用从 `O(N²)` 降到 `O(N)`。反向传播时从正向重新计算部分值而非存储——又省了一笔内存。

**数值技巧。** 分块运行 softmax 时，跨 tile 维护 `(max, sum)`，最终归一化是精确的。这不是近似——Flash Attention 的输出与标准 attention 逐位相同（忽略 fp16 非结合性的微小差异）。

**版本演进：**

| 版本 | 年份 | 关键改进 | 参考硬件上的加速比 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 分块 SRAM kernel | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性、causal-first 排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步、FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线、software exp2 | 推理优先（最初仅正向） |

Flash 4 发布时仅支持正向传播。训练仍使用 Flash 3。Flash 4 的 GQA 和变长支持待定（2026 年中）。

### 投机解码 —— 另一个延迟优化

用廉价模型提出 N 个 token，大模型并行验证。如果验证通过 k 个 token，你就只花了 1 次大模型前向传播换来了 k 次生成。典型场景下 k=3–5，适用于代码和散文。

2026 年默认配置：
- **EAGLE 2 / Medusa。** 集成的 draft head，共享验证器的隐藏状态。无损 2–3 倍加速。
- **基于 draft model 的投机解码。** 消费级硬件上 2–4 倍加速。
- **Lookahead decoding。** Jacobi 迭代；无需 draft model。小众但免费。

### 连续批处理（Continuous batching）

经典批处理推理：等最慢的序列完成，再开新 batch。短回复提前完成时 GPU 空转浪费。

连续批处理（首次在 Orca 中实现，现已在 vLLM、TensorRT-LLM、SGLang 中普及）：旧请求一完成就立即把新请求换入 batch。典型聊天负载下吞吐量提升 5–10 倍。

### PagedAttention —— KV 缓存虚拟内存

vLLM 的招牌功能。KV 缓存按 16-token 块分配；页表将逻辑位置映射到物理块。支持并行采样时共享 KV（beam search、parallel sampling）、热切换前缀用于 prompt 缓存、以及内存碎片整理。相比朴素连续分配，吞吐量提升 4 倍。

## 动手实现

参见 `code/main.py`。我们将实现：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个带 `O(N)` KV 缓存的解码器。
3. 一个分块 softmax，模拟 Flash Attention 的 running-max 算法。

### 步骤 1：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每一层、每个 head 中，持续追加每个 token 的 K、V 向量。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与一次性计算的 `softmax(qK) V` 逐位相同，但任意时刻的工作集都是一个 `tile × d_head` 块，而非完整的 `N × d_head`。

### 步骤 3：对比朴素解码与缓存解码在 100 token 生成上的差异

统计 attention 操作数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会打印两者。

## 实际应用

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产环境：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的 prefix caching 是 2026 年的重大收益——相同的系统提示、few-shot 示例或长上下文文档，其 KV 可在多次调用间复用。对于重复工具提示的 agent 工作负载，prefix caching 通常带来 5 倍吞吐量提升。

## 实战部署

参见 `outputs/skill-inference-optimizer.md`。该技能要求为新推理部署选择 attention 实现、KV 缓存策略、量化和投机解码方案。

## 练习题

1. **简单。** 运行 `code/main.py`。确认朴素解码器和缓存解码器输出相同；注意操作数的差异。
2. **中等。** 实现 prefix caching：给定一个 prompt P 和多个 completion，先对 P 做一次前向传播填充 KV 缓存，然后每个 completion 分别分支。测量相比每次重新编码 P 的加速比。
3. **困难。** 实现一个简化版 PagedAttention：KV 缓存按固定 16-token 块分配，带空闲列表。序列完成时，将其块归还池中。模拟 1000 次长度各异的聊天 completion。对比与连续分配相比的内存碎片情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| KV cache | "让解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新 query 直接 attend 到它们，无需重新计算。 |
| HBM | "GPU 主存" | High Bandwidth Memory；H100 上 80 GB，B200 上 192 GB。带宽约 3 TB/s。 |
| SRAM | "片上内存" | 每个 SM 的高速内存，H100 上每个 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | "分块 attention kernel" | 计算 attention 时不将 N×N 矩阵物化到 HBM。 |
| Continuous batching | "无等待批处理" | 完成的序列换出、新序列换入，无需清空整个 batch。 |
| PagedAttention | "vLLM 的招牌" | KV 缓存按固定块分配，带页表；消除碎片。 |
| Prefix caching | "复用长 prompt" | 跨请求缓存共享前缀的 KV；对 agent 场景大幅降低成本。 |
| Speculative decoding | "Draft + verify" | 廉价 draft model 提出 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 级流水线与 software-exp2 技巧；阅读 repo README 了解本课提到的 forward-only 发布注意事项。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 论文，本课引用的集成 draft 方法。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 本课与 EAGLE 一并提及的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16-token 块和页表设计的权威深入解析。
