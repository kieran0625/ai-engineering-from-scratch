# AI 网关 —— LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于你的应用与模型提供商之间。核心功能包括：提供商路由、故障转移（fallback）、重试、速率限制、密钥引用、可观测性、护栏（guardrails）。2026 年市场格局：**LiteLLM** 是 MIT 协议的开源软件，支持 100+ 家提供商，兼容 OpenAI API，但在约 2000 RPS 时性能会崩溃（内存占用 8 GB，公开基准测试中会出现级联故障）；最适合 Python 应用、<500 RPS 的场景以及开发/原型验证。**Portkey** 定位为控制平面（提供护栏、PII 数据脱敏、越狱检测、审计追踪），于 2026 年 3 月转为 Apache 2.0 开源，每次请求延迟开销为 20-40 毫秒，生产环境套餐为每月 49 美元。**Kong AI Gateway** 基于 Kong Gateway 构建——在相同的 12 核 CPU 上，Kong 官方基准测试显示其比 Portkey 快 228%，比 LiteLLM 快 859%；定价为每月每模型 100 美元（Plus 套餐最多支持 5 个模型）；如果你已在使用 Kong，它非常适合企业级部署。**Bifrost**（Maxim AI）——支持自动重试与可配置退避策略，当 OpenAI 返回 429 错误时可降级至 Anthropic。**Cloudflare / Vercel AI 网关**——托管服务，零运维，提供基础重试功能。数据驻留要求是决定自托管的关键因素；Portkey 和 Kong 处于中间位置，提供“开源 + 可选托管”模式。

**类型：** 学习
**语言：** Python（标准库、简易网关路由模拟器）
**前置条件：** 阶段 17 · 01（托管 LLM 平台）、阶段 17 · 16（模型路由）
**耗时：** 约 60 分钟

## 学习目标

- 列举网关的六大核心功能（路由、故障转移、重试、速率限制、密钥引用、可观测性、护栏）。
- 将四种 2026 年的网关（LiteLLM、Portkey、Kong AI、Bifrost）与其性能上限和使用场景对应起来。
- 引用 Kong 的基准测试结果（比 Portkey 快 228%，比 LiteLLM 快 859%），并解释其对 >500 RPS 场景的意义。
- 根据数据驻留要求和运维预算，选择自托管还是托管方案。

## 问题背景

你的产品需要调用 OpenAI、Anthropic 以及一个自托管的 Llama 模型。每个提供商都有各自的 SDK、错误模型、速率限制和认证方案。你希望实现故障转移（如果 OpenAI 返回 429，则尝试 Anthropic）、统一的凭据存储、统一的可观测性，并按租户设置速率限制。

在应用层重复造轮子会使每个服务都与每个提供商强耦合。引入一层网关可以将这些功能整合到一个进程中，并提供一个统一的 API（通常为 OpenAI 兼容格式）来分发请求到各个提供商。

## 核心概念

### 六大核心功能

1. **提供商路由** —— 通过单一 API 接入 OpenAI、Anthropic、Gemini、自托管模型等。
2. **故障转移（Fallback）** —— 遇到 429、5xx 错误或质量不达标时，自动重试其他提供商。
3. **重试机制** —— 指数退避，限制最大重试次数。
4. **速率限制** —— 按租户、按密钥、按模型进行限制。
5. **密钥引用** —— 运行时从密钥库拉取凭据（绝不硬编码在应用中）。
6. **可观测性** —— OTel + GenAI 属性（阶段 17 · 13）+ 成本归因。
7. **护栏（Guardrails）** —— PII 数据脱敏、越狱检测、允许的话题过滤。

### LiteLLM —— MIT 协议开源，Python

- 支持 100+ 家提供商，兼容 OpenAI API，提供路由器配置、故障转移和基本可观测性。
- 在 Kong 的基准测试中，约 2000 RPS 时性能崩溃；内存占用 8 GB，在高负载下会出现级联故障。
- 最佳适用场景：Python 应用、<500 RPS、开发/预发环境网关、实验性路由。
- 成本：开源版免费；云版本提供免费套餐。

### Portkey —— 控制平面定位

- 自 2026 年 3 月起采用 Apache 2.0 开源协议。提供护栏、PII 脱敏、越狱检测、审计追踪。
- 每次请求延迟开销为 20-40 毫秒。
- 生产环境套餐每月 49 美元，包含数据保留与 SLA 保障。
- 最佳适用场景：需要内置护栏与可观测性的受监管行业。

### Kong AI Gateway —— 面向高并发扩展

- 基于 Kong Gateway（成熟的企业级 API 网关产品，使用 lua + OpenResty 构建）。
- Kong 官方在等效 12 核 CPU 上的基准测试显示：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：每月每模型 100 美元，Plus 套餐最多支持 5 个模型。
- 最佳适用场景：已在使用 Kong 生态；并发需求 >1000 RPS；愿意接受商业授权。

### Bifrost（Maxim AI）

- 支持自动重试与可配置退避策略。
- “OpenAI 返回 429 时降级至 Anthropic”是其典型的标准实践。
- 较新的市场参与者；商业软件。

### Cloudflare AI 网关 / Vercel AI 网关

- 托管服务，零运维。提供基础重试和可观测性。
- 最佳适用场景：部署在 Cloudflare/Vercel 边缘的 JavaScript 应用。
- 在护栏和速率限制方面，功能不如 Kong/Portkey 丰富。

### 自托管 vs 托管

数据驻留要求是决定性因素。医疗和金融领域默认选择自托管（LiteLLM 或 Portkey 开源版或 Kong）。消费级产品默认选择托管（Cloudflare AI 网关）或中间方案（Portkey 托管版）。混合架构：受监管租户使用自托管，其他租户使用托管。

### 延迟预算

- LiteLLM：典型延迟开销 5-15 毫秒。
- Portkey：延迟开销 20-40 毫秒。
- Kong：延迟开销 3-8 毫秒。
- Cloudflare/Vercel：边缘优势带来 1-3 毫秒延迟开销。

网关延迟会直接累加到首字生成时间（TTFT）中。若需满足 TTFT P99 < 100 毫秒的 SLA，应选择 Kong 或 Cloudflare。若 P99 < 500 毫秒，则任意网关均可满足。

### 速率限制语义很重要

简单的令牌桶算法适用于中等规模。多租户场景需要滑动窗口 + 突发允许 + 按租户分级。LiteLLM 内置令牌桶；Kong 内置滑动窗口；Portkey 内置分级限速。

### 网关、可观测性与路由的组合

阶段 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产环境中属于同一层级。可以选择一款覆盖全部三项功能的工具，或者谨慎地进行集成：大多数 2026 年的部署会将 Helicone（可观测性）或 Portkey（护栏）与 Kong（高并发扩展）组合使用，以实现职责分离。

### 关键数据速记

- LiteLLM：约 2000 RPS 时崩溃，内存占用 8 GB。
- Portkey：20-40 毫秒延迟开销；自 2026 年 3 月起采用 Apache 2.0 协议。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：每月每模型 100 美元，Plus 套餐最多 5 个模型。
- Cloudflare/Vercel：边缘部署延迟开销 1-3 毫秒。

## 动手实践

`code/main.py` 模拟了在注入 429/5xx 错误时，跨 3 家提供商进行网关路由与故障转移的过程。报告延迟、重试率和故障转移命中率。

## 交付成果

本课程将产出 `outputs/skill-gateway-picker.md`。根据并发规模、运维策略、合规要求及延迟预算，选择合适的网关。

## 练习

1. 运行 `code/main.py`。配置故障转移路径 OpenAI→Anthropic→自托管。在提供商错误率为 5% 的情况下，预期的命中率是多少？
2. 你的 SLA 要求基线为 300 毫秒时，TTFT P99 < 200 毫秒。哪些网关能满足此延迟预算？
3. 某医疗健康客户要求进行自托管 + PII 脱敏 + 审计追踪。请选择 Portkey 开源版或 Kong。
4. 对比 LiteLLM 与 Kong：团队应在多少 RPS 阈值时进行迁移？
5. 为多租户 SaaS 设计速率限制策略：免费版、试用版、付费版。应使用令牌桶还是滑动窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Gateway | “API 代理” | 位于应用与提供商之间的进程 |
| LiteLLM | “MIT 那个” | Python 开源，支持 100+ 提供商，2K RPS 时崩溃 |
| Portkey | “护栏网关” | 控制平面 + 可观测性，Apache 2.0 协议 |
| Kong AI Gateway | “高并发首选” | 基于 Kong Gateway 构建，基准测试领先 |
| Bifrost | “Maxim 的网关” | 重试机制 + Anthropic 降级方案 |
| Cloudflare AI Gateway | “边缘托管” | 边缘部署的托管网关，零运维 |
| PII redaction | “数据清洗” | 发送前使用正则表达式 + NER 进行掩码处理 |
| Jailbreak detection | “提示词注入防护” | 对用户输入进行分类器检测 |
| Audit trail | “合规日志” | 每条 LLM 调用的不可篡改记录 |
| Token-bucket | “简单限速” | 基于补充速率的限流器 |
| Sliding-window | “精确限速” | 基于时间窗口的限流器；公平性更好 |

## 延伸阅读

- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry —— 2026 年 AI 网关对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy —— 2026 年顶级 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
