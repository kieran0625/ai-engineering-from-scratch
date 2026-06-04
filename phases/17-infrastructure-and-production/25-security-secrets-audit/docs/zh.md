# 安全——密钥、API 密钥轮换、审计日志与护栏

> 通过集中式密钥库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥管理混乱。切勿将凭据存储在配置文件、VCS 中的环境文件或电子表格中。优先使用 IAM 角色而非静态密钥；CI/CD 流程使用 OIDC。AI 网关模式是 2026 年的解决方案：应用 → 网关 → 模型提供商，网关在运行时从密钥库拉取凭据。在密钥库中轮换后，所有应用在几分钟内即可生效——无需重新部署，也无需在 Slack 上询问“谁有新密钥”。轮换策略 ≤90 天；每次提交时使用 TruffleHog / GitGuardian / Gitleaks 进行扫描。零信任：强制 MFA、SSO、RBAC/ABAC、短期令牌、设备安全状态检查。PII 清理使用实体识别技术在转发前对 PHI/PII 进行掩码处理；采用一致性标记化（Mesh 方法）将敏感值映射为稳定的占位符，以便 LLM 保留代码/关系语义。网络出口：LLM 服务位于专用 VPC/VNet 子网中，仅白名单放行 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站流量。2026 年事故主因：Vercel 供应链攻击，利用被窃取的 CI/CD 凭据从数千个客户部署中泄露了环境变量。

**类型：** 学习
**语言：** Python（标准库、简易 PII 清理器 + 审计日志写入器）
**前置条件：** 第 17 阶段 · 19（AI 网关）、第 17 阶段 · 13（可观测性）
**耗时：** 约 60 分钟

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码环境变量、电子表格、静态密钥），并指出其替代方案。
- 解释“AI 网关从密钥库拉取”模式为何是 2026 年的生产环境标准。
- 实现具有稳定性标记化的 PII 清理器（相同值映射为相同占位符），以保留语义关系。
- 说出 2026 年 Vercel 供应链事件及其关于 CI/CD 凭据管理的教训。

## 问题所在

一名实习生提交了包含 `.env` 和 API 密钥的代码。他们很快删除了该提交。但密钥已经存在于 git 历史中——GitGuardian 扫描发现了它，而你的轮换流程变成了“在 Slack 上通知团队、更新 40 个配置文件、重新部署所有服务。”8 小时后，一半服务已上线，另一半仍在等待部署窗口。

另外，用户提示词中包含“我的社保号是 123-45-6789。”提示词被发往 OpenAI。你虽然签署了 BAA，但内部政策要求在转发前对 PII 进行掩码处理。你没有执行。

此外，你的 EKS 集群中的 LLM Pod 可以访问任何互联网主机。有人通过向攻击者控制的域名发起 DNS 查询来外泄数据。没有任何机制拦截它。

LLM 服务的安全必须同时应对这三个向量。基于密钥库的凭据、PII 清理、网络出口过滤、审计日志。

## 核心概念

### 集中式密钥库 + IAM 角色拉取

**密钥库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实来源。

**IAM 角色**：应用/网关通过其 IAM 身份进行认证，而非使用静态密钥。密钥库在令牌有效期内返回该密钥。

**AI 网关模式**：网关在请求时从密钥库拉取 `OPENAI_API_KEY`。在密钥库中完成轮换后，下一次请求即获取新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

涵盖所有 API 密钥、密钥库根令牌、CI/CD 凭据。尽可能实现自动化轮换。手动轮换需记录并跟踪。

### 密钥扫描

- **TruffleHog** —— 基于正则表达式与熵值的提交扫描。
- **GitGuardian** —— 商业软件，高精度。
- **Gitleaks** —— 开源，可在 CI 中运行。

每次提交均运行扫描。若检测到新密钥，则阻断 PR。

### 零信任架构

- 所有账户强制启用 MFA。
- 通过 SAML/OIDC 实现单点登录（SSO）。
- 采用 RBAC（基于角色）或 ABAC（基于属性）实现细粒度访问控制。
- 使用短期令牌（小时级，而非天级）。
- 检查设备安全状态——仅限带有磁盘加密的企业设备。

### PII / PHI 清理

在提示词离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio 或商业方案）。
2. 掩码匹配到的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性标记化（Mesh 方法）：相同值始终映射到相同的占位符，以便 LLM 保留关系语义。
4. （可选）为 LLM 响应提供反向映射。

静态正则过滤器可捕获基础模式；NER 能捕获更多复杂情况。建议两者结合使用。

### 输入与输出护栏

输入侧：拦截已知越狱指令与禁止话题；按用户限流。

输出侧：使用正则表达式清理泄露的密钥（如拒绝场景中的 API 密钥模式、邮箱模式）；使用分类器检测策略违规。

### 网络出口白名单

LLM 服务部署在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、密钥库端点。
- 其他所有流量：丢弃。
- DNS 解析：仅允许通过白名单解析器（防止通过 DNS 隧道外泄）。

### 审计日志

每条 LLM 调用记录的不可变日志，包含：
- 时间戳。
- 用户/租户。
- 提示词哈希值（出于隐私保护，不存原始提示词）。
- 模型及版本。
- Token 数量。
- 成本。
- 响应哈希值。
- 触发的任何护栏规则。

保留期限遵循法规要求（SOC 2 为 1 年，HIPAA 为 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被攻破的 CI/CD 凭据导致数千个客户部署中的环境变量被外泄。教训：CI/CD 凭据等同于生产环境权限。必须存入密钥库。严格限制作用范围。积极频繁轮换。

### 关键数据记忆点

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- 2026 年 Vercel 事件：CI/CD 凭据泄露 → 数千个客户环境变量外泄。
- 审计日志保留期：SOC 2 = 1 年，HIPAA = 6 年。

## 实践应用

`code/main.py` 实现了一个简易的 PII 清理器，具备一致性标记化功能与仅追加模式的审计日志。

## 交付成果

本课程将产出 `outputs/skill-llm-security-plan.md`。结合监管范围与当前现状，规划密钥库迁移、清理器、出口策略及审计日志方案。

## 练习

1. 运行 `code/main.py`。发送两条引用相同社保号的提示词。确认两者均获得相同的占位符。
2. 为部署在 EKS 上的 vLLM 设计网络出口策略，该部署需调用 OpenAI、Anthropic 和 Weaviate。
3. 你在 git 历史中发现了一个密钥（已存在 2 年）。正确的应对措施是什么——轮换密钥、清理历史记录，还是两者都做？请说明理由。
4. 你的审计日志每天增长 10 GB。请设计保留层级（热数据 30 天、温数据 12 个月、冷数据 6 年）。
5. 论证是否值得引入反向标记化（将真实值替换回 LLM 响应中），与其带来的复杂度相比，直接保留占位符是否更优？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Vault | “密钥存储” | 集中式凭据管理服务 |
| IAM role | “基于身份的认证” | 应用扮演的角色；返回短期凭据 |
| OIDC for CI/CD | “云签发令牌” | CI 中不使用静态密钥——通过 OIDC 进行身份验证 |
| TruffleHog / GitGuardian / Gitleaks | “密钥扫描工具” | 提交时的密钥检测 |
| RBAC / ABAC | “访问控制” | 基于角色 vs 基于属性 |
| PII scrubbing | “数据掩码” | 移除或标记化敏感实体 |
| Consistent tokenization | “稳定占位符” | 相同值每次映射为相同的标记 |
| Mesh approach | “Mesh 标记化” | 保留语义的标记化模式 |
| Egress whitelist | “出站白名单” | 仅允许访问指定的域名 |
| Audit log | “不可变历史” | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) —— PII 检测与匿名化。
- [HashiCorp Vault 官方文档](https://developer.hashicorp.com/vault/docs)
