# MCP Apps — 通过 `ui://` 提供交互式 UI 资源

> 纯文本工具输出限制了代理（Agent）的展示能力。MCP Apps（SEP-1724，官方发布于 2026 年 1 月 26 日）允许工具返回在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中以内联方式渲染的沙箱化交互式 HTML。仪表盘、表单、地图、3D 场景，全部通过一个扩展实现。本课程将讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME 类型、iframe 沙箱 postMessage 协议，以及让服务器渲染 HTML 所带来的安全面。

**类型：** 构建
**语言：** Python（标准库，UI 资源发射器）、HTML（示例应用）
**前置条件：** 第 13 阶段 · 07（MCP 服务器）、第 13 阶段 · 10（资源）
**预计时间：** 约 75 分钟

## 学习目标

- 从工具调用中返回 `ui://` 资源，并设置正确的 MIME 类型和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具的关联 UI。
- 实现用于 UI 与宿主通信的 iframe 沙箱 postMessage JSON-RPC。
- 应用防御 UI 来源攻击的 CSP 和 permissions-policy 默认配置。

## 问题背景

一款 2025 年时代的 `visualize_timeline` 工具可能会返回“以下是按时间顺序组织的 14 条笔记：……”。这只是一段文本。用户真正想要的是交互式的时间轴。在 MCP Apps 出现之前，选项只有：客户端特定的组件 API（Claude artifacts、OpenAI Custom GPT HTML），或者完全没有 UI。

MCP Apps（SEP-1724，于 2026 年 1 月 26 日发布）标准化了这一契约。工具结果包含一个 `resource`，其 URI 为 `ui://...`，MIME 类型为 `text/html;profile=mcp-app`。宿主会在带有受限 CSP 且除非明确授权否则无网络访问权限的沙箱 iframe 中渲染它。iframe 内的 UI 通过一种轻量级的 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容的客户端（Claude Desktop、ChatGPT、Goose、VS Code）都会以相同的方式渲染相同的 `ui://` 资源。一个服务器，一套 HTML 包，通用的 UI。

## 核心概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

随后宿主对 `ui://notes/timeline` URI 调用 `resources/read`，并获取到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主在沙箱化的 `<iframe>` 内渲染 HTML，并配置如下：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明更严格的限制）
- 通过响应头应用服务器声明的 CSP。
- 不共享宿主页面的 cookies 和 localStorage。
- 网络访问仅限于 CSP 中的 `connectSrc`。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。这是一种轻量级的 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定为对等方的确切源地址，并在接收端在处理任何负载前，对照白名单验证 `event.origin`。切勿在此通道的任一端使用 `"*"` —— 该通道主体承载工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可调用的可用宿主侧方法：

- `host.callTool(name, arguments)` —— 调用服务器工具。
- `host.readResource(uri)` —— 读取 MCP 资源。
- `host.getPrompt(name, arguments)` —— 获取提示词模板。
- `host.close()` —— 关闭 UI。

每次调用仍会经过 MCP 协议，并继承服务器的权限控制。

### 权限

`_meta.ui.permissions` 列表请求额外功能：

- `camera` —— 访问用户摄像头（用于扫描文档类 UI）。
- `microphone` —— 语音输入。
- `geolocation` —— 地理位置。
- `network:*` —— 比仅靠 `connectSrc` 允许的网络访问范围更广。

每项权限都会在 UI 渲染前作为提示显示给用户。

### 安全风险

iframe 中的 HTML 依然是 HTML。新的攻击面包括：

- **UI 注入提示词。** 恶意服务器 UI 可以显示看起来像系统消息的文本，从而欺骗用户。宿主渲染时应明显区分服务器 UI 与宿主 UI。
- **通过 `connectSrc` 进行数据外泄。** 如果 CSP 允许 `connect-src: *`，UI 可将数据发送到任意位置。默认配置应严格限制。
- **点击劫持。** UI 会覆盖宿主界面元素。宿主必须防止 z-index 操纵并强制执行透明度规则。
- **窃取焦点。** UI 获取键盘焦点并捕获下一条消息。宿主必须进行拦截。

第 13 阶段 · 15 将在 MCP 安全部分深入探讨这些问题；本课程仅作引入。

### `ui/initialize` 握手

iframe 加载完成后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主回复功能列表和会话令牌。UI 在后续所有宿主调用中使用该会话令牌。

### AppRenderer / AppFrame SDK 基础组件

ext-apps SDK 提供了两个便捷的基础组件：

- `AppRenderer`（服务端）—— 包装 React / Vue / Solid 组件，并生成带有正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）—— 接收资源，挂载 iframe，并中介处理 postMessage。

你可以直接使用这些组件，也可以手动编写 HTML 和 JSON-RPC。

### 生态现状

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持情况：

- **Claude Desktop。** 自 2026 年 1 月起全面支持。
- **ChatGPT。** 通过 Apps SDK 全面支持（底层使用相同的 MCP Apps 协议）。
- **Cursor。** Beta 版；需通过设置启用。
- **VS Code。** 仅限 Insider 版本。
- **Goose。** 全面支持。
- **Zed, Windsurf。** 已列入路线图。

生产环境中的服务器应用：仪表盘、地图可视化、数据表格、图表构建器、沙箱 IDE 预览。

## 实践使用

`code/main.py` 扩展了笔记服务器，增加了一个 `visualize_timeline` 工具，该工具返回 `ui://notes/timeline` 资源，并附带一个针对该 URI 上 `resources/read` 的处理程序，返回一个虽小但完整的 HTML 包，内含 SVG 时间轴。HTML 采用标准库模板化生成 —— 无需构建系统。由于标准库无法驱动浏览器，postMessage 的实现仅以 JS 注释形式示意。

重点关注：

- 工具响应中的 `_meta.ui` 携带 resourceUri、CSP 和 permissions。
- HTML 在无网络访问的情况下渲染；所有数据均已内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在本标准库演示中已记录但处于非活动状态）。

## 交付部署

本课程产出 `outputs/skill-mcp-apps-spec.md`。对于适合添加交互式 UI 的工具，该技能将生成完整的 MCP Apps 契约：`ui://` URI、CSP、permissions、postMessage 入口点以及安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查生成的 HTML。直接在浏览器中打开该 HTML；验证 SVG 是否正常渲染。然后草拟 UI 调用 `host.callTool("notes_update", ...)` 所需的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码需要做哪些调整？

3. 添加第二个 UI 资源 `ui://notes/editor`，包含一个用于就地编辑笔记的表单。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可能在哪里注入内容？iframe 沙箱能防御什么，又无法防御什么？

5. 阅读 SEP-1724 规范，找出 MCP Apps SDK 中本玩具实现未使用的一项功能。（提示：组件级状态同步。）

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|------------------------|
| MCP Apps | “交互式 UI 资源” | 2026-01-26 发布的 SEP-1724 扩展 |
| `ui://` | “App URI 方案” | 用于 UI 包的资源方案 |
| `text/html;profile=mcp-app` | “该 MIME 类型” | MCP App HTML 的 Content-Type |
| Iframe sandbox | “渲染容器” | 结合 CSP 和权限对 UI 进行的浏览器沙箱隔离 |
| postMessage JSON-RPC | “UI 到宿主的通信链路” | 基于 postMessage 的轻量级 JSON-RPC 方言，用于调用宿主功能 |
| `_meta.ui` | “工具与 UI 绑定” | 将工具结果链接到 UI 资源的元数据 |
| CSP | “内容安全策略” | 声明脚本、网络、样式等允许的来源 |
| AppRenderer | “服务端 SDK 基础组件” | 将框架组件转换为 `ui://` 资源的工具 |
| AppFrame | “客户端 SDK 基础组件” | 负责挂载 iframe 并中介 postMessage 的辅助函数 |
| `ui/initialize` | “握手” | UI 发送给宿主的首条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) —— 参考实现与 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) —— 正式规范文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) —— 高层级文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) —— 2026 年 1 月发布博文
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) —— JSDoc 风格的 SDK 参考
