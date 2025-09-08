# 使用 Lagrange.OneBot 部署 QQ 群机器人（HTTP 回调对接）

本文档说明如何用 Lagrange.OneBot 替换 go-cqhttp，并与本仓库的 `qq_group_bot.py` 通过 OneBot v11 HTTP 回调对接。无需修改现有 Python 代码。

## 架构概览
- Lagrange.OneBot：实现 OneBot v11 协议，登录 QQ 账号；
- 我方机器人：`qq_group_bot.py`（HTTP 服务器），接收 Lagrange 的事件回调，调用 SQL Agent 生成回复，并通过 OneBot HTTP API 发消息；
- 对接方式：OneBot v11 HTTP API + HTTP POST 回调（带 HMAC-SHA1 签名）。

## 前置要求
- QQ 小号（遵守平台规则，避免主号）；
- 已安装 Python，激活本项目虚拟环境 `.venv`；
- 预留端口：Lagrange HTTP API（如 5700）、机器人回调端口（如 8080）；
- 建议开启签名服务（签名服务器）与稳定协议（Android Pad/iPad），降低登录/发信风控（具体以 Lagrange 文档为准）。

## 安装 Lagrange.OneBot（概述）
1. 从其发行页面下载与你系统匹配的版本，解压到目录（例如 `~/apps/lagrange`）。
2. 首次启动将生成配置文件（具体名称/路径以该实现为准）。推荐扫码登录（不保存密码）。

> 注：不同版本/分支的配置文件格式略有差异，请以官方 README 为准。本文仅给出“字段含义与对接要点”，不强制具体键名。

## 关键配置项（对应关系）
- 账号与协议：
  - `uin`：你的 QQ 号；
  - `protocol`：选择稳定协议（如 Android Pad / iPad）；
- HTTP API（供我方机器人主动发消息）：
  - `http.enabled: true`
  - `http.host: 127.0.0.1`
  - `http.port: 5700`
  - `http.access_token`：可选，若设置则我方请求需带 `Authorization: Bearer <token>`；
- HTTP 回调（事件上报给我方机器人）：
  - `post.enabled: true`
  - `post.url: http://127.0.0.1:8080/`（我方机器人监听地址）
  - `post.secret: <强随机密钥>`（与我方 `GOCQHTTP_SECRET` 一致，用于 HMAC-SHA1 校验）
- 签名服务（强烈推荐）：
  - `sign-server` 或 `sign-servers`：填写你的签名服务地址与密钥（仅内网/本机暴露）。

> 若你的 Lagrange 版本仅提供 WebSocket（或推荐反向 WS），也可改为 WS 对接；本项目当前实现的是 HTTP 回调，如需 WS 方案请提 Issue/需求，我们可新增适配器。

## 启动我方机器人
在本项目根目录：

1) 激活虚拟环境
```
source .venv/bin/activate
```

2) 设置环境变量（示例）
```
export GOCQHTTP_API_BASE=http://127.0.0.1:5700
export GOCQHTTP_SECRET=your_post_secret      # 与 Lagrange 回调 secret 对应
export OPENAI_API_KEY=sk-...                 # 使用默认模型时需要
# 可选：仅响应某些群
export ALLOWED_GROUPS=123456789,234567890
# 可选：开启网络检索工具
export ENABLE_TOOLS=1
export TAVILY_API_KEY=tvly-...
# 可选：无 Postgres 时用内存检查点
export DRY_RUN=1
```

3) 启动机器人
```
python qq_group_bot.py
```

看到日志：
```
[QQBot] listening http://127.0.0.1:8080 api=http://127.0.0.1:5700
```
即表示回调服务就绪。

## 启动 Lagrange 并联调
1) 根据其文档启动 Lagrange，完成扫码/验证登录；
2) 确认其 HTTP API 与 HTTP 回调已启用（控制台应显示对应服务启动日志）；
3) 在目标 QQ 群发送一条消息（可先试“你好”）；
4) 我方机器人应收到事件回调并将最终回复发回群聊（控制台会打印生成过程的流式 token）。

## 常见问题与排错
- 警告“登录需要滑块/短信校验”：
  - 优先使用扫码登录；若走密码登录，请按 Lagrange 提示完成手动提票（ticket）且与 Lagrange 出口 IP 一致；
  - 时间同步、网络代理/VPN、浏览器脚本拦截等也会影响校验；
- 警告“未配置签名服务器/No usable sign server”：
  - 部署并启用签名服务（版本需与 Lagrange 匹配），按文档填写 `sign-server(s)`；
- 回调 401 invalid signature：
  - 确认 `post.secret` 与我方 `GOCQHTTP_SECRET` 完全一致；
- 能收到回调但发送失败：
  - 校验 `GOCQHTTP_API_BASE` 可达、端口正确；若配置了 access_token，需在我方请求中带上 Authorization；
- 我方不应答：
  - 确认 `.venv` 已激活、`OPENAI_API_KEY` 有效、`MODEL_NAME` 可用；
  - 无 Postgres 时设置 `DRY_RUN=1` 或配置 `LANGGRAPH_PG`。

## 可选：仅在 @机器人 或特定前缀时应答
当前实现对任意群文本应答。若需要“仅 @ 机器人”或“以 `!ai` 前缀触发”，可在 `qq_group_bot.py` 的 `do_POST` 中对 `raw_message/message` 与 CQ 码解析后做判断，我们可按需提供补丁。

