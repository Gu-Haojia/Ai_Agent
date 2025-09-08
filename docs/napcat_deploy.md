# 使用 NapCat.OneBot 部署与对接（HTTP 回调）

本文档说明如何用 NapCat.OneBot 对接本仓库 `qq_group_bot.py`（OneBot v11 HTTP 回调）。

## NapCat.OneBot 安装与启用
1. 按 NapcatQQ 官方说明安装 NapCat 与其 OneBot 插件（NapCat.OneBot）。
2. 运行后生成配置（不同版本路径与键名可能略有不同，下例为常见含义）：

```yaml
http:
  enable: true
  host: 127.0.0.1
  port: 5700
  access_token: ""        # 可选，配置后需要在请求头携带 Authorization: Bearer <token>

http_post:
  enable: true
  url: http://127.0.0.1:8080/    # 我方机器人回调地址
  secret: your_post_secret        # 建议设置，用于 HMAC-SHA1 验签

# （可选）签名服务器/协议相关：按 Napcat 文档配置
sign-server:
  enable: true
  url: http://127.0.0.1:8081
  key: use-strong-random-key
```

> 具体键名以你安装的 NapCat.OneBot 版本为准；若提供 `websocket` 反向连接等模式，请保持 HTTP 回调与 HTTP API 启用以配合当前 Python 机器人。

## 启动本项目机器人
在项目根目录：

```bash
source .venv/bin/activate

export ONEBOT_API_BASE=http://127.0.0.1:5700
export ONEBOT_SECRET=your_post_secret         # 必须与 http_post.secret 一致
# 若 Napcat 配置了 access_token：
# export ONEBOT_ACCESS_TOKEN=your_token

# OpenAI/Agent 相关
export OPENAI_API_KEY=sk-...
# export ENABLE_TOOLS=1
# export TAVILY_API_KEY=tvly-...
# export DRY_RUN=1              # 无 Postgres 时使用内存检查点

python qq_group_bot.py
```

看到日志：

```
[QQBot] listening http://127.0.0.1:8080 api=http://127.0.0.1:5700
```

即表示回调服务已就绪。

## 验证
- 在目标 QQ 群发送一条文本消息（如“你好”），应看到机器人回复；
- 控制台会打印 Agent 的流式 token，群内仅发送最终汇总文本。

## 故障排查
- 401 invalid signature：`ONEBOT_SECRET` 与 Napcat `http_post.secret` 不一致；
- 发送失败：检查 `ONEBOT_API_BASE` 可达与端口正确；如配置了 access_token，需携带 `Authorization: Bearer <token>`；
- 未生成回复：检查 `.venv` 是否激活、`OPENAI_API_KEY` 是否有效、是否需要 `DRY_RUN=1`；
- 登录/风控问题：按 Napcat 文档启用签名服务与稳定协议；必要时改用扫码登录降低滑块概率。

## 可选：只在 @ 机器人或前缀触发时应答
当前实现对任意群文本应答。可在 `qq_group_bot.py` 的 `do_POST` 中解析 CQ 码与文本，限制仅在被 @ 或命令前缀（如 `!ai`）时触发，以降低打扰与风控风险。

