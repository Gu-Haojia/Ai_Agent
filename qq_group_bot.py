"""
基于 OneBot v11 HTTP 回调的 QQ 群机器人（兼容 NapCat.OneBot / Lagrange.OneBot / go-cqhttp）。

特点：
- 使用 Python 标准库 http.server + urllib，无需安装第三方依赖；
- 通过 OneBot 的 HTTP 回调接收事件，调用 LangGraph SQL Agent 生成回复；
- 通过 OneBot 的 HTTP API 发送群消息；
- 所有敏感信息通过环境变量注入，不在代码中硬编码。

运行前提：
- 你已部署并登录可用的 OneBot v11 实现（如 NapCat.OneBot / Lagrange.OneBot / go-cqhttp）；
- 启用 HTTP API 与 HTTP POST 回调（回调指向本服务地址）。

环境变量：
- BOT_HOST: 监听地址，默认 127.0.0.1
- BOT_PORT: 监听端口，默认 8080
- ONEBOT_API_BASE: OneBot HTTP API Base，例如 http://127.0.0.1:5700 （推荐）
- ONEBOT_SECRET: OneBot 回调密钥（可选，若配置了回调 secret 则必须）
- ONEBOT_ACCESS_TOKEN: OneBot HTTP API access_token（可选，若配置了需在请求头携带）
- 兼容旧名：GOCQHTTP_API_BASE / GOCQHTTP_SECRET / GOCQHTTP_ACCESS_TOKEN
- ALLOWED_GROUPS: 允许响应的群ID，逗号分隔；为空表示不限制
- MODEL_NAME, LANGGRAPH_PG, THREAD_ID, ENABLE_TOOLS 等：透传给 SQL Agent

安全要求：
- 必须通过虚拟环境运行；
- 严禁硬编码密钥，统一使用环境变量。
"""

from __future__ import annotations

import hmac
import json
import os
import sys
import time
from dataclasses import dataclass
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def _load_env_from_files(files: list[str]) -> None:
    """
    从 .env/.env.local 读取 KEY=VALUE 并注入到进程环境（不覆盖已存在的环境变量）。
    Args:
        files (list[str]): 依次尝试读取的文件路径
    """
    for fp in files:
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_env_from_files([".env.local", ".env"])

# 复用现有 Agent
from sql_agent_cli_stream_plus import (
    AgentConfig,
    SQLCheckpointAgentStreamingPlus,
)


@dataclass
class BotConfig:
    """QQ 机器人配置。"""

    host: str = "127.0.0.1"
    port: int = 8080
    api_base: str = ""
    secret: str = ""
    access_token: str = ""
    allowed_groups: tuple[int, ...] = ()

    @staticmethod
    def from_env() -> "BotConfig":
        host = os.environ.get("BOT_HOST", "127.0.0.1")
        port = int(os.environ.get("BOT_PORT", "8080"))
        api_base = (os.environ.get("ONEBOT_API_BASE") or os.environ.get("GOCQHTTP_API_BASE", "")).rstrip("/")
        secret = os.environ.get("ONEBOT_SECRET") or os.environ.get("GOCQHTTP_SECRET", "")
        access_token = os.environ.get("ONEBOT_ACCESS_TOKEN") or os.environ.get("GOCQHTTP_ACCESS_TOKEN", "")
        groups_env = os.environ.get("ALLOWED_GROUPS", "").strip()
        groups: tuple[int, ...] = ()
        if groups_env:
            groups = tuple(int(x) for x in groups_env.split(",") if x.strip().isdigit())
        cfg = BotConfig(host=host, port=port, api_base=api_base, secret=secret, access_token=access_token, allowed_groups=groups)
        # 基础校验
        assert cfg.api_base, "缺少 ONEBOT_API_BASE（或 GOCQHTTP_API_BASE），例如 http://127.0.0.1:5700"
        return cfg


def _verify_signature(secret: str, body: bytes, signature: str) -> bool:
    """验证 OneBot X-Signature（HMAC-SHA1，值形如 sha1=...）。"""
    if not secret:
        return True
    if not signature or not signature.startswith("sha1="):
        return False
    mac = hmac.new(secret.encode("utf-8"), body, sha1).hexdigest()
    return hmac.compare_digest("sha1=" + mac, signature)


def _send_group_msg(api_base: str, group_id: int, text: str, access_token: str = "") -> None:
    """调用 OneBot HTTP API 发送群消息。"""
    url = urljoin(api_base + "/", "send_group_msg")
    payload = {"group_id": group_id, "message": text}
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urlopen(req, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"send_group_msg HTTP {resp.status}")


class QQBotHandler(BaseHTTPRequestHandler):
    """处理 OneBot HTTP 回调的 Handler。"""

    # 共享对象（由主程序注入）
    bot_cfg: BotConfig
    agent: SQLCheckpointAgentStreamingPlus

    def _send_json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        # 避免 http.server 默认 stdout 噪音，简化为一行
        sys.stderr.write("[HTTP] " + fmt % args + "\n")

    def do_POST(self) -> None:  # noqa: N802
        # 读取 body
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            self._send_json(400, {"error": "bad content-length"})
            return
        body = self.rfile.read(length)

        # 签名校验（可选）
        sig = self.headers.get("X-Signature", "")
        if not _verify_signature(self.bot_cfg.secret, body, sig):
            self._send_json(401, {"error": "invalid signature"})
            return

        # 解析事件
        try:
            event = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return

        # 仅处理群消息
        if not (
            event.get("post_type") == "message"
            and event.get("message_type") == "group"
        ):
            self._send_json(200, {"ok": True})
            return

        group_id = int(event.get("group_id", 0))
        user_id = int(event.get("user_id", 0))
        raw_message = str(event.get("raw_message") or event.get("message") or "").strip()

        if not raw_message:
            self._send_json(200, {"ok": True})
            return

        # 群白名单
        if self.bot_cfg.allowed_groups and group_id not in self.bot_cfg.allowed_groups:
            self._send_json(200, {"ok": True})
            return

        # 简单指令：@机器人 或 以特定前缀触发（前缀可按需扩展）
        # 这里直接对任意文本做应答，实际可加入白名单、命令开关等控制

        # 调用 Agent 生成回复（返回最后聚合文本）
        try:
            # 为流式打印添加前缀标记到服务端日志，QQ 群内仅发送最终汇总
            self.agent.set_token_printer(lambda s: sys.stdout.write(s))
            answer = self.agent.chat_once_stream(raw_message, thread_id=f"qq-group-{group_id}")
            answer = (answer or "").strip()
            if not answer:
                answer = "（未生成回复）"
        except KeyboardInterrupt:
            answer = "（生成已中断）"
        except AssertionError as e:
            answer = f"（配置错误）{e}"
        except Exception as e:
            answer = f"（内部错误）{e}"

        # 发送回群
        try:
            _send_group_msg(self.bot_cfg.api_base, group_id, answer, self.bot_cfg.access_token)
        except Exception as e:
            # 回应失败也返回 200，避免上游不断重试
            self._send_json(200, {"ok": False, "error": str(e)})
            return

        # 正常结束
        self._send_json(200, {"ok": True})


def _build_agent_from_env() -> SQLCheckpointAgentStreamingPlus:
    """从环境变量构建 SQL Agent（保持与主程序一致的约束）。"""
    model = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    pg = os.environ.get("LANGGRAPH_PG", "")
    thread = os.environ.get("THREAD_ID", "qq-demo")
    use_memory = os.environ.get("DRY_RUN") == "1" or not bool(pg)
    cfg = AgentConfig(model_name=model, pg_conn=pg, thread_id=thread, use_memory_ckpt=use_memory)
    agent = SQLCheckpointAgentStreamingPlus(cfg)
    return agent


def main() -> None:
    """启动 HTTP 服务器，接收 OneBot 回调并处理群消息。"""
    # 必须使用虚拟环境
    assert os.environ.get("VIRTUAL_ENV") or sys.prefix.endswith(
        ".venv"
    ), "必须先激活虚拟环境 (.venv)。"

    bot_cfg = BotConfig.from_env()
    agent = _build_agent_from_env()

    QQBotHandler.bot_cfg = bot_cfg
    QQBotHandler.agent = agent

    server = ThreadingHTTPServer((bot_cfg.host, bot_cfg.port), QQBotHandler)
    print(f"[QQBot] listening http://{bot_cfg.host}:{bot_cfg.port} api={bot_cfg.api_base}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[QQBot] stopped.")
    finally:
        try:
            server.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    # 可选：等待本机 Postgres/依赖准备（如需要）
    if os.environ.get("WAIT_PG") == "1":
        for _ in range(30):
            code = os.system("pg_isready -q")
            if code == 0:
                break
            time.sleep(1)
    main()
