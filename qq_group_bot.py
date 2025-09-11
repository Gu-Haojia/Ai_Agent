"""
NapCatQQ 专用的 QQ 群机器人（OneBot v11 HTTP 回调）。

目标与拓扑：
- 仅在被 @ 机器人时触发回复；
- NapCat 运行在 Docker：
  - 宿主机上的机器人通过 `http://127.0.0.1:3000` 请求 NapCat HTTP API；
  - NapCat 容器通过 `http://host.docker.internal:8080` 回调到机器人。

特点：
- 标准库 http.server + urllib，无需第三方依赖；
- 解析 NapCat Array 格式消息段，精准识别 @；
- 通过 OneBot HTTP API 发送群消息；
- 所有敏感信息仅用环境变量注入；
- 内置 .env 加载（不覆盖现有环境变量）；
- 提供 `/healthz` GET 健康检查。

环境变量（NapCat 场景推荐）：
- BOT_HOST: 监听地址，默认 0.0.0.0（容器回调需要）
- BOT_PORT: 监听端口，默认 8080
- ONEBOT_API_BASE: NapCat HTTP API Base，默认 http://127.0.0.1:3000
- ONEBOT_SECRET: 回调签名密钥（可选）
- ONEBOT_ACCESS_TOKEN: NapCat HTTP API token（若开启验证）
- ALLOWED_GROUPS: 允许响应的群ID，逗号分隔；为空表示不限制
- CMD_ALLOWED_USERS: 命令白名单用户QQ号，逗号分隔；为空表示所有人可执行命令
- MODEL_NAME, LANGGRAPH_PG, THREAD_ID, ENABLE_TOOLS 等：透传给 SQL Agent

安全：
- 必须通过虚拟环境运行；
- 严禁硬编码密钥。
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
    """NapCat 机器人配置。"""

    host: str = "0.0.0.0"
    port: int = 8080
    api_base: str = "http://127.0.0.1:3000"
    secret: str = ""
    access_token: str = ""
    allowed_groups: tuple[int, ...] = ()
    cmd_allowed_users: tuple[int, ...] = ()  # 命令白名单用户（为空表示放行）

    @staticmethod
    def from_env() -> "BotConfig":
        host = os.environ.get("BOT_HOST", "0.0.0.0")
        port = int(os.environ.get("BOT_PORT", "8080"))
        api_base = (os.environ.get("ONEBOT_API_BASE", "http://127.0.0.1:3000")).rstrip(
            "/"
        )
        secret = os.environ.get("ONEBOT_SECRET") or ""
        access_token = os.environ.get("ONEBOT_ACCESS_TOKEN") or ""
        groups_env = os.environ.get("ALLOWED_GROUPS", "").strip()
        groups: tuple[int, ...] = ()
        if groups_env:
            groups = tuple(int(x) for x in groups_env.split(",") if x.strip().isdigit())
        # 命令白名单（逗号分隔 QQ 号），为空表示放行
        cmd_env = os.environ.get("CMD_ALLOWED_USERS", "").strip()
        cmd_allowed: tuple[int, ...] = ()
        if cmd_env:
            cmd_allowed = tuple(
                int(x) for x in cmd_env.split(",") if x.strip().isdigit()
            )
        cfg = BotConfig(
            host=host,
            port=port,
            api_base=api_base,
            secret=secret,
            access_token=access_token,
            allowed_groups=groups,
            cmd_allowed_users=cmd_allowed,
        )
        # 基础校验
        assert cfg.api_base, "缺少 ONEBOT_API_BASE，例如 http://127.0.0.1:3000"
        return cfg


def _verify_signature(secret: str, body: bytes, signature: str) -> bool:
    """验证 OneBot X-Signature（HMAC-SHA1，值形如 sha1=...）。"""
    if not secret:
        return True
    if not signature or not signature.startswith("sha1="):
        return False
    mac = hmac.new(secret.encode("utf-8"), body, sha1).hexdigest()
    return hmac.compare_digest("sha1=" + mac, signature)


def _send_group_msg(
    api_base: str, group_id: int, text: str, access_token: str = ""
) -> None:
    """调用 OneBot HTTP API 发送群消息（NapCat 兼容）。

    Args:
        api_base (str): NapCat HTTP API 基地址。
        group_id (int): 群号。
        text (str): 文本内容。
        access_token (str): API Token，可为空。
    """
    url = urljoin(api_base + "/", "send_group_msg")
    payload = {"group_id": group_id, "message": text}
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urlopen(req, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"send_group_msg HTTP {resp.status}")


def _parse_message_and_at(event: dict) -> tuple[str, bool]:
    """解析 NapCat 群消息，返回纯文本与是否@机器人。

    NapCat 的消息可为两种格式：
    - String（CQ 码）；
    - Array（段落列表，如 {"type":"text"|"at"|...}）。

    Args:
        event (dict): OneBot v11 事件。

    Returns:
        tuple[str, bool]: (纯文本, 是否@机器人)。
    """
    self_id = str(event.get("self_id") or "")

    msg = event.get("message")
    if isinstance(msg, list):
        texts: list[str] = []
        at_me = False
        for seg in msg:
            try:
                typ = seg.get("type")
                data = seg.get("data") or {}
            except Exception:
                continue
            if typ == "text":
                texts.append(str(data.get("text", "")))
            elif typ == "at":
                qq = str(data.get("qq", ""))
                if self_id and qq == self_id:
                    at_me = True
        return ("".join(texts).strip(), at_me)

    raw = str(event.get("raw_message") or msg or "").strip()
    at_me = False
    if self_id and "[CQ:at,qq=" in raw:
        at_me = f"[CQ:at,qq={self_id}]" in raw
    return (raw, at_me)

def _extract_sender_name(event: dict) -> str:
    """从事件中提取发送者名称（不额外请求 API）。

    优先级：sender.card > sender.nickname > user_id。

    Args:
        event (dict): OneBot v11 事件

    Returns:
        str: 显示名或 QQ 字符串
    """
    try:
        sender = event.get("sender") or {}
        card = str(sender.get("card") or "").strip()
        if card:
            return card
        nickname = str(sender.get("nickname") or "").strip()
        if nickname:
            return nickname
    except Exception:
        pass
    uid = event.get("user_id")
    return str(uid) if uid is not None else "user"


def _list_prompt_names() -> list[str]:
    """列出 prompts 目录下的可用提示词名称（不含扩展名）。

    Returns:
        list[str]: 文件名去掉后缀的列表，按名称排序。
    """
    base = os.path.abspath(os.path.join(os.getcwd(), "prompts"))
    if not os.path.isdir(base):
        return []
    names: list[str] = []
    try:
        for fn in os.listdir(base):
            if fn.startswith("."):
                continue
            if fn.lower().endswith(".txt"):
                names.append(os.path.splitext(fn)[0])
    except Exception:
        pass
    return sorted(set(names))


class QQBotHandler(BaseHTTPRequestHandler):
    """处理 NapCat / OneBot HTTP 回调的 Handler。"""

    # 共享对象（由主程序注入）
    bot_cfg: BotConfig
    agent: SQLCheckpointAgentStreamingPlus
    # 群 -> 线程ID 映射，用于 /clear 后为群对话分配新线程
    _group_threads: dict[int, str] = {}
    _thread_store_file: str = ""
    _env_consistency_checked: bool = False  # 本次运行仅检查一次

    @classmethod
    def setup_thread_store(cls, path: str, current_env_tid: str) -> None:
        """设置并加载群线程映射配置文件；仅首次执行环境一致性检查。

        当发现任何保存的线程ID与当前环境线程前缀不一致时，清空并覆盖保存文件。

        Args:
            path (str): 配置文件路径
            current_env_tid (str): 当前环境变量提供的线程前缀（agent._config.thread_id）
        """
        cls._thread_store_file = path
        try:
            if os.path.isfile(path):
                # 兼容空文件：当内容为空或仅空白时按空映射处理
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                data = {} if not (raw and raw.strip()) else json.loads(raw)
                if isinstance(data, dict):
                    m: dict[int, str] = {}
                    for k, v in data.items():
                        if str(k).isdigit() and isinstance(v, (str, int)):
                            m[int(k)] = str(v)
                    # 一致性检查：仅在本次运行的第一次做
                    if not cls._env_consistency_checked:
                        def _env_part_from_full(gid: int, tid: str) -> str:
                            prefix = f"qq-napcat-{gid}-"
                            if not tid.startswith(prefix):
                                return ""
                            tail = tid[len(prefix):]
                            # 若末段为纯数字（时间戳），则去掉末段
                            parts = tail.split("-")
                            if parts and parts[-1].isdigit():
                                return "-".join(parts[:-1])
                            return tail

                        mismatch = False
                        for gid, tid in m.items():
                            env_part = _env_part_from_full(gid, tid)
                            if env_part and env_part != current_env_tid:
                                mismatch = True
                                break
                        if mismatch:
                            cls._group_threads = {}
                            cls._env_consistency_checked = True
                            cls.save_thread_store()
                            print(f"[QQBot] Thread store cleared due to env mismatch: {path}")
                        else:
                            cls._group_threads = m
                            cls._env_consistency_checked = True
                            print(f"[QQBot] Loaded group threads from {path}: {len(m)} groups")
                    else:
                        cls._group_threads = m
                        print(f"[QQBot] Loaded group threads from {path}: {len(m)} groups (env check skipped)")
        except Exception as e:
            sys.stderr.write(f"[QQBot] Load group threads failed: {e}\n")

    @classmethod
    def save_thread_store(cls) -> None:
        """将 `_group_threads` 字典保存到配置文件（若已设置路径）。"""
        if not cls._thread_store_file:
            return
        try:
            body = {str(k): str(v) for k, v in cls._group_threads.items()}
            tmp = cls._thread_store_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=2)
            os.replace(tmp, cls._thread_store_file)
        except Exception as e:
            sys.stderr.write(f"[QQBot] save group threads failed: {e}\n")

    def _send_json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_no_content(self) -> None:
        """返回 204，无内容（避免触发 NapCat 快速操作）。"""
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        # 避免 http.server 默认 stdout 噪音，简化为一行
        sys.stderr.write("[HTTP] " + fmt % args + "\n")

    # ---- 内部工具：读取请求体（支持 chunked） ----
    def _read_body(self) -> tuple[bytes, Optional[str]]:
        """读取请求体，支持 Content-Length 与 chunked。

        Returns:
            tuple[bytes, Optional[str]]: (body, error)。当 error 非空时表示读取失败。
        """
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            try:
                chunks: list[bytes] = []
                while True:
                    line = self.rfile.readline()
                    if not line:
                        return b"", "bad chunk stream"
                    size_s = line.strip().split(b";", 1)[0]
                    try:
                        size = int(size_s, 16)
                    except Exception:
                        return b"", "bad chunk size"
                    if size == 0:
                        _ = self.rfile.readline()
                        break
                    data = self.rfile.read(size)
                    _ = self.rfile.read(2)
                    if data is None:
                        return b"", "incomplete chunk"
                    chunks.append(data)
                return b"".join(chunks), None
            except Exception as e:
                return b"", f"chunked read error: {e}"
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            return b"", "bad content-length"
        if length <= 0:
            return b"", "empty body"
        try:
            body = self.rfile.read(length)
        except Exception:
            return b"", "read body failed"
        return body or b"", None

    def do_GET(self) -> None:  # noqa: N802
        """健康检查：仅 /healthz 返回 200，其他 GET 返回 501。"""
        if self.path.rstrip("/") == "/healthz":
            self._send_json(200, {"ok": True, "service": "qq-napcat-bot"})
            return
        self.send_error(501, "Unsupported method ('GET')")

    def do_POST(self) -> None:  # noqa: N802
        # 读取 body（支持 chunked）
        body, err = self._read_body()
        if err:
            sys.stderr.write(f"[HTTP] bad request body: {err}\n")
            self._send_json(400, {"error": err})
            return

        # 签名校验（可选）
        sig = self.headers.get("X-Signature", "")
        if not _verify_signature(self.bot_cfg.secret, body, sig):
            self._send_json(401, {"error": "invalid signature"})
            return

        # 解析事件
        try:
            event = json.loads(body.decode("utf-8"))
        except Exception as e:
            preview = body[:256]
            sys.stderr.write(f"[HTTP] invalid json: {e}; body[:256]={preview!r}\n")
            self._send_json(400, {"error": "invalid json"})
            return

        # 仅处理群消息
        if not (
            event.get("post_type") == "message" and event.get("message_type") == "group"
        ):
            self._send_no_content()
            return

        group_id = int(event.get("group_id", 0))
        user_id = int(event.get("user_id", 0))
        text, at_me = _parse_message_and_at(event)

        if not text:
            self._send_no_content()
            return

        # 群白名单
        if self.bot_cfg.allowed_groups and group_id not in self.bot_cfg.allowed_groups:
            self._send_no_content()
            return

        # 仅在被 @ 机器人时响应
        if not at_me:
            self._send_no_content()
            return

        # 内部命令：当文本完全命中命令格式时优先处理，并直接在群内回复
        t = text.strip()
        if self._handle_commands(group_id, user_id, t):
            self._send_no_content()
            return

        # 调用 Agent 生成回复（返回最后聚合文本）
        try:
            # 终端打印服务消息
            author = _extract_sender_name(event)
            print(f"[Chat] Request get: Group {group_id} Id {user_id} User {author}: {text}")
            print("[Chat] Generating reply...")
            # 为流式打印添加前缀标记到服务端日志，QQ 群内仅发送最终汇总
            self.agent.set_token_printer(lambda s: sys.stdout.write(s))
            # 轻量方案：在发给 Agent 的文本前加入说话人标识，提升区分度
            model_input = f"User_id: [{user_id}]; User_name: {author}; Text: {text}"
            answer = self.agent.chat_once_stream(
                model_input, thread_id=self._thread_id_for(group_id)
            )
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
            # 轻量方案：使用 CQ at 前缀 @ 该用户，便于区分接收者
            #at_prefix = f"[CQ:at,qq={user_id}] "
            _send_group_msg(
                self.bot_cfg.api_base, group_id, answer, self.bot_cfg.access_token
            )
        except Exception as e:
            # 回应失败也返回 204，避免 NapCat 将响应体解析为快速操作
            sys.stderr.write(f"[HTTP] send_group_msg failed: {e}\n")
            self._send_no_content()
            return

        # 正常结束
        self._send_no_content()

    # ---- 命令与线程工具 ----
    def _thread_id_for(self, group_id: int) -> str:
        """返回当前群使用的线程 ID；若缺失则创建与 /clear 相同规则的标准线程并保存。

        Args:
            group_id (int): 群号

        Returns:
            str: 线程ID
        """
        if group_id in self._group_threads:
            return self._group_threads[group_id]
        new_tid = f"qq-napcat-{group_id}-{self.agent._config.thread_id}-{int(time.time())}"
        self._group_threads[group_id] = new_tid
        QQBotHandler.save_thread_store()
        return new_tid

    def _handle_commands(self, group_id: int, user_id: int, text: str) -> bool:
        """处理内部命令。

        仅在被 @ 且文本完全命中时触发：
        - /cmd                → 返回命令列表
        - /switch             → 列出 prompts 目录下可用文件名（不含后缀）
        - /switch <name>      → 切换到 prompts/<name>.txt（设置 SYS_MSG_FILE）并重建 Agent
        - /clear              → 为当前群新建线程
        - /whoami             → 先回当前系统提示词，再基于“你是谁”生成一条消息

        Args:
            group_id (int): 群号
            user_id (int): 触发命令的用户 QQ 号
            text (str): 纯文本

        Returns:
            bool: 是否命中并处理了命令
        """

        if not text.startswith("/"):
            if text != "让我忘记一切吧":
                return False
        parts = text.split()
        cmd = parts[0]
        # 命令白名单校验：设置了 CMD_ALLOWED_USERS 时，仅白名单内用户可执行
        allow_users = getattr(self.bot_cfg, "cmd_allowed_users", ()) or ()
        if allow_users and user_id not in allow_users:
            _send_group_msg(
                self.bot_cfg.api_base,
                group_id,
                "无权执行命令（需在白名单内）。",
                self.bot_cfg.access_token,
            )
            return True

        if cmd == "/cmd" and len(parts) == 1:
            msg = (
                "可用命令:\n"
                "1) /cmd — 显示命令列表\n"
                "2) /switch — 列出可用 prompts\n"
                "3) /switch <name> — 切换到 <name> 并重建 Agent\n"
                "4) /clear — 为当前群新建对话线程\n"
                "5) /whoami — 你是？"
            )
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        if cmd == "/switch" and len(parts) == 1:
            names = _list_prompt_names()
            msg = (
                ("可用 prompts: \n    " + "\n    ".join(names))
                if names
                else "未找到可用 prompts（请在 prompts/ 放置 .txt 文件）。"
            )
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        if cmd == "/switch" and len(parts) == 2:
            name = parts[1].strip()
            base = os.path.abspath(os.path.join(os.getcwd(), "prompts"))
            path = os.path.join(base, f"{name}.txt")
            if not os.path.isfile(path):
                msg = f"切换失败：文件不存在 prompts/{name}.txt"
                _send_group_msg(
                    self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
                )
                return True
            os.environ["SYS_MSG_FILE"] = path
            try:
                new_agent = _build_agent_from_env()
                QQBotHandler.agent = new_agent
                msg = f"已切换到 {name} 并重建 Agent。需要清除记忆请使用/clear 命令。"
            except AssertionError as e:
                msg = f"切换失败：{e}"
            except Exception as e:
                msg = f"切换失败（内部错误）：{e}"
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        if cmd in {"/clear", "让我忘记一切吧"} and len(parts) == 1:
            new_tid = f"qq-napcat-{group_id}-{self.agent._config.thread_id}-{int(time.time())}"
            self._group_threads[group_id] = new_tid
            QQBotHandler.save_thread_store()
            msg = f"已为当前群新建线程：{new_tid}"
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        if cmd == "/whoami" and len(parts) == 1:
            # 1) 读取当前系统提示文件名（不泄露全文）
            prompt_file = os.environ.get("SYS_MSG_FILE") or ""
            prompt_name = (
                os.path.splitext(os.path.basename(prompt_file))[0]
                if prompt_file
                else "未配置"
            )

            # 2) 基于“你是谁”生成一条消息
            try:
                self.agent.set_token_printer(lambda s: sys.stdout.write(s))
                answer = self.agent.chat_once_stream(
                    "你是谁", thread_id=self._thread_id_for(group_id)
                )
                answer = (answer or "").strip() or "（未生成回复）"
            except Exception as e:
                answer = f"（生成失败）{e}"

            # 3) 合并为一条群消息发送
            combined = f"当前Prompt：{prompt_name}\n{answer}"
            _send_group_msg(
                self.bot_cfg.api_base, group_id, combined, self.bot_cfg.access_token
            )
            return True

        return False


def _build_agent_from_env() -> SQLCheckpointAgentStreamingPlus:
    """从环境变量构建 SQL Agent（NapCat 适配，保持与主程序一致约束）。"""
    model = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    pg = os.environ.get("LANGGRAPH_PG", "")
    thread = os.environ.get("THREAD_ID", "qq-napcat-demo")
    use_memory = os.environ.get("DRY_RUN") == "1" or not bool(pg)
    cfg = AgentConfig(
        model_name=model, pg_conn=pg, thread_id=thread, use_memory_ckpt=use_memory
    )
    agent = SQLCheckpointAgentStreamingPlus(cfg)
    return agent


def main() -> None:
    """启动 HTTP 服务器，接收 OneBot 回调并处理群消息。"""
    # 必须使用虚拟环境
    assert os.environ.get("VIRTUAL_ENV") or sys.prefix.endswith(
        ".venv"
    ), "必须先激活虚拟环境 (.venv)。"

    print('--------------------------------------------------------------------------------------------------------')
    bot_cfg = BotConfig.from_env()
    agent = _build_agent_from_env()

    # 启动时加载群线程映射配置（仅保存/加载 _group_threads 字典）
    thread_store = os.environ.get("THREAD_STORE_FILE", ".qq_group_threads.json")
    QQBotHandler.setup_thread_store(thread_store, agent._config.thread_id)

    QQBotHandler.bot_cfg = bot_cfg
    QQBotHandler.agent = agent

    server = ThreadingHTTPServer((bot_cfg.host, bot_cfg.port), QQBotHandler)
    print(
        f"[QQBot] Listening http://{bot_cfg.host}:{bot_cfg.port} api={bot_cfg.api_base} groups={bot_cfg.allowed_groups or 'ALL'} thread={agent._config.thread_id} model={agent._config.model_name} dry_run={'YES' if agent._config.use_memory_ckpt else 'NO'} store={thread_store}"
    )
    print("[QQBot] Allowed command users:", bot_cfg.cmd_allowed_users or "ALL")
    print("[QQBot] Bot now started, press Ctrl+C to stop.")
    print('--------------------------------------------------------------------------------------------------------')
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
