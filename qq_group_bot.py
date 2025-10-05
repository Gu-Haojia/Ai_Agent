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

import base64
import hmac
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import ClassVar, Optional, Sequence, Union
from urllib.parse import urljoin
from urllib.request import Request, urlopen


_CQ_REPLY_PATTERN = re.compile(r"\[CQ:reply,([^\]]*)\]")


def _extract_reply_ids_from_raw(raw: str) -> tuple[str, tuple[str, ...]]:
    """
    从包含 CQ reply 标签的字符串中提取引用消息 ID 并移除标签。

    Args:
        raw (str): 原始消息字符串。

    Returns:
        tuple[str, tuple[str, ...]]: 包含清理后文本与引用 message_id 的元组。

    Raises:
        None: 不会在此函数中抛出异常。
    """
    ids: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        params = match.group(1)
        for part in params.split(","):
            key, _, value = part.partition("=")
            key = key.strip()
            value = value.strip()
            if key in {"message_id", "id"} and value:
                ids.append(value)
        return ""

    cleaned = _CQ_REPLY_PATTERN.sub(_replace, raw)
    return cleaned.strip(), tuple(ids)


def _extract_message_content(
    message: object, raw_fallback: Optional[str] = None
) -> MessageContent:
    """
    从消息对象提取文本与图像信息。

    Args:
        message (object): OneBot 返回的 message 字段。
        raw_fallback (Optional[str]): 当 message 为空时的原始字符串。

    Returns:
        MessageContent: 标准化的消息内容。
    """

    if isinstance(message, list):
        text, _at_me, images, _reply_ids = _normalize_message_segments(message)
        return MessageContent(text=text, images=images)

    raw = str(message or raw_fallback or "")
    raw, _ = _extract_reply_ids_from_raw(raw)
    images = _extract_cq_images(raw) if raw else ()
    return MessageContent(text=raw, images=images)


def _fetch_message_content(
    api_base: str, message_id: str, access_token: str = ""
) -> MessageContent:
    """
    通过 OneBot HTTP API 获取指定消息的文本与图像内容。

    Args:
        api_base (str): OneBot HTTP API 的基地址。
        message_id (str): 需要获取内容的消息 ID。
        access_token (str): API Token，可为空字符串。

    Returns:
        MessageContent: 对应消息的文本与图像内容。

    Raises:
        RuntimeError: 当 HTTP 响应码或返回 retcode 不符合预期时抛出。
    """

    assert message_id, "message_id 不可为空"
    url = urljoin(api_base + "/", "get_msg")
    payload: dict[str, object] = {}
    if message_id.isdigit():
        payload["message_id"] = int(message_id)
    else:
        payload["message_id"] = message_id
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urlopen(req, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"get_msg HTTP {resp.status}")
        body = json.loads(resp.read().decode("utf-8"))
    if body.get("status") != "ok" or body.get("retcode") not in {0, None}:
        raise RuntimeError(f"get_msg failed: retcode={body.get('retcode')}")
    data = body.get("data") or {}
    return _extract_message_content(data.get("message"), data.get("raw_message"))


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
from image_storage import GeneratedImage, ImageStorageManager, StoredImage


@dataclass
class BotConfig:
    """NapCat 机器人配置。"""

    host: str = "0.0.0.0"
    port: int = 8080
    api_base: str = "http://127.0.0.1:3000"
    secret: str = ""
    access_token: str = ""
    allowed_groups: tuple[int, ...] = ()
    blacklist_groups: tuple[int, ...] = ()
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
        blacklist_env = os.environ.get("BLACKLIST_GROUPS", "").strip()
        blacklist: tuple[int, ...] = ()
        if blacklist_env:
            blacklist = tuple(
                int(x) for x in blacklist_env.split(",") if x.strip().isdigit()
            )
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
            blacklist_groups=blacklist,
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


MessagePayload = Union[str, Sequence[dict[str, dict[str, str]]]]


def _send_group_msg(
    api_base: str, group_id: int, message: MessagePayload, access_token: str = ""
) -> None:
    """调用 OneBot HTTP API 发送群消息（NapCat 兼容）。

    Args:
        api_base (str): NapCat HTTP API 基地址。
        group_id (int): 群号。
        message (MessagePayload): 文本或消息段列表。
        access_token (str): API Token，可为空。
    """
    url = urljoin(api_base + "/", "send_group_msg")
    payload = {"group_id": group_id, "message": message}
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urlopen(req, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"send_group_msg HTTP {resp.status}")


def _send_group_at_message(
    api_base: str, group_id: int, at_qq: int, text: str, access_token: str = ""
) -> None:
    """
    调用 OneBot HTTP API 在群内发送 @ 提醒消息。

    Args:
        api_base (str): NapCat/OneBot HTTP API 基地址。
        group_id (int): 群号。
        at_qq (int): 被 @ 的 QQ 号。
        text (str): 文本内容（将放在 @ 之后）。
        access_token (str): API Token，可为空。

    Raises:
        RuntimeError: 当 HTTP 响应码非 200 时抛出。
    """
    assert isinstance(group_id, int) and group_id > 0, "group_id 必须为正整数"
    assert isinstance(at_qq, int) and at_qq > 0, "at_qq 必须为正整数"
    assert isinstance(text, str), "text 必须为字符串"
    # 复用发送群消息接口，使用 CQ 码进行 @
    msg = f"[CQ:at,qq={at_qq}] {text}"
    _send_group_msg(api_base, group_id, msg, access_token)


@dataclass(frozen=True)
class ImageSegmentInfo:
    """消息段中的图片信息。"""

    url: Optional[str]
    file_id: Optional[str]
    filename: Optional[str]


@dataclass(frozen=True)
class MessageContent:
    """标准化的消息内容（文本与图像）。"""

    text: str
    images: tuple[ImageSegmentInfo, ...]


@dataclass(frozen=True)
class ParsedMessage:
    """标准化的消息解析结果。"""

    text: str
    at_me: bool
    images: tuple[ImageSegmentInfo, ...]
    reply_message_ids: tuple[str, ...]


def _extract_cq_images(raw: str) -> tuple[ImageSegmentInfo, ...]:
    """
    从原始 CQ 文本中解析图像段。

    Args:
        raw (str): 原始消息字符串。

    Returns:
        tuple[ImageSegmentInfo, ...]: 解析出的图像段列表。
    """
    images: list[ImageSegmentInfo] = []
    idx = 0
    length = len(raw)
    while idx < length:
        start = raw.find("[CQ:image", idx)
        if start == -1:
            break
        end = raw.find("]", start)
        if end == -1:
            break
        body = raw[start + 1 : end]
        parts = body.split(",")
        data: dict[str, str] = {}
        for segment in parts[1:]:
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            data[key.strip()] = value.strip()
        images.append(
            ImageSegmentInfo(
                url=data.get("url"),
                file_id=data.get("file") or data.get("file_id"),
                filename=data.get("file") or data.get("name"),
            )
        )
        idx = end + 1
    return tuple(images)


def _normalize_message_segments(
    segments: Sequence[dict], self_id: str = ""
) -> tuple[str, bool, tuple[ImageSegmentInfo, ...], tuple[str, ...]]:
    """
    将消息段标准化为文本、@ 标记、图像与引用消息 ID。

    Args:
        segments (Sequence[dict]): OneBot 消息段列表。
        self_id (str): 机器人自身 QQ 号，用于识别 @。

    Returns:
        tuple[str, bool, tuple[ImageSegmentInfo, ...], tuple[str, ...]]: 包含文本、是否@、图片段与引用 ID。
    """
    texts: list[str] = []
    at_me = False
    images: list[ImageSegmentInfo] = []
    reply_ids: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        typ = seg.get("type")
        data = seg.get("data") or {}
        if typ == "text":
            texts.append(str(data.get("text", "")))
        elif typ == "at" and self_id:
            qq = str(data.get("qq", ""))
            if qq == self_id:
                at_me = True
        elif typ == "image":
            url = data.get("url")
            file_id = data.get("file") or data.get("file_id")
            filename = data.get("name") or data.get("file")
            images.append(
                ImageSegmentInfo(
                    url=str(url) if url else None,
                    file_id=str(file_id) if file_id else None,
                    filename=str(filename) if filename else None,
                )
            )
        elif typ == "reply":
            candidate = ""
            for key in ("message_id", "id"):
                candidate = str(data.get(key) or "").strip()
                if candidate:
                    break
            if candidate:
                reply_ids.append(candidate)
    return "".join(texts).strip(), at_me, tuple(images), tuple(reply_ids)


def _parse_message_and_at(event: dict) -> ParsedMessage:
    """
    解析 NapCat 群消息，返回文本、@ 状态与图像段。

    Args:
        event (dict): OneBot v11 事件。

    Returns:
        ParsedMessage: 标准化后的消息内容。
    """
    self_id = str(event.get("self_id") or "")

    msg = event.get("message")
    if isinstance(msg, list):
        text, at_me, images, reply_ids = _normalize_message_segments(msg, self_id)
        return ParsedMessage(text, at_me, images, reply_ids)

    raw = str(event.get("raw_message") or msg or "").strip()
    raw, reply_ids = _extract_reply_ids_from_raw(raw)
    at_me = False
    if self_id and "[CQ:at,qq=" in raw:
        at_me = f"[CQ:at,qq={self_id}]" in raw
    images = _extract_cq_images(raw) if raw else ()
    print(f"[Debug] Alternative way enabled")
    return ParsedMessage(raw, at_me, images, reply_ids)


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
    image_storage: Optional[ImageStorageManager] = None
    _post_lock: ClassVar[Lock] = Lock()  # 串行化处理 POST 请求
    # 群 -> 线程ID 映射，用于 /clear 后为群对话分配新线程
    _group_threads: dict[int, str] = {}
    _thread_store_file: str = ""
    _env_consistency_checked: bool = False  # 本次运行仅检查一次
    # 群 -> 持久记忆命名空间 映射，与线程隔离相似
    _group_namespaces: dict[int, str] = {}
    _ns_store_file: str = ""
    _env_ns_checked: bool = False

    @classmethod
    def _require_image_storage(cls) -> ImageStorageManager:
        """
        获取已初始化的图像存储管理器。

        Returns:
            ImageStorageManager: 全局共享的图像存储管理器。

        Raises:
            AssertionError: 当未注入图像存储实例时抛出。
        """
        if not isinstance(cls.image_storage, ImageStorageManager):
            raise AssertionError("图像存储管理器尚未配置")
        return cls.image_storage

    @staticmethod
    def _build_multimodal_content(
        model_input: str, images: Sequence[StoredImage]
    ) -> list[dict[str, object]]:
        """
        构造多模态消息内容列表。

        Args:
            model_input (str): 拼接后的文本输入。
            images (Sequence[StoredImage]): 已保存的图像集合。

        Returns:
            list[dict[str, object]]: 可直接传递给多模态模型的内容结构，
                在包含 Base64 数据时会同步提供对应的本地文件名。
        """
        content: list[dict[str, object]] = [{"type": "text", "text": model_input}]
        if images:
            content.append(
                {
                    "type": "text",
                    "text": f"用户同时附带了 {len(images)} 张图片，请结合视觉分析并不要回传原始图片。",
                }
            )
        for idx, stored in enumerate(images, 1):
            file_path = Path(stored.path)
            assert file_path.name, "图像文件名不能为空"
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"第 {idx} 张图像已经以内嵌 data URL 形式提供，"
                        f"本地文件名为 {file_path.name}。"
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": stored.data_url()},
                }
            )
        return content

    @staticmethod
    def _compose_group_message(
        answer: str, image_payloads: Sequence[tuple[str, str]]
    ) -> str:
        """组合文本与图片 CQ 码，图片使用 base64 内联。"""
        parts: list[str] = []
        text = answer.strip()
        if text:
            parts.append(text)
        ts = int(time.time())
        for idx, (b64, mime) in enumerate(image_payloads, 1):
            suffix = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/gif": "gif",
                "image/webp": "webp",
            }.get(mime, "jpg")
            name = f"img_{ts}_{idx}.{suffix}"
            parts.append(
                f"[CQ:image,file=base64://{b64},name={name},cache=0]"
            )
        if not parts:
            return "（未生成回复）"
        return "\n".join(parts)

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
                            prefix = f"thread-{gid}-"
                            if not tid.startswith(prefix):
                                return ""
                            tail = tid[len(prefix) :]
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
                            print(
                                f"[QQBot] Thread store cleared due to env mismatch: {path}"
                            )
                        else:
                            cls._group_threads = m
                            cls._env_consistency_checked = True
                            print(
                                f"[QQBot] Loaded group threads from {path}: {len(m)} groups"
                            )
                    else:
                        cls._group_threads = m
                        print(
                            f"[QQBot] Loaded group threads from {path}: {len(m)} groups (env check skipped)"
                        )
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

    @classmethod
    def setup_namespace_store(cls, path: str, current_env_store_id: str) -> None:
        """
        设置并加载群记忆命名空间映射配置；首次执行时做环境一致性检查。

        规则：当发现任何保存的 namespace 与当前 STORE_ID 前缀不一致时，清空并覆盖保存文件。

        Args:
            path (str): 配置文件路径
            current_env_store_id (str): 当前环境变量提供的持久记忆 store_id
        """
        cls._ns_store_file = path
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                data = {} if not (raw and raw.strip()) else json.loads(raw)
                if isinstance(data, dict):
                    m: dict[int, str] = {}
                    for k, v in data.items():
                        if str(k).isdigit() and isinstance(v, (str, int)):
                            m[int(k)] = str(v)
                    if not cls._env_ns_checked:

                        def _env_part_from_ns(gid: int, ns: str) -> str:
                            prefix = f"store-{gid}-"
                            if not ns.startswith(prefix):
                                return ""
                            tail = ns[len(prefix) :]
                            parts = tail.split("-")
                            if parts and parts[-1].isdigit():
                                return "-".join(parts[:-1])
                            return tail

                        mismatch = False
                        for gid, ns in m.items():
                            env_part = _env_part_from_ns(gid, ns)
                            if env_part and env_part != current_env_store_id:
                                mismatch = True
                                break
                        if mismatch:
                            cls._group_namespaces = {}
                            cls._env_ns_checked = True
                            cls.save_namespace_store()
                            print(
                                f"[QQBot] Namespace store cleared due to env mismatch: {path}"
                            )
                        else:
                            cls._group_namespaces = m
                            cls._env_ns_checked = True
                            print(
                                f"[QQBot] Loaded group namespaces from {path}: {len(m)} groups"
                            )
                    else:
                        cls._group_namespaces = m
                        print(
                            f"[QQBot] Loaded group namespaces from {path}: {len(m)} groups (env check skipped)"
                        )
        except Exception as e:
            sys.stderr.write(f"[QQBot] Load group namespaces failed: {e}\n")

    @classmethod
    def save_namespace_store(cls) -> None:
        """保存 `_group_namespaces` 到配置文件（若已设置路径）。"""
        if not cls._ns_store_file:
            return
        try:
            body = {str(k): str(v) for k, v in cls._group_namespaces.items()}
            tmp = cls._ns_store_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=2)
            os.replace(tmp, cls._ns_store_file)
        except Exception as e:
            sys.stderr.write(f"[QQBot] save group namespaces failed: {e}\n")

    def _send_json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_no_content(self) -> None:
        """返回 204，无内容（避免触发 NapCat 快速操作）。"""
        self._suppress_http_log = True
        self.send_response(204)
        self.end_headers()

    def _log_ignore_request(
        self, group_id: int, user_id: int, author: str, text: str
    ) -> None:
        """
        记录被忽略的非 @ 消息，并抑制默认 HTTP 日志输出。

        Args:
            group_id (int): 群号。
            user_id (int): 用户 QQ 号。
            author (str): 用户昵称或名片。
            text (str): 用户发送的原始文本。
        """
        clean_text = text.replace("\n", " ")
        print(
            f"\033[31m[Ignore]\033[0m Message get: Group {group_id} Id {user_id} User {author}: {clean_text}", flush=True
        )
        self._suppress_http_log = True

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """自定义请求日志，允许在特定场景下禁用默认输出。"""
        if getattr(self, "_suppress_http_log", False):
            self._suppress_http_log = False
            return
        super().log_request(code, size)

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

    def _collect_reply_contents(self, message_ids: Sequence[str]) -> list[MessageContent]:
        """
        批量获取引用消息对应的内容（文本与图像）。

        Args:
            message_ids (Sequence[str]): 需要查询的 message_id 列表。

        Returns:
            list[MessageContent]: 成功获取的引用消息内容列表。

        Raises:
            None: 不会在此函数中抛出异常。
        """
        contents: list[MessageContent] = []
        for mid in message_ids:
            mid_norm = mid.strip()
            if not mid_norm:
                continue
            try:
                content = _fetch_message_content(
                    self.bot_cfg.api_base, mid_norm, self.bot_cfg.access_token
                )
            except Exception as err:
                sys.stderr.write(
                    f"[Chat] 获取引用消息失败: message_id={mid_norm} err={err}\n"
                )
                continue
            contents.append(content)
        return contents

    def do_GET(self) -> None:  # noqa: N802
        """健康检查：仅 /healthz 返回 200，其他 GET 返回 501。"""
        if self.path.rstrip("/") == "/healthz":
            self._send_json(200, {"ok": True, "service": "qq-napcat-bot"})
            return
        self.send_error(501, "Unsupported method ('GET')")

    def do_POST(self) -> None:  # noqa: N802
        """
        处理 NapCat HTTP 回调的 POST 请求，并通过互斥锁串行化。

        Returns:
            None: 无返回值。

        Raises:
            None: 未在此函数中额外抛出异常。
        """
        with self.__class__._post_lock:
            self._handle_post_locked()

    def _handle_post_locked(self) -> None:
        """
        在互斥锁保护下执行 POST 请求的详细处理逻辑。

        Returns:
            None: 无返回值。

        Raises:
            None: 不在此处额外抛出异常。
        """
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
        parsed = _parse_message_and_at(event)

        if not parsed.text and not parsed.images and not parsed.reply_message_ids:
            author = _extract_sender_name(event)
            self._log_ignore_request(group_id, user_id, author, "[No text]")
            self._send_no_content()
            return

        # 群白名单
        if self.bot_cfg.allowed_groups and group_id not in self.bot_cfg.allowed_groups:
            self._send_no_content()
            return

        # 群黑名单
        if self.bot_cfg.blacklist_groups and group_id in self.bot_cfg.blacklist_groups:
            self._send_no_content()
            return

        # 仅在被 @ 机器人时响应
        if not parsed.at_me:
            author = _extract_sender_name(event)
            self._log_ignore_request(group_id, user_id, author, parsed.text if not parsed.images else parsed.text+"[with images]")
            self._send_no_content()
            return

        # 内部命令：当文本完全命中命令格式时优先处理，并直接在群内回复
        t = parsed.text.strip()
        if self._handle_commands(group_id, user_id, t):
            self._send_no_content()
            return

        # 调用 Agent 生成回复（返回最后聚合文本）
        try:
            # 终端打印服务消息
            author = _extract_sender_name(event)
            print(
                f"\033[34m[Chat]\033[0m Request get: Group {group_id} Id {user_id} User {author}: {parsed.text if not parsed.images else parsed.text+'[with images]'}"
            )
            print("\033[34m[Chat]\033[0m Thread lock enabled.Generating reply...")
            # 为流式打印添加前缀标记到服务端日志，QQ 群内仅发送最终汇总
            self.agent.set_token_printer(lambda s: sys.stdout.write(s))
            # 设置当前群的持久记忆命名空间（langmem 工具使用）
            ns = self._namespace_for(group_id)
            self.agent.set_memory_namespace(ns)
            # 轻量方案：在发给 Agent 的文本前加入说话人标识，提升区分度
            reply_contents = (
                self._collect_reply_contents(parsed.reply_message_ids)
                if parsed.reply_message_ids
                else []
            )
            reply_texts = [c.text for c in reply_contents if c.text]
            user_text = parsed.text
            if not user_text:
                if parsed.images:
                    user_text = "（用户未提供文本，仅包含图片）"
                elif reply_texts:
                    user_text = "（当前消息正文为空，仅引用其他消息）"
                else:
                    user_text = "（用户未提供文本）"
            if reply_contents:
                context_lines: list[str] = []
                for idx, content in enumerate(reply_contents, 1):
                    summary = content.text.strip()
                    if not summary:
                        summary = "（引用消息未提供文本）"
                    if content.images:
                        summary += f"（包含{len(content.images)}张图片）"
                    context_lines.append(f"引用消息{idx}: {summary}")
                context_lines.append(f"当前消息: {user_text}")
                user_text = "\n".join(context_lines)
            model_input = (
                f"Group_id: [{group_id}]; User_id: [{user_id}]; User_name: {author}; Text: {user_text}"
            )
            image_segments: list[ImageSegmentInfo] = []
            if parsed.images:
                image_segments.extend(parsed.images)
            for content in reply_contents:
                if content.images:
                    image_segments.extend(content.images)
            stored_images: list[StoredImage] = []
            if image_segments:
                storage = self._require_image_storage()
                seen_tokens: set[str] = set()
                for seg in image_segments:
                    assert (
                        seg.url
                    ), "当前仅支持通过 URL 获取的图片消息"
                    token = seg.url or seg.file_id or seg.filename or ""
                    if token and token in seen_tokens:
                        continue
                    if token:
                        seen_tokens.add(token)
                    stored_images.append(
                        storage.save_remote_image(seg.url, seg.filename)
                    )
            payload = (
                self._build_multimodal_content(model_input, stored_images)
                if stored_images
                else model_input
            )
            answer = self.agent.chat_once_stream(
                payload, thread_id=self._thread_id_for(group_id)
            )
            answer = (answer or "").strip()
            if not answer:
                answer = "（未生成回复）"
            generated_images = self.agent.consume_generated_images()
        except KeyboardInterrupt:
            answer = "（生成已中断）"
            generated_images = []
        except AssertionError as e:
            answer = f"（配置错误）{e}"
            generated_images = []
        except Exception as e:
            answer = f"（内部错误）{e}"
            generated_images = []

        image_payloads: list[tuple[str, str]] = []
        for img in generated_images:
            try:
                data_b64 = base64.b64encode(img.path.read_bytes()).decode("ascii")
                image_payloads.append((data_b64, img.mime_type))
            except Exception as err:
                sys.stderr.write(f"[Chat] 读取生成图片失败: {img.path} -> {err}\n")
        # 解析 Agent 回复中的 [IMAGE]url[/IMAGE] 标签，转换为本地图片
        image_tags = re.findall(r"\[IMAGE\](.+?)\[/IMAGE\]", answer, flags=re.IGNORECASE)
        if image_tags:
            manager = self._require_image_storage()
            failed_urls: list[str] = []
            downloaded = False
            for url in image_tags:
                url_norm = url.strip()
                if not url_norm:
                    continue
                if manager.is_generated_path(url_norm):
                    continue
                try:
                    saved = manager.save_remote_image(url_norm)
                    image_payloads.append((saved.base64_data, saved.mime_type))
                    downloaded = True
                except Exception as err:
                    failed_urls.append(url_norm)
                    sys.stderr.write(f"\033[34m[Chat]\033[0m 下载回复图片失败: {url_norm} -> {err}\n")
            cleaned = re.sub(r"\[IMAGE\].+?\[/IMAGE\]", "", answer, flags=re.IGNORECASE).strip()
            if failed_urls and downloaded:
                note = "（部分图片下载失败，已忽略无法访问的链接）"
                answer = f"{cleaned}\n{note}" if cleaned else note
            elif failed_urls and not downloaded and not image_payloads:
                answer = cleaned or "（未能下载图片，请稍后重试）"
            else:
                answer = cleaned or ("（图片已发送）" if downloaded else cleaned)

        # 解析 Agent 回复中的 CQ 图片段并下载本地
        cq_pattern = re.compile(r"\[CQ:image,([^\]]+)\]")
        cq_matches = list(cq_pattern.finditer(answer))
        if cq_matches:
            manager = self._require_image_storage()
            failed_urls: list[str] = []
            success = False
            for match in cq_matches:
                data_str = match.group(1)
                params = {}
                for part in data_str.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        params[k.strip()] = v.strip()
                file_val = params.get("file") or ""
                if not file_val:
                    continue
                if file_val.startswith("base64://"):
                    raw_b64 = file_val[len("base64://") :]
                    try:
                        stored = manager.save_base64_image(raw_b64)
                        image_payloads.append((stored.base64_data, stored.mime_type))
                        success = True
                    except Exception as err:
                        failed_urls.append("base64-data")
                        sys.stderr.write(
                            f"[Chat] 保存CQ Base64图片失败: {err}\n"
                        )
                    continue
                if file_val.startswith("http"):
                    try:
                        saved = manager.save_remote_image(file_val)
                        image_payloads.append((saved.base64_data, saved.mime_type))
                        success = True
                    except Exception as err:
                        failed_urls.append(file_val)
                        sys.stderr.write(
                            f"[Chat] 下载CQ图片失败: {file_val} -> {err}\n"
                        )
            if cq_matches:
                answer = cq_pattern.sub("", answer).strip()
            if failed_urls and success:
                note = "（部分图片下载失败，已忽略无法访问的 CQ 图片链接）"
                answer = f"{answer}\n{note}" if answer else note
            elif failed_urls and not success and not image_payloads:
                answer = answer or "（未能下载图片，请稍后重试）"

        if answer:
            #计算字数
            char_count = len(answer)
            lines = []
            prev_blank = False
            for line in answer.splitlines():
                if line.strip():  # 非空行
                    lines.append(line)
                    prev_blank = False
                else:  # 空行
                    if not prev_blank and char_count > 100:  # 上一行不是空行且字数超过100,允许一个空行
                        lines.append("")
                    prev_blank = True
            # 去掉结尾的空行
            while lines and lines[-1] == "":
                lines.pop()
            answer = "\n".join(lines)

        # 发送回群
        try:
            # 轻量方案：使用 CQ at 前缀 @ 该用户，便于区分接收者
            # at_prefix = f"[CQ:at,qq={user_id}] "
            message_body = self._compose_group_message(answer, image_payloads)
            _send_group_msg(
                self.bot_cfg.api_base,
                group_id,
                message_body,
                self.bot_cfg.access_token,
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
        new_tid = f"thread-{group_id}-{self.agent._config.thread_id}-{int(time.time())}"
        self._group_threads[group_id] = new_tid
        QQBotHandler.save_thread_store()
        return new_tid

    def _namespace_for(self, group_id: int) -> str:
        """返回当前群使用的持久记忆命名空间；若缺失则创建并保存。

        Args:
            group_id (int): 群号

        Returns:
            str: 命名空间
        """
        if group_id in self._group_namespaces:
            return self._group_namespaces[group_id]
        new_ns = f"store-{group_id}-{self.agent._config.store_id}-{int(time.time())}"
        self._group_namespaces[group_id] = new_ns
        QQBotHandler.save_namespace_store()
        return new_ns

    def _handle_commands(self, group_id: int, user_id: int, text: str) -> bool:
        """处理内部命令。

        仅在被 @ 且文本完全命中时触发：
        - /cmd                → 返回命令列表
        - /switch             → 列出 prompts 目录下可用文件名（不含后缀）
        - /switch <name>      → 切换到 prompts/<name>.txt（设置 SYS_MSG_FILE）并重建 Agent
        - /clear              → 为当前群新建线程
        - /whoami             → 先回当前系统提示词，再基于“你是谁”生成一条消息
        - /token              → 统计当前群对应线程的消息 token 数
        - /forget             → 清除当前线程的上下文记忆

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
                "高性能AI萝卜子-小妃那 Ver. 2.0\n"
                "可用命令:\n"
                "1) /cmd — 命令列表\n"
                "2) /switch — 可用 prompts\n"
                "3) /switch <name> — 切换到 <name> prompt\n"
                "4) /clear — 清除当前群聊全部记忆\n"
                "5) /whoami — 你是？\n"
                "6) /token — 输出当前 token 数\n"
                "7) /forget - 清除上下文记忆\n"
                "8) /rmdata - 清除长期记忆"
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
                msg = f"已切换到 {name} 并重建 Agent。需要清除记忆请使用/forget 命令。"
            except AssertionError as e:
                msg = f"切换失败：{e}"
            except Exception as e:
                msg = f"切换失败（内部错误）：{e}"
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        if cmd in {"/clear", "让我忘记一切吧"} and len(parts) == 1:
            new_tid = (
                f"thread-{group_id}-{self.agent._config.thread_id}-{int(time.time())}"
            )
            self._group_threads[group_id] = new_tid
            QQBotHandler.save_thread_store()
            # 同时新建长期记忆命名空间
            new_ns = (
                f"store-{group_id}-{self.agent._config.store_id}-{int(time.time())}"
            )
            self._group_namespaces[group_id] = new_ns
            QQBotHandler.save_namespace_store()
            msg = f"已为当前群新建线程：{new_tid}\n已新建长期记忆命名空间：{new_ns}"
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        if cmd == "/rmdata" and len(parts) == 1:
            # 仅新建长期记忆命名空间
            new_ns = (
                f"store-{group_id}-{self.agent._config.store_id}-{int(time.time())}"
            )
            self._group_namespaces[group_id] = new_ns
            QQBotHandler.save_namespace_store()
            msg = f"已新建长期记忆命名空间：{new_ns}"
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        if cmd == "/forget" and len(parts) == 1:
            tid = self._thread_id_for(group_id)
            try:
                self.agent.del_latest_messages(thread_id=tid)
                msg = "已清除当前线程的上下文记忆。"
            except Exception as e:
                msg = f"清除失败（内部错误）：{e}"
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

        if cmd == "/token" and len(parts) == 1:
            # 统计当前群对应线程的消息 token 数
            try:
                tid = self._thread_id_for(group_id)
                tok, cnt = self.agent.count_tokens(thread_id=tid)
                SINGLE_PRICE = 2  # cl100k_base 每 1M tokens 价格，单位美元
                PRICE = tok / 1000000 * SINGLE_PRICE
                msg = f"当前线程消息条数={cnt}，估算 tokens={tok} (cl100k_base)，下次输入费用约为 ${PRICE:.4f}"
            except AssertionError as e:
                msg = f"统计失败：{e}"
            except Exception as e:
                msg = f"统计失败（内部错误）：{e}"
            _send_group_msg(
                self.bot_cfg.api_base, group_id, msg, self.bot_cfg.access_token
            )
            return True

        return False


def _build_agent_from_env() -> SQLCheckpointAgentStreamingPlus:
    """从环境变量构建 SQL Agent（NapCat 适配，保持与主程序一致约束）。"""
    model = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    pg = os.environ.get("LANGGRAPH_PG", "")
    thread = os.environ.get("THREAD_ID", "qq-napcat-demo")
    store_id = os.environ.get("STORE_ID", "").strip()
    assert store_id, "必须通过环境变量 STORE_ID 提供持久记忆的 store_id。"
    use_memory = os.environ.get("DRY_RUN") == "1" or not bool(pg)
    cfg = AgentConfig(
        model_name=model,
        pg_conn=pg,
        thread_id=thread,
        use_memory_ckpt=use_memory,
        store_id=store_id,
    )
    agent = SQLCheckpointAgentStreamingPlus(cfg)
    return agent


def main() -> None:
    """启动 HTTP 服务器，接收 OneBot 回调并处理群消息。"""
    # 必须使用虚拟环境
    assert os.environ.get("VIRTUAL_ENV") or sys.prefix.endswith(
        ".venv"
    ), "必须先激活虚拟环境 (.venv)。"

    print(
        "--------------------------------------------------------------------------------------------------------"
    )
    bot_cfg = BotConfig.from_env()
    agent = _build_agent_from_env()
    image_dir = os.environ.get(
        "QQ_IMAGE_DIR",
        os.path.join(os.getcwd(), "local_backup", "qq_images"),
    )
    image_manager = ImageStorageManager(image_dir)
    agent.set_image_manager(image_manager)

    # 启动时加载群线程映射配置（仅保存/加载 _group_threads 字典）
    thread_store = os.environ.get("THREAD_STORE_FILE", ".qq_group_threads.json")
    QQBotHandler.setup_thread_store(thread_store, agent._config.thread_id)
    # 启动时加载群持久记忆命名空间映射配置
    ns_store = os.environ.get("MEMORY_STORE_FILE", ".qq_group_memnames.json")
    QQBotHandler.setup_namespace_store(ns_store, agent._config.store_id)

    QQBotHandler.bot_cfg = bot_cfg
    QQBotHandler.agent = agent
    QQBotHandler.image_storage = image_manager

    server = ThreadingHTTPServer((bot_cfg.host, bot_cfg.port), QQBotHandler)
    print(
        f"[QQBot] Listening http://{bot_cfg.host}:{bot_cfg.port} api={bot_cfg.api_base} whitelist={bot_cfg.allowed_groups or 'ALL'} blacklist={bot_cfg.blacklist_groups or 'NONE'} model={agent._config.model_name} dry_run={'YES' if agent._config.use_memory_ckpt else 'NO'} thread_id={agent._config.thread_id} thread_store={thread_store} mem_id={agent._config.store_id} mem_store={ns_store}"
    )
    print("[QQBot] Allowed command users:", bot_cfg.cmd_allowed_users or "ALL")
    print("[QQBot] Bot now started, press Ctrl+C to stop.")
    print(
        "--------------------------------------------------------------------------------------------------------"
    )
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
