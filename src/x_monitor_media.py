"""
X 推文监控图片抓取与群发工具。

仅在内存中下载图片并转为 base64，供 OneBot 消息段直接发送。
"""

from __future__ import annotations

import base64
import json
import sys
import time
from typing import Callable, Optional, Sequence
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import requests

from src.x_monitor import XPostResult

MessagePayload = Sequence[dict[str, dict[str, str]]] | str


def _guess_suffix(mime: str) -> str:
    """
    根据 MIME 类型推断文件后缀。

    Args:
        mime (str): MIME 类型。

    Returns:
        str: 文件后缀，不含点。

    Raises:
        None: 本函数不抛出异常。
    """
    normalized = (mime or "").lower().split(";")[0].strip()
    mapping = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }
    return mapping.get(normalized, "jpg")


def _send_group_msg(
    api_base: str, group_id: int, message: MessagePayload, access_token: str = ""
) -> None:
    """
    轻量版 OneBot send_group_msg。

    Args:
        api_base (str): OneBot HTTP API 基地址。
        group_id (int): 群号。
        message (MessagePayload): 文本或消息段列表。
        access_token (str): API Token。

    Returns:
        None: 无返回值。

    Raises:
        RuntimeError: 当 HTTP 响应码不是 200 时抛出。
    """
    assert group_id > 0, "group_id 必须为正整数"
    url = urljoin(api_base.rstrip("/") + "/", "send_group_msg")
    payload = {"group_id": group_id, "message": message}
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urlopen(req, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"send_group_msg HTTP {resp.status}")


def _pick_image_urls(items: Sequence[XPostResult], max_count: int = 5) -> list[str]:
    """
    每条推文挑一张图，按顺序去重后返回。

    Args:
        items (Sequence[XPostResult]): 推文列表。
        max_count (int): 最多返回数量。

    Returns:
        list[str]: 需下载的图片 URL。

    Raises:
        AssertionError: 当 max_count 非正数时抛出。
    """
    assert max_count > 0, "max_count 必须大于 0"
    urls: list[str] = []
    for item in items:
        for url in item.image_urls:
            if url.startswith("http") and url not in urls:
                urls.append(url)
                break
        if len(urls) >= max_count:
            break
    return urls


def _download_as_base64(
    url: str, timeout: int = 12, max_bytes: int = 900_000
) -> tuple[str, str]:
    """
    下载图片并返回 base64 字符串及 MIME。

    Args:
        url (str): 图片 URL。
        timeout (int): 超时时间（秒）。
        max_bytes (int): 允许的最大体积。

    Returns:
        tuple[str, str]: base64 字符串与 MIME 类型。

    Raises:
        ValueError: 当类型非法或体积超限时抛出。
        requests.RequestException: 网络异常。
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
    if not mime.startswith("image/"):
        raise ValueError(f"非法图片类型: {mime or 'unknown'}")
    data = resp.content
    if len(data) > max_bytes:
        raise ValueError(f"图片过大，大小 {len(data)} bytes")
    b64 = base64.b64encode(data).decode("ascii")
    return b64, mime or "image/jpeg"


def compose_x_media_message(
    text: str,
    items: Sequence[XPostResult],
    fetcher: Optional[Callable[[str], tuple[str, str]]] = None,
    max_images: int = 5,
) -> MessagePayload:
    """
    生成包含图片的 OneBot 消息段列表。

    Args:
        text (str): 文本内容。
        items (Sequence[XPostResult]): 推文列表。
        fetcher (Optional[Callable[[str], tuple[str, str]]]): 可注入图片下载器。
        max_images (int): 最多附图数量。

    Returns:
        MessagePayload: 可直接发送的消息体。

    Raises:
        AssertionError: 当文本为空时抛出。
    """
    assert text.strip(), "文本不可为空"
    urls = _pick_image_urls(items, max_images)
    segments: list[dict[str, dict[str, str]]] = [
        {"type": "text", "data": {"text": text}}
    ]
    if not urls:
        return segments
    fetch = fetcher or _download_as_base64
    ts = int(time.time())
    for idx, url in enumerate(urls, 1):
        try:
            b64, mime = fetch(url)
        except Exception as err:
            sys.stderr.write(f"[XMonitor] 图片下载失败 {url}: {err}\n")
            continue
        suffix = _guess_suffix(mime)
        segments.append(
            {
                "type": "image",
                "data": {
                    "file": f"base64://{b64}",
                    "name": f"x_{ts}_{idx}.{suffix}",
                    "cache": "0",
                },
            }
        )
    return segments


def send_x_message_with_images(
    api_base: str,
    group_id: int,
    access_token: str,
    text: str,
    items: Sequence[XPostResult],
    fetcher: Optional[Callable[[str], tuple[str, str]]] = None,
    max_images: int = 5,
) -> None:
    """
    发送携带图片的 X 推文监控消息。

    Args:
        api_base (str): OneBot API 基地址。
        group_id (int): 目标群号。
        access_token (str): API Token。
        text (str): 文本内容。
        items (Sequence[XPostResult]): 推文列表。
        fetcher (Optional[Callable[[str], tuple[str, str]]]): 自定义下载器。
        max_images (int): 附图上限。

    Returns:
        None: 无返回值。

    Raises:
        RuntimeError: 当 OneBot 发送失败时抛出。
    """
    payload = compose_x_media_message(
        text, items, fetcher=fetcher, max_images=max_images
    )
    _send_group_msg(api_base, group_id, payload, access_token)
