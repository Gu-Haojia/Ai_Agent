"""
X 推文监控图片抓取与群发工具。

仅在内存中下载图片并转为 base64，供 OneBot 消息段直接发送。
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Callable, Optional, Sequence
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import requests

from src.x_monitor import XPostResult
from src.x_monitor_render import BrowserTweetRenderer, XRenderedTweet, XTweetPayloadParser
from src.x_monitor_translate import (
    XRenderedTweetTextTranslator,
    XTweetTranslationMode,
)

MessagePayload = Sequence[dict[str, dict[str, str]]] | str
RenderedImageFetcher = Callable[[XPostResult], tuple[str, str]]
LEGACY_MEDIA_ENV = "X_MONITOR_LEGACY_MEDIA"
INCLUDE_TEXT_ENV = "X_MONITOR_INCLUDE_TEXT"


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


def _env_flag(name: str) -> bool:
    """
    判断环境变量是否开启。

    Args:
        name (str): 环境变量名。

    Returns:
        bool: 变量值为 1/true/yes/on 时返回 True。

    Raises:
        AssertionError: 当环境变量名为空时抛出。
    """
    assert name.strip(), "环境变量名不能为空"
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    with urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"send_group_msg HTTP {resp.status}")


def _pick_image_urls(
    items: Sequence[XPostResult], max_count: Optional[int] = None
) -> list[str]:
    """
    收集全部推文图片，按顺序去重后返回。

    Args:
        items (Sequence[XPostResult]): 推文列表。
        max_count (Optional[int]): 最多返回数量，None 表示不限制。

    Returns:
        list[str]: 需下载的图片 URL。

    Raises:
        AssertionError: 当 max_count 非正数时抛出。
    """
    if max_count is not None:
        assert max_count > 0, "max_count 必须大于 0"
    urls: list[str] = []
    for item in items:
        for url in item.image_urls:
            if url.startswith("http") and url not in urls:
                urls.append(url)
            if max_count is not None and len(urls) >= max_count:
                return urls
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


class XPostImagePayloadBuilder:
    """
    将 XPostResult 渲染为 OneBot 可发送的 base64 图片。
    """

    def __init__(
        self,
        parser: Optional[XTweetPayloadParser] = None,
        renderer: Optional[BrowserTweetRenderer] = None,
        text_translator: Optional[XRenderedTweetTextTranslator] = None,
    ) -> None:
        """
        初始化推文图片构建器。

        Args:
            parser (Optional[XTweetPayloadParser]): X API payload 解析器。
            renderer (Optional[BrowserTweetRenderer]): 推文截图渲染器。
            text_translator (Optional[XRenderedTweetTextTranslator]):
                推文正文翻译器。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self._parser = parser or XTweetPayloadParser()
        self._renderer = renderer or BrowserTweetRenderer()
        self._text_translator = text_translator or XRenderedTweetTextTranslator()

    def render(self, item: XPostResult) -> tuple[str, str]:
        """
        将单条推文渲染为 base64 PNG。

        Args:
            item (XPostResult): 推文结果。

        Returns:
            tuple[str, str]: base64 图片内容与 MIME。

        Raises:
            AssertionError: 当推文缺少原始 payload 或无法匹配推文 ID 时抛出。
            RuntimeError: 当浏览器截图失败时抛出。
        """
        payload = item.source_payload
        assert payload is not None, "生成推文图片需要 source_payload"
        tweets = self._parser.parse(payload)
        tweet = self._find_tweet(tweets, item.post_id)
        self._text_translator.apply(tweet, XTweetTranslationMode.from_env())
        png = self._renderer.render_to_png_bytes(tweet)
        assert png, "生成的推文图片不能为空"
        return base64.b64encode(png).decode("ascii"), "image/png"

    def _find_tweet(
        self, tweets: Sequence[XRenderedTweet], post_id: str
    ) -> XRenderedTweet:
        """
        从解析结果中查找指定推文。

        Args:
            tweets (Sequence[XRenderedTweet]): 解析后的推文集合。
            post_id (str): 目标推文 ID。

        Returns:
            XRenderedTweet: 匹配的推文。

        Raises:
            AssertionError: 当推文 ID 为空或没有匹配结果时抛出。
        """
        normalized = post_id.strip()
        assert normalized, "post_id 不能为空"
        for tweet in tweets:
            if tweet.tweet_id == normalized:
                return tweet
        raise AssertionError(f"source_payload 中缺少推文 {normalized}")


def _compose_rendered_tweet_message(
    text: str,
    items: Sequence[XPostResult],
    renderer: Optional[RenderedImageFetcher] = None,
    include_text: Optional[bool] = None,
) -> MessagePayload:
    """
    生成解析后推文截图消息段。

    Args:
        text (str): 原文本通知内容。
        items (Sequence[XPostResult]): 推文列表。
        renderer (Optional[RenderedImageFetcher]): 可注入图片渲染器。
        include_text (Optional[bool]): 是否附带原文本，None 时读取环境变量。

    Returns:
        MessagePayload: OneBot 消息体。

    Raises:
        AssertionError: 当推文列表或渲染结果为空时抛出。
    """
    assert items, "推文列表不能为空"
    should_include_text = (
        _env_flag(INCLUDE_TEXT_ENV) if include_text is None else include_text
    )
    segments: list[dict[str, dict[str, str]]] = []
    if should_include_text:
        assert text.strip(), "文本不可为空"
        segments.append({"type": "text", "data": {"text": text}})
    render = renderer or XPostImagePayloadBuilder().render
    ts = int(time.time())
    for idx, item in enumerate(items, 1):
        b64, mime = render(item)
        assert b64.strip(), "推文图片 base64 不能为空"
        assert mime.startswith("image/"), "推文图片 MIME 必须为 image/*"
        suffix = _guess_suffix(mime)
        segments.append(
            {
                "type": "image",
                "data": {
                    "file": f"base64://{b64}",
                    "name": f"x_rendered_{ts}_{idx}.{suffix}",
                    "cache": "0",
                },
            }
        )
    assert segments, "消息段不能为空"
    return segments


def _compose_legacy_media_message(
    text: str,
    items: Sequence[XPostResult],
    fetcher: Optional[Callable[[str], tuple[str, str]]] = None,
    max_images: Optional[int] = None,
) -> MessagePayload:
    """
    生成旧版文本与原推文图片消息段。

    Args:
        text (str): 文本内容。
        items (Sequence[XPostResult]): 推文列表。
        fetcher (Optional[Callable[[str], tuple[str, str]]]): 可注入图片下载器。
        max_images (Optional[int]): 最多附图数量，None 表示发送全部图片。

    Returns:
        MessagePayload: OneBot 消息体。

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
        except (requests.RequestException, ValueError, RuntimeError) as err:
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


def compose_x_media_message(
    text: str,
    items: Sequence[XPostResult],
    fetcher: Optional[Callable[[str], tuple[str, str]]] = None,
    max_images: Optional[int] = None,
    renderer: Optional[RenderedImageFetcher] = None,
    include_text: Optional[bool] = None,
) -> MessagePayload:
    """
    生成 X 推文监控 OneBot 消息段列表。

    Args:
        text (str): 文本内容。
        items (Sequence[XPostResult]): 推文列表。
        fetcher (Optional[Callable[[str], tuple[str, str]]]): 旧版原图下载器。
        max_images (Optional[int]): 旧版原图附图数量上限。
        renderer (Optional[RenderedImageFetcher]): 解析后截图渲染器。
        include_text (Optional[bool]): 是否附带文本，None 时读取环境变量。

    Returns:
        MessagePayload: 可直接发送的消息体。

    Raises:
        AssertionError: 当推文列表或消息内容为空时抛出。
    """
    if _env_flag(LEGACY_MEDIA_ENV):
        return _compose_legacy_media_message(
            text, items, fetcher=fetcher, max_images=max_images
        )
    return _compose_rendered_tweet_message(
        text, items, renderer=renderer, include_text=include_text
    )


def send_x_message_with_images(
    api_base: str,
    group_id: int,
    access_token: str,
    text: str,
    items: Sequence[XPostResult],
    fetcher: Optional[Callable[[str], tuple[str, str]]] = None,
    max_images: Optional[int] = None,
    renderer: Optional[RenderedImageFetcher] = None,
    include_text: Optional[bool] = None,
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
        max_images (Optional[int]): 附图上限，None 表示发送全部图片。
        renderer (Optional[RenderedImageFetcher]): 解析后截图渲染器。
        include_text (Optional[bool]): 是否附带文本，None 时读取环境变量。

    Returns:
        None: 无返回值。

    Raises:
        RuntimeError: 当 OneBot 发送失败时抛出。
    """
    payload = compose_x_media_message(
        text,
        items,
        fetcher=fetcher,
        max_images=max_images,
        renderer=renderer,
        include_text=include_text,
    )
    _send_group_msg(api_base, group_id, payload, access_token)
