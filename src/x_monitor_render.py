"""
X 推文 payload 解析与截图渲染工具。

本模块将 X API v2 响应解析为轻量推文模型，并使用与 XMonitor 项目一致的
HTML/CSS + Chromium 截图方式生成推文图片。
"""

from __future__ import annotations

import copy
import html
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(slots=True)
class XRenderedUser:
    """
    表示用于渲染的 X 用户。

    Attributes:
        user_id (str): X 用户 ID。
        name (str): 用户显示名。
        username (str): 用户名。
        profile_image_url (Optional[str]): 头像 URL。
    """

    user_id: str
    name: str
    username: str
    profile_image_url: Optional[str] = None

    @property
    def handle(self) -> str:
        """
        返回带 @ 前缀的用户名。

        Returns:
            str: 用于展示的 handle。

        Raises:
            None: 本属性不抛出异常。
        """
        return f"@{self.username}" if self.username else "@unknown"

    @property
    def high_res_profile_image_url(self) -> Optional[str]:
        """
        返回高分辨率头像 URL。

        Returns:
            Optional[str]: 高分辨率头像 URL，缺失时为 None。

        Raises:
            None: 本属性不抛出异常。
        """
        if not self.profile_image_url:
            return None
        return self.profile_image_url.replace("_normal.", "_400x400.")


@dataclass(slots=True)
class XRenderedMedia:
    """
    表示用于渲染的推文媒体。

    Attributes:
        media_key (str): X API 媒体键。
        media_type (str): 媒体类型。
        url (Optional[str]): 图片 URL。
        preview_image_url (Optional[str]): 视频或 GIF 预览图 URL。
        width (Optional[int]): 媒体宽度。
        height (Optional[int]): 媒体高度。
        alt_text (Optional[str]): 替代文本。
    """

    media_key: str
    media_type: str
    url: Optional[str] = None
    preview_image_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    alt_text: Optional[str] = None

    @property
    def best_url(self) -> Optional[str]:
        """
        返回最适合渲染的媒体 URL。

        Returns:
            Optional[str]: 图片 URL 或预览图 URL。

        Raises:
            None: 本属性不抛出异常。
        """
        return self.url or self.preview_image_url


@dataclass(slots=True)
class XRenderedReference:
    """
    表示推文引用关系。

    Attributes:
        reference_type (str): 引用类型，如 quoted 或 retweeted。
        tweet (XRenderedTweet): 被引用的推文。
    """

    reference_type: str
    tweet: XRenderedTweet


@dataclass(slots=True)
class XRenderedTweet:
    """
    表示用于渲染的推文。

    Attributes:
        tweet_id (str): 推文 ID。
        text (str): 清洗后的推文正文。
        author (XRenderedUser): 推文作者。
        created_at (Optional[str]): X API 原始 ISO 时间。
        media (list[XRenderedMedia]): 推文媒体集合。
        references (list[XRenderedReference]): 引用推文集合。
        translation_text (Optional[str]): 对照模式下单独渲染的简体中文译文。
        is_translated_text (bool): 主正文是否已经替换为简体中文译文。
        raw (dict[str, Any]): 原始推文对象副本。
    """

    tweet_id: str
    text: str
    author: XRenderedUser
    created_at: Optional[str] = None
    media: list[XRenderedMedia] = field(default_factory=list)
    references: list[XRenderedReference] = field(default_factory=list)
    translation_text: Optional[str] = None
    is_translated_text: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def first_reference(self, reference_type: str) -> Optional[XRenderedReference]:
        """
        返回指定类型的第一条引用。

        Args:
            reference_type (str): 引用类型。

        Returns:
            Optional[XRenderedReference]: 匹配的引用，缺失时为 None。

        Raises:
            None: 本方法不主动抛出异常。
        """
        for reference in self.references:
            if reference.reference_type == reference_type:
                return reference
        return None

    @property
    def repost(self) -> Optional[XRenderedReference]:
        """
        返回转发引用。

        Returns:
            Optional[XRenderedReference]: 转发引用，缺失时为 None。

        Raises:
            None: 本属性不抛出异常。
        """
        return self.first_reference("retweeted")

    @property
    def quote(self) -> Optional[XRenderedReference]:
        """
        返回引用推文。

        Returns:
            Optional[XRenderedReference]: 引用推文，缺失时为 None。

        Raises:
            None: 本属性不抛出异常。
        """
        return self.first_reference("quoted")


@dataclass(frozen=True)
class BrowserRenderConfig:
    """
    浏览器截图渲染配置。

    Attributes:
        width (int): 截图宽度。
        padding (int): 推文外边距。
        avatar_size (int): 头像尺寸。
        device_scale_factor (float): 浏览器截图设备缩放因子。
        timeout_ms (int): 浏览器等待超时时间。
        timezone (str): 时间展示时区。
    """

    width: int = 900
    padding: int = 34
    avatar_size: int = 58
    device_scale_factor: float = 1.2
    timeout_ms: int = 30000
    timezone: str = "Asia/Tokyo"


class XTweetPayloadParser:
    """
    将 X API v2 响应解析为可渲染推文模型。
    """

    def parse(self, payload: Mapping[str, Any]) -> list[XRenderedTweet]:
        """
        解析 X API 响应。

        Args:
            payload (Mapping[str, Any]): X API v2 响应。

        Returns:
            list[XRenderedTweet]: 解析后的推文列表。

        Raises:
            AssertionError: 当 payload 结构与预期不符时抛出。
        """
        includes = payload.get("includes") or {}
        assert isinstance(includes, Mapping), "X API includes 必须为对象"
        users = {
            str(user["id"]): self._parse_user(user)
            for user in _as_mapping_list(includes.get("users"))
            if user.get("id") is not None
        }
        media = {
            str(item["media_key"]): self._parse_media(item)
            for item in _as_mapping_list(includes.get("media"))
            if item.get("media_key") is not None
        }
        raw_tweets = {
            str(tweet["id"]): tweet
            for tweet in _all_raw_tweets(payload)
            if tweet.get("id") is not None
        }
        building: set[str] = set()
        built: dict[str, XRenderedTweet] = {}

        def build(tweet_id: str) -> XRenderedTweet:
            """
            构造单条推文并处理递归引用。

            Args:
                tweet_id (str): 推文 ID。

            Returns:
                XRenderedTweet: 构造后的推文。

            Raises:
                AssertionError: 当引用链结构异常时抛出。
            """
            if tweet_id in built:
                return built[tweet_id]
            if tweet_id in building:
                return _unknown_tweet(tweet_id)
            raw = raw_tweets.get(tweet_id)
            if raw is None:
                return _unknown_tweet(tweet_id)

            building.add(tweet_id)
            author_id = str(raw.get("author_id") or "")
            author = users.get(author_id) or XRenderedUser(
                user_id=author_id or "unknown",
                name="Unknown",
                username="unknown",
            )
            media_items = [
                media[str(key)]
                for key in _attachment_media_keys(raw)
                if str(key) in media
            ]
            references = [
                XRenderedReference(
                    reference_type=str(reference.get("type") or "unknown"),
                    tweet=build(str(reference.get("id"))),
                )
                for reference in _as_mapping_list(raw.get("referenced_tweets"))
                if reference.get("id") is not None
            ]
            tweet = XRenderedTweet(
                tweet_id=tweet_id,
                text=clean_tweet_text(raw),
                author=author,
                created_at=_optional_str(raw.get("created_at")),
                media=media_items,
                references=references,
                raw=dict(raw),
            )
            built[tweet_id] = tweet
            building.remove(tweet_id)
            return tweet

        return [
            build(str(tweet["id"]))
            for tweet in _as_mapping_list(payload.get("data"))
            if tweet.get("id") is not None
        ]

    def _parse_user(self, raw: Mapping[str, Any]) -> XRenderedUser:
        """
        解析 X API 用户对象。

        Args:
            raw (Mapping[str, Any]): 用户对象。

        Returns:
            XRenderedUser: 可渲染用户。

        Raises:
            None: 本方法不主动抛出异常。
        """
        return XRenderedUser(
            user_id=str(raw.get("id") or ""),
            name=str(raw.get("name") or raw.get("username") or "Unknown"),
            username=str(raw.get("username") or "unknown"),
            profile_image_url=_optional_str(raw.get("profile_image_url")),
        )

    def _parse_media(self, raw: Mapping[str, Any]) -> XRenderedMedia:
        """
        解析 X API 媒体对象。

        Args:
            raw (Mapping[str, Any]): 媒体对象。

        Returns:
            XRenderedMedia: 可渲染媒体。

        Raises:
            AssertionError: 当媒体尺寸字段非法时抛出。
        """
        return XRenderedMedia(
            media_key=str(raw.get("media_key") or ""),
            media_type=str(raw.get("type") or "photo"),
            url=_optional_str(raw.get("url")),
            preview_image_url=_optional_str(raw.get("preview_image_url")),
            width=_optional_int(raw.get("width")),
            height=_optional_int(raw.get("height")),
            alt_text=_optional_str(raw.get("alt_text")),
        )


class BrowserTweetRenderer:
    """
    使用无头 Chromium 将推文 HTML 截图为 PNG。
    """

    def __init__(self, config: Optional[BrowserRenderConfig] = None) -> None:
        """
        初始化渲染器。

        Args:
            config (Optional[BrowserRenderConfig]): 渲染配置。

        Returns:
            None: 构造函数无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self.config = config or BrowserRenderConfig()

    def render_to_png_bytes(self, tweet: XRenderedTweet) -> bytes:
        """
        将推文渲染为 PNG 字节。

        Args:
            tweet (XRenderedTweet): 需要渲染的推文。

        Returns:
            bytes: PNG 图片字节。

        Raises:
            RuntimeError: 当 playwright 未安装或浏览器截图失败时抛出。
            AssertionError: 当截图结果为空时抛出。
        """
        try:
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError("生成 X 推文图片需要安装 playwright。") from exc

        html_doc = render_tweet_html(tweet, self.config)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    viewport={"width": self.config.width, "height": 1600},
                    device_scale_factor=self.config.device_scale_factor,
                )
                page.set_content(
                    html_doc,
                    wait_until="networkidle",
                    timeout=self.config.timeout_ms,
                )
                page.evaluate(_IMAGE_SETTLE_SCRIPT)
                data = page.locator("#capture").screenshot(omit_background=False)
            finally:
                browser.close()
        assert isinstance(data, bytes) and data, "推文截图结果不能为空"
        return data


def merge_api_payloads(
    base: Mapping[str, Any], *extras: Mapping[str, Any]
) -> dict[str, Any]:
    """
    合并 X API 响应并保留第一个顶层 data。

    Args:
        base (Mapping[str, Any]): 基础响应。
        *extras (Mapping[str, Any]): 额外响应。

    Returns:
        dict[str, Any]: 合并后的响应。

    Raises:
        AssertionError: 当 includes 结构异常时抛出。
    """
    merged = copy.deepcopy(dict(base))
    includes = merged.setdefault("includes", {})
    assert isinstance(includes, dict), "X API includes 必须为对象"
    for extra in extras:
        extra_includes = extra.get("includes") or {}
        assert isinstance(extra_includes, Mapping), "额外响应 includes 必须为对象"
        _merge_include_list(includes, "users", extra_includes.get("users"), "id")
        _merge_include_list(includes, "tweets", extra_includes.get("tweets"), "id")
        _merge_include_list(includes, "media", extra_includes.get("media"), "media_key")
        _merge_include_list(includes, "tweets", _as_mapping_list(extra.get("data")), "id")
    return merged


def collect_referenced_tweet_ids(payload: Mapping[str, Any]) -> list[str]:
    """
    收集响应中出现的引用推文 ID。

    Args:
        payload (Mapping[str, Any]): X API v2 响应。

    Returns:
        list[str]: 按出现顺序去重后的引用推文 ID。

    Raises:
        AssertionError: 当引用结构异常时抛出。
    """
    ids: list[str] = []
    seen: set[str] = set()
    for tweet in _all_raw_tweets(payload):
        for reference in _as_mapping_list(tweet.get("referenced_tweets")):
            tweet_id = str(reference.get("id") or "")
            if tweet_id and tweet_id not in seen:
                seen.add(tweet_id)
                ids.append(tweet_id)
    return ids


def clean_tweet_text(raw_tweet: Mapping[str, Any]) -> str:
    """
    清洗推文正文并替换 URL entity。

    Args:
        raw_tweet (Mapping[str, Any]): X API 原始推文对象。

    Returns:
        str: 清洗后的正文。

    Raises:
        AssertionError: 当 entities.urls 类型异常时抛出。
    """
    note_tweet = raw_tweet.get("note_tweet")
    if isinstance(note_tweet, Mapping) and note_tweet.get("text"):
        text = str(note_tweet["text"])
    else:
        text = str(raw_tweet.get("text") or "")
    text = html.unescape(text)
    quoted_ids = _referenced_tweet_ids(raw_tweet, "quoted")
    entities = raw_tweet.get("entities") or {}
    assert isinstance(entities, Mapping), "X API entities 必须为对象"
    for entity in _as_mapping_list(entities.get("urls")):
        url = entity.get("url")
        if not url:
            continue
        replacement = _url_entity_replacement(entity, quoted_ids)
        text = text.replace(str(url), replacement)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_tweet_html(
    tweet: XRenderedTweet, config: Optional[BrowserRenderConfig] = None
) -> str:
    """
    将推文渲染为完整 HTML 文档。

    Args:
        tweet (XRenderedTweet): 需要渲染的推文。
        config (Optional[BrowserRenderConfig]): 渲染配置。

    Returns:
        str: HTML 文档。

    Raises:
        None: 本函数不主动抛出异常。
    """
    resolved_config = config or BrowserRenderConfig()
    body = (
        _render_repost(tweet, resolved_config)
        if tweet.repost
        else _render_tweet(tweet, compact=False, config=resolved_config)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      color-scheme: light;
      --black: #0f1419;
      --gray: #536471;
      --border: #cfd9de;
      --soft: #eff3f4;
      --blue: #1d9bf0;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      background: #fff;
      color: var(--black);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
        Arial, "Noto Sans CJK JP", "Noto Sans CJK SC", "Apple Color Emoji",
        "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif;
      font-synthesis: none;
      text-rendering: optimizeLegibility;
      -webkit-font-smoothing: antialiased;
    }}
    #capture {{
      width: {resolved_config.width}px;
      padding: {resolved_config.padding}px;
      background: #fff;
    }}
    .repost-indicator {{
      margin: 0 0 8px {resolved_config.avatar_size + 14}px;
      color: var(--gray);
      font-size: 18px;
      line-height: 22px;
      font-weight: 600;
    }}
    .tweet {{
      display: grid;
      grid-template-columns: {resolved_config.avatar_size}px minmax(0, 1fr);
      column-gap: 14px;
      align-items: start;
    }}
    .tweet.compact {{
      grid-template-columns: 36px minmax(0, 1fr);
      column-gap: 10px;
    }}
    .avatar {{
      width: {resolved_config.avatar_size}px;
      height: {resolved_config.avatar_size}px;
      border-radius: 999px;
      object-fit: cover;
      background: var(--soft);
      display: block;
    }}
    .compact .avatar {{
      width: 36px;
      height: 36px;
    }}
    .avatar-fallback {{
      display: grid;
      place-items: center;
      color: var(--gray);
      font-weight: 700;
    }}
    .header {{
      display: flex;
      min-width: 0;
      align-items: baseline;
      gap: 6px;
      line-height: 26px;
      white-space: nowrap;
    }}
    .compact .header {{ line-height: 22px; }}
    .name {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 22px;
      font-weight: 700;
    }}
    .compact .name {{ font-size: 17px; }}
    .handle {{
      flex: none;
      color: var(--gray);
      font-size: 20px;
      font-weight: 400;
    }}
    .timestamp {{
      flex: none;
      color: var(--gray);
      font-size: 20px;
      font-weight: 400;
    }}
    .compact .handle,
    .compact .timestamp {{
      font-size: 16px;
    }}
    .text {{
      margin-top: 6px;
      font-size: 24px;
      line-height: 1.36;
      font-weight: 400;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .hashtag,
    .mention {{
      color: var(--blue);
    }}
    .text.translated-text,
    .translation {{
      font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei",
        "Noto Sans CJK JP", -apple-system, BlinkMacSystemFont, "Segoe UI",
        "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji",
        sans-serif;
    }}
    .translation {{
      margin-top: 14px;
      padding-top: 12px;
      position: relative;
      font-size: 24px;
      line-height: 1.36;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .translation::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 90%;
      height: 1px;
      background: var(--border);
    }}
    .compact .text {{
      margin-top: 4px;
      font-size: 18px;
      line-height: 1.35;
    }}
    .compact .translation {{
      margin-top: 10px;
      padding-top: 10px;
      font-size: 18px;
      line-height: 1.35;
    }}
    .media-stack {{
      margin-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .media {{
      width: 90%;
      height: auto;
      display: block;
      border-radius: 16px;
      background: #f7f9f9;
      border: 1px solid var(--soft);
    }}
    .media-placeholder {{
      width: 90%;
      aspect-ratio: var(--ratio, 1.6);
      border-radius: 16px;
      background: #f7f9f9;
      border: 1px solid var(--soft);
    }}
    .quote {{
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}
    .agent-footer {{
      margin-top: 18px;
      color: #b8c1c9;
      font-size: 14px;
      line-height: 18px;
      text-align: center;
    }}
  </style>
</head>
<body>
  <main id="capture">{body}<div class="agent-footer">本推文由 筱泽广Agent 提供</div></main>
</body>
</html>"""


def _url_entity_replacement(entity: Mapping[str, Any], quoted_ids: set[str]) -> str:
    """
    返回 URL entity 在正文中的替换值。

    Args:
        entity (Mapping[str, Any]): URL entity。
        quoted_ids (set[str]): 引用推文 ID 集合。

    Returns:
        str: 替换文本。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if entity.get("media_key"):
        return ""
    if _is_embedded_quote_url(entity, quoted_ids):
        return ""
    return str(entity.get("display_url") or entity.get("expanded_url") or "")


def _is_embedded_quote_url(entity: Mapping[str, Any], quoted_ids: set[str]) -> bool:
    """
    判断 URL entity 是否为嵌入引用推文链接。

    Args:
        entity (Mapping[str, Any]): URL entity。
        quoted_ids (set[str]): 引用推文 ID 集合。

    Returns:
        bool: 是引用推文链接时为 True。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if not quoted_ids:
        return False
    candidates = [
        str(entity.get("expanded_url") or ""),
        str(entity.get("unwound_url") or ""),
        str(entity.get("display_url") or ""),
    ]
    return any(
        f"/status/{tweet_id}" in candidate or f"/statuses/{tweet_id}" in candidate
        for tweet_id in quoted_ids
        for candidate in candidates
    )


def _referenced_tweet_ids(
    raw_tweet: Mapping[str, Any], reference_type: str
) -> set[str]:
    """
    返回指定类型的引用推文 ID。

    Args:
        raw_tweet (Mapping[str, Any]): 原始推文对象。
        reference_type (str): 引用类型。

    Returns:
        set[str]: 引用推文 ID 集合。

    Raises:
        AssertionError: 当 referenced_tweets 类型异常时抛出。
    """
    return {
        str(reference["id"])
        for reference in _as_mapping_list(raw_tweet.get("referenced_tweets"))
        if reference.get("type") == reference_type and reference.get("id") is not None
    }


def _merge_include_list(
    includes: dict[str, Any], key: str, items: object, id_key: str
) -> None:
    """
    将 includes 中的列表按 ID 去重合并。

    Args:
        includes (dict[str, Any]): includes 对象。
        key (str): includes 下的列表键。
        items (object): 需要合并的候选列表。
        id_key (str): 去重 ID 字段。

    Returns:
        None: 原地修改 includes。

    Raises:
        AssertionError: 当目标列表类型异常时抛出。
    """
    target = includes.setdefault(key, [])
    assert isinstance(target, list), f"X API includes.{key} 必须为列表"
    seen = {str(item.get(id_key)) for item in target if isinstance(item, Mapping)}
    for item in _as_mapping_list(items):
        item_id = item.get(id_key)
        if item_id is None:
            continue
        normalized = str(item_id)
        if normalized not in seen:
            target.append(copy.deepcopy(dict(item)))
            seen.add(normalized)


def _attachment_media_keys(raw_tweet: Mapping[str, Any]) -> list[str]:
    """
    从原始推文对象中读取媒体键。

    Args:
        raw_tweet (Mapping[str, Any]): 原始推文对象。

    Returns:
        list[str]: 媒体键列表。

    Raises:
        AssertionError: 当 attachments 不是对象或 media_keys 不是列表时抛出。
    """
    attachments = raw_tweet.get("attachments") or {}
    assert isinstance(attachments, Mapping), "X API attachments 必须为对象"
    return _as_string_list(attachments.get("media_keys"))


def _all_raw_tweets(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """
    返回 data 与 includes.tweets 中的全部原始推文。

    Args:
        payload (Mapping[str, Any]): X API 响应。

    Returns:
        list[Mapping[str, Any]]: 原始推文列表。

    Raises:
        AssertionError: 当 includes 结构异常时抛出。
    """
    tweets = _as_mapping_list(payload.get("data"))
    includes = payload.get("includes") or {}
    assert isinstance(includes, Mapping), "X API includes 必须为对象"
    tweets.extend(_as_mapping_list(includes.get("tweets")))
    return tweets


def _as_mapping_list(value: object) -> list[Mapping[str, Any]]:
    """
    将对象标准化为 Mapping 列表。

    Args:
        value (object): 候选对象。

    Returns:
        list[Mapping[str, Any]]: Mapping 列表。

    Raises:
        AssertionError: 当 value 不是可接受类型时抛出。
    """
    if value is None:
        return []
    if isinstance(value, list):
        assert all(isinstance(item, Mapping) for item in value), "列表元素必须为对象"
        return value
    if isinstance(value, Mapping):
        return [value]
    raise AssertionError("字段必须为对象或对象列表")


def _as_string_list(value: object) -> list[str]:
    """
    将对象标准化为字符串列表。

    Args:
        value (object): 候选对象。

    Returns:
        list[str]: 字符串列表。

    Raises:
        AssertionError: 当 value 不是列表时抛出。
    """
    if value is None:
        return []
    assert isinstance(value, list), "字段必须为列表"
    return [str(item) for item in value]


def _optional_str(value: object) -> Optional[str]:
    """
    将可选对象转换为字符串。

    Args:
        value (object): 候选值。

    Returns:
        Optional[str]: 字符串或 None。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> Optional[int]:
    """
    将可选对象转换为整数。

    Args:
        value (object): 候选值。

    Returns:
        Optional[int]: 整数或 None。

    Raises:
        AssertionError: 当值不是整数或整数字符串时抛出。
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = str(value).strip()
    assert normalized.isdigit(), "整数字段必须为数字"
    return int(normalized)


def _unknown_tweet(tweet_id: str) -> XRenderedTweet:
    """
    构造未知占位推文。

    Args:
        tweet_id (str): 推文 ID。

    Returns:
        XRenderedTweet: 未知占位推文。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return XRenderedTweet(
        tweet_id=tweet_id,
        text="",
        author=XRenderedUser(user_id="unknown", name="Unknown", username="unknown"),
    )


def _render_repost(tweet: XRenderedTweet, config: BrowserRenderConfig) -> str:
    """
    渲染转发推文 HTML。

    Args:
        tweet (XRenderedTweet): 推文。
        config (BrowserRenderConfig): 渲染配置。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    repost = tweet.repost
    if repost is None:
        return _render_tweet(tweet, compact=False, config=config)
    return (
        f'<div class="repost-indicator">{_escape(tweet.author.name)} reposted</div>'
        f"{_render_tweet(repost.tweet, compact=False, config=config)}"
    )


def _render_tweet(
    tweet: XRenderedTweet, *, compact: bool, config: BrowserRenderConfig
) -> str:
    """
    渲染单条推文 HTML。

    Args:
        tweet (XRenderedTweet): 推文。
        compact (bool): 是否为紧凑引用样式。
        config (BrowserRenderConfig): 渲染配置。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    compact_class = " compact" if compact else ""
    quote = (
        f'<div class="quote">{_render_tweet(tweet.quote.tweet, compact=True, config=config)}</div>'
        if tweet.quote and not compact
        else ""
    )
    return f"""
    <article class="tweet{compact_class}">
      {_render_avatar(tweet.author)}
      <div class="body">
        {_render_header(tweet, config)}
        {_render_text(tweet.text, translated=tweet.is_translated_text)}
        {_render_translation(tweet.translation_text)}
        {_render_media_stack(tweet.media)}
        {quote}
      </div>
    </article>
    """


def _render_header(tweet: XRenderedTweet, config: BrowserRenderConfig) -> str:
    """
    渲染推文头部 HTML。

    Args:
        tweet (XRenderedTweet): 推文。
        config (BrowserRenderConfig): 渲染配置。

    Returns:
        str: HTML 片段。

    Raises:
        ValueError: 当推文时间或时区非法时抛出。
    """
    user = tweet.author
    timestamp = _format_created_at(tweet.created_at, config.timezone)
    time_html = f'<span class="timestamp">· {_escape(timestamp)}</span>' if timestamp else ""
    return (
        '<div class="header">'
        f'<span class="name">{_escape(user.name)}</span>'
        f'<span class="handle">{_escape(user.handle)}</span>'
        f"{time_html}"
        "</div>"
    )


def _render_avatar(user: XRenderedUser) -> str:
    """
    渲染头像 HTML。

    Args:
        user (XRenderedUser): 用户。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    url = user.high_res_profile_image_url
    if not url:
        initial = _escape((user.name or user.username or "?")[:1].upper())
        return f'<div class="avatar avatar-fallback">{initial}</div>'
    return f'<img class="avatar" src="{_attr(url)}" alt="">'


def _render_text(text: str, translated: bool = False) -> str:
    """
    渲染正文 HTML。

    Args:
        text (str): 推文正文。
        translated (bool): 是否使用简体中文译文字体栈。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if not text:
        return ""
    class_name = "text translated-text" if translated else "text"
    return f'<div class="{class_name}">{_render_text_entities(text)}</div>'


def _render_translation(text: Optional[str]) -> str:
    """
    渲染独立译文容器。

    Args:
        text (Optional[str]): 简体中文译文。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if not text:
        return ""
    return (
        '<section class="translation" lang="zh-CN">'
        f'<div class="translation-text">{_render_text_entities(text)}</div>'
        "</section>"
    )


def _render_media_stack(media: list[XRenderedMedia]) -> str:
    """
    渲染媒体栈 HTML。

    Args:
        media (list[XRenderedMedia]): 媒体集合。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    items = [
        _render_media(item)
        for item in media
        if item.media_type in {"photo", "video", "animated_gif"}
    ]
    if not items:
        return ""
    return '<div class="media-stack">' + "".join(items) + "</div>"


def _render_media(item: XRenderedMedia) -> str:
    """
    渲染单个媒体 HTML。

    Args:
        item (XRenderedMedia): 媒体对象。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    url = item.best_url
    if url and _url_scheme(url) not in {"", "placeholder"}:
        return f'<img class="media" src="{_attr(url)}" alt="{_attr(item.alt_text or "")}">'
    ratio = _media_ratio(item)
    return f'<div class="media-placeholder" style="--ratio:{ratio:.4f}"></div>'


def _media_ratio(item: XRenderedMedia) -> float:
    """
    计算媒体宽高比。

    Args:
        item (XRenderedMedia): 媒体对象。

    Returns:
        float: 限制在合理范围内的宽高比。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if item.width and item.height and item.width > 0 and item.height > 0:
        return max(0.4, min(2.5, item.width / item.height))
    return 1.6


def _format_created_at(created_at: Optional[str], timezone_name: str) -> Optional[str]:
    """
    格式化推文创建时间。

    Args:
        created_at (Optional[str]): X API ISO 时间。
        timezone_name (str): 目标时区。

    Returns:
        Optional[str]: 本地化后的时间文本。

    Raises:
        ValueError: 当时间或时区非法时抛出。
    """
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"推文时间格式非法：{created_at}") from exc
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"未知时区：{timezone_name}") from exc
    local = created.astimezone(tz)
    return f"{local.year}年{local.month}月{local.day}日 {local.hour:02d}:{local.minute:02d}"


def _render_text_entities(text: str) -> str:
    """
    渲染正文中的 hashtag 与 mention。

    Args:
        text (str): 推文正文。

    Returns:
        str: HTML 片段。

    Raises:
        None: 本函数不主动抛出异常。
    """
    parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "#" and _is_entity_start(text, index, _is_hashtag_char):
            end = _find_entity_end(text, index, _is_hashtag_char)
            if end > index + 1:
                tag = _escape(text[index:end])
                parts.append(f'<span class="hashtag">{tag}</span>')
                index = end
                continue
        if text[index] == "@" and _is_entity_start(text, index, _is_mention_char):
            end = _find_mention_end(text, index)
            if end > index + 1:
                mention = _escape(text[index:end])
                parts.append(f'<span class="mention">{mention}</span>')
                index = end
                continue
        parts.append(_escape(text[index]))
        index += 1
    return "".join(parts)


def _is_entity_start(
    text: str, index: int, char_checker: Callable[[str], bool]
) -> bool:
    """
    判断当前位置是否可以开始一个文本 entity。

    Args:
        text (str): 推文正文。
        index (int): 候选起始位置。
        char_checker (Callable[[str], bool]): entity 字符判断函数。

    Returns:
        bool: 可以作为 entity 起点时为 True。

    Raises:
        AssertionError: 当 index 越界时抛出。
    """
    assert 0 <= index < len(text), "index 必须位于文本范围内"
    if index == 0:
        return True
    return not char_checker(text[index - 1])


def _find_entity_end(
    text: str, index: int, char_checker: Callable[[str], bool]
) -> int:
    """
    查找文本 entity 的结束位置。

    Args:
        text (str): 推文正文。
        index (int): entity 起始符位置。
        char_checker (Callable[[str], bool]): entity 字符判断函数。

    Returns:
        int: entity 结束位置。

    Raises:
        AssertionError: 当 index 越界时抛出。
    """
    assert 0 <= index < len(text), "index 必须位于文本范围内"
    end = index + 1
    while end < len(text) and char_checker(text[end]):
        end += 1
    return end


def _find_mention_end(text: str, index: int) -> int:
    """
    查找 X mention 的结束位置。

    Args:
        text (str): 推文正文。
        index (int): @ 符号位置。

    Returns:
        int: mention 结束位置。

    Raises:
        AssertionError: 当 index 越界时抛出。
    """
    end = _find_entity_end(text, index, _is_mention_char)
    if end - index > 16:
        return index + 16
    return end


def _is_mention_char(char: str) -> bool:
    """
    判断字符是否属于 X 用户名。

    Args:
        char (str): 单个字符。

    Returns:
        bool: 属于用户名时为 True。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if char == "_":
        return True
    if "0" <= char <= "9":
        return True
    if "A" <= char <= "Z":
        return True
    if "a" <= char <= "z":
        return True
    return False


def _is_hashtag_char(char: str) -> bool:
    """
    判断字符是否属于 hashtag。

    Args:
        char (str): 单个字符。

    Returns:
        bool: 属于 hashtag 时为 True。

    Raises:
        None: 本函数不主动抛出异常。
    """
    if char in {"_", "\u30fc", "\uff70"}:
        return True
    category = unicodedata.category(char)
    return category[0] in {"L", "N"} or category in {"Mn", "Mc"}


def _escape(value: str) -> str:
    """
    转义 HTML 文本。

    Args:
        value (str): 原始文本。

    Returns:
        str: 转义后的文本。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return html.escape(value or "", quote=False)


def _attr(value: str) -> str:
    """
    转义 HTML 属性。

    Args:
        value (str): 原始属性值。

    Returns:
        str: 转义后的属性值。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return html.escape(value or "", quote=True)


def _url_scheme(value: str) -> str:
    """
    解析 URL scheme。

    Args:
        value (str): URL 字符串。

    Returns:
        str: URL scheme。

    Raises:
        None: 本函数不主动抛出异常。
    """
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*):", value)
    return match.group(1).lower() if match else ""


_IMAGE_SETTLE_SCRIPT = """async () => {
    if (document.fonts && document.fonts.ready) {
        await document.fonts.ready;
    }
    const images = Array.from(document.images);
    await Promise.all(images.map((img) => {
        if (img.complete) return Promise.resolve();
        return new Promise((resolve) => {
            img.addEventListener('load', resolve, { once: true });
            img.addEventListener('error', resolve, { once: true });
        });
    }));
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}"""
