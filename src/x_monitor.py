"""
X 固定账号推文监控模块。

基于 X API v2 提供用户时间线拉取、后台轮询与 JSON 持久化能力。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Optional, Sequence
from urllib.parse import urlparse

import requests

USER_LOOKUP_URL = "https://api.x.com/2/users/by/username/{username}"
USER_POSTS_URL = "https://api.x.com/2/users/{user_id}/tweets"
TWEET_LOOKUP_URL = "https://api.x.com/2/tweets"
POST_URL = "https://x.com/{username}/status/{post_id}"
DEFAULT_LIMIT = 5
MAX_WATCH_TASKS = 20
DEFAULT_STORE_PATH = ".x_monitor.json"
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,15}$")
X_LINK_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com"}
NEW_POST_NOTICE_ENV = "X_MONITOR_NEW_POST_NOTICE"
SOURCE_ENV = "X_MONITOR_SOURCE"
BROWSER_SOURCE_ALIASES = {"browser", "playwright", "headless"}


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


def _normalize_username(username: str) -> str:
    """
    标准化 X 用户名。

    Args:
        username (str): 用户输入的 X 用户名，可带 `@` 前缀。

    Returns:
        str: 去除 `@` 后的用户名。

    Raises:
        AssertionError: 当用户名为空或格式不合法时抛出。
    """
    cleaned = username.strip()
    if cleaned.startswith("@"):
        cleaned = cleaned[1:].strip()
    assert cleaned, "用户名不能为空"
    assert USERNAME_PATTERN.fullmatch(cleaned), "用户名格式不合法"
    return cleaned


def _format_created_at(value: object) -> str:
    """
    将 X API 的 ISO 时间转换为本地时间字符串。

    Args:
        value (object): X API 返回的 created_at 字段。

    Returns:
        str: 本地时间字符串，缺失时返回空字符串。

    Raises:
        ValueError: 当时间字段格式非法时抛出。
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
    return dt.strftime("%m-%d %H:%M")


def _post_id_key(post_id: str) -> int:
    """
    将 X 推文 ID 转换为可比较的整数。

    Args:
        post_id (str): X 推文 ID。

    Returns:
        int: 推文 ID 对应的整数值。

    Raises:
        AssertionError: 当推文 ID 为空时抛出。
        ValueError: 当推文 ID 不是数字字符串时抛出。
    """
    normalized = post_id.strip()
    assert normalized, "post_id 不能为空"
    return int(normalized)


def _tweet_payload_params() -> dict[str, str]:
    """
    返回渲染推文图片所需的 X API 字段参数。

    Returns:
        dict[str, str]: 可传给 X API v2 tweets 相关接口的字段参数。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return {
        "tweet.fields": (
            "attachments,author_id,created_at,entities,note_tweet,public_metrics,"
            "referenced_tweets"
        ),
        "expansions": (
            "author_id,attachments.media_keys,referenced_tweets.id,"
            "referenced_tweets.id.author_id,"
            "referenced_tweets.id.attachments.media_keys"
        ),
        "user.fields": "id,name,username,profile_image_url",
        "media.fields": "media_key,type,url,preview_image_url,width,height,alt_text",
    }


def _extract_api_error(payload: object) -> str:
    """
    提取 X API 错误摘要。

    Args:
        payload (object): X API 返回的 JSON 对象。

    Returns:
        str: 适合展示的错误摘要。

    Raises:
        None: 本函数不抛出异常。
    """
    if not isinstance(payload, dict):
        return ""
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return ""
    first = errors[0]
    if not isinstance(first, dict):
        return str(first)
    detail = str(first.get("detail") or "").strip()
    title = str(first.get("title") or "").strip()
    return detail or title


@dataclass(frozen=True)
class XTweetLink:
    """
    X 推文链接解析结果。

    Attributes:
        username (str): 链接路径中的用户名，缺失时为空字符串。
        tweet_id (str): 推文 ID。
    """

    username: str
    tweet_id: str


def _parse_tweet_link(url: str) -> XTweetLink:
    """
    从 X/Twitter 推文链接中解析用户名与推文 ID。

    Args:
        url (str): 用户输入的推文链接。

    Returns:
        XTweetLink: 解析后的链接信息。

    Raises:
        AssertionError: 当链接为空、域名不支持或缺少合法推文 ID 时抛出。
    """
    raw = url.strip()
    assert raw, "推文链接不能为空"
    candidate = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    assert host in X_LINK_HOSTS, "仅支持 x.com/twitter.com 推文链接"

    parts = [part for part in parsed.path.split("/") if part]
    status_index = next(
        (idx for idx, part in enumerate(parts) if part in {"status", "statuses"}),
        -1,
    )
    assert status_index >= 0, "链接中缺少 status 路径"
    assert status_index + 1 < len(parts), "链接中缺少推文 ID"
    tweet_id = parts[status_index + 1].strip()
    assert tweet_id.isdigit(), "推文 ID 必须为数字"
    raw_username = parts[status_index - 1].strip() if status_index > 0 else ""
    username = raw_username if USERNAME_PATTERN.fullmatch(raw_username) else ""
    return XTweetLink(username=username, tweet_id=tweet_id)


def _collect_post_image_urls(
    post: dict[str, object],
    media_by_key: dict[str, dict[str, object]],
    limit: int = 4,
) -> tuple[str, ...]:
    """
    从推文与媒体 expansion 中抽取图片 URL。

    Args:
        post (dict[str, object]): 单条推文对象。
        media_by_key (dict[str, dict[str, object]]): media_key 到媒体对象的映射。
        limit (int): 最多返回的图片数量。

    Returns:
        tuple[str, ...]: 去重后的图片或视频预览图 URL。

    Raises:
        AssertionError: 当 limit 非正数或 API 字段类型异常时抛出。
    """
    assert limit > 0, "limit 必须大于 0"
    attachments = post.get("attachments") or {}
    if not attachments:
        return ()
    assert isinstance(attachments, dict), "attachments 必须为对象"
    media_keys = attachments.get("media_keys") or []
    assert isinstance(media_keys, list), "media_keys 必须为列表"
    urls: list[str] = []
    for raw_key in media_keys:
        key = str(raw_key or "").strip()
        if not key:
            continue
        media = media_by_key.get(key)
        if media is None:
            continue
        media_type = str(media.get("type") or "").strip()
        url = ""
        if media_type == "photo":
            url = str(media.get("url") or "").strip()
        elif media_type in {"video", "animated_gif"}:
            url = str(media.get("preview_image_url") or "").strip()
        else:
            url = str(
                media.get("url") or media.get("preview_image_url") or ""
            ).strip()
        if url.startswith("http") and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return tuple(urls)


@dataclass(frozen=True)
class XUserProfile:
    """
    X 用户资料。

    Attributes:
        user_id (str): X 用户 ID。
        username (str): X 用户名。
        name (str): 显示名称。
    """

    user_id: str
    username: str
    name: str


@dataclass(frozen=True)
class XPostResult:
    """
    X 推文结果。

    Attributes:
        username (str): 推文所属 X 用户名。
        post_id (str): 推文 ID。
        text (str): 推文正文。
        created_label (str): 本地时间字符串。
        url (str): 推文链接。
        image_urls (tuple[str, ...]): 推文关联图片 URL。
        source_payload (Optional[dict[str, object]]): 生成图片所需的原始 X API 响应。
        display_name (str): 推文作者显示名。
    """

    username: str
    post_id: str
    text: str
    created_label: str
    url: str
    image_urls: tuple[str, ...] = ()
    source_payload: Optional[dict[str, object]] = None
    display_name: str = ""

    def to_line(self, prefix: str = "") -> str:
        """
        格式化为单行文本。

        Args:
            prefix (str): 行前缀。

        Returns:
            str: 单行展示文本。

        Raises:
            None: 本方法不抛出异常。
        """
        text = " ".join((self.text or "(无正文)").split())
        parts = [prefix.strip(), f"@{self.username}", text]
        if self.created_label:
            parts.append(self.created_label)
        if self.url:
            parts.append(self.url)
        return " | ".join(part for part in parts if part)


class XAPIClient:
    """
    X API v2 客户端。
    """

    def __init__(self, bearer_token: Optional[str] = None) -> None:
        """
        初始化 X API 客户端。

        Args:
            bearer_token (Optional[str]): X API Bearer Token。为空时读取环境变量。

        Raises:
            AssertionError: 当 Bearer Token 未提供时抛出。
        """
        token = bearer_token or os.environ.get("X_BEARER_TOKEN", "")
        token = token.strip()
        assert token, "缺少 X_BEARER_TOKEN 环境变量"
        self._bearer_token = token

    def get_user_profile(self, username: str) -> XUserProfile:
        """
        按用户名获取 X 用户资料。

        Args:
            username (str): X 用户名。

        Returns:
            XUserProfile: 用户资料。

        Raises:
            AssertionError: 当响应缺少必要字段时抛出。
            RuntimeError: 当网络或 API 调用失败时抛出。
            ValueError: 当 API 返回非 JSON 数据时抛出。
        """
        normalized = _normalize_username(username)
        payload = self._request_json(
            USER_LOOKUP_URL.format(username=normalized),
            params={"user.fields": "id,name,username"},
        )
        data = payload.get("data")
        assert isinstance(data, dict), "X API 用户响应缺少 data 对象"
        user_id = str(data.get("id") or "").strip()
        api_username = str(data.get("username") or "").strip()
        name = str(data.get("name") or "").strip()
        assert user_id, "X API 用户响应缺少 id"
        assert api_username, "X API 用户响应缺少 username"
        return XUserProfile(user_id=user_id, username=api_username, name=name)

    def get_user_posts(
        self,
        user_id: str,
        max_results: int = DEFAULT_LIMIT,
        since_id: Optional[str] = None,
    ) -> dict[str, object]:
        """
        获取指定用户的最新推文。

        Args:
            user_id (str): X 用户 ID。
            max_results (int): 本次最多返回的推文数量，范围 5 到 100。
            since_id (Optional[str]): 仅返回该 ID 之后的新推文。

        Returns:
            dict[str, object]: X API 原始 JSON 响应。

        Raises:
            AssertionError: 当参数非法时抛出。
            RuntimeError: 当网络或 API 调用失败时抛出。
            ValueError: 当 API 返回非 JSON 数据时抛出。
        """
        normalized_user_id = user_id.strip()
        assert normalized_user_id, "user_id 不能为空"
        assert 5 <= max_results <= 100, "max_results 必须在 5 到 100 之间"
        params = _tweet_payload_params()
        params["max_results"] = str(max_results)
        params["exclude"] = "retweets"
        if since_id is not None:
            normalized_since_id = since_id.strip()
            assert normalized_since_id, "since_id 不能为空字符串"
            params["since_id"] = normalized_since_id
        return self._request_json(
            USER_POSTS_URL.format(user_id=normalized_user_id),
            params=params,
        )

    def get_tweet_by_id(self, tweet_id: str) -> dict[str, object]:
        """
        按推文 ID 获取单条推文及渲染所需 expansions。

        Args:
            tweet_id (str): X 推文 ID。

        Returns:
            dict[str, object]: X API 原始 JSON 响应。

        Raises:
            AssertionError: 当推文 ID 为空或不是数字时抛出。
            RuntimeError: 当网络或 API 调用失败时抛出。
            ValueError: 当 API 返回非 JSON 数据时抛出。
        """
        normalized = tweet_id.strip()
        assert normalized, "tweet_id 不能为空"
        assert normalized.isdigit(), "tweet_id 必须为数字"
        params = _tweet_payload_params()
        params["ids"] = normalized
        return self._request_json(TWEET_LOOKUP_URL, params=params)

    def _request_json(
        self, url: str, params: Optional[dict[str, str]] = None
    ) -> dict[str, object]:
        """
        发起 GET 请求并解析 JSON 响应。

        Args:
            url (str): 请求 URL。
            params (Optional[dict[str, str]]): 查询参数。

        Returns:
            dict[str, object]: 解析后的 JSON 对象。

        Raises:
            AssertionError: 当响应 JSON 不是对象时抛出。
            RuntimeError: 当网络或 API 调用失败时抛出。
            ValueError: 当 API 返回非 JSON 数据时抛出。
        """
        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "Accept": "application/json",
            "User-Agent": "XMonitor/1.0",
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=20)
        except requests.RequestException as exc:
            raise RuntimeError("请求 X API 失败，请检查网络连接。") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("X API 返回非 JSON 数据。") from exc
        assert isinstance(payload, dict), "X API 返回格式异常，应为 JSON 对象"
        if response.status_code == 401:
            raise RuntimeError("X API 鉴权失败：请确认 X_BEARER_TOKEN 是否正确。")
        if response.status_code == 403:
            detail = _extract_api_error(payload)
            suffix = f"：{detail}" if detail else ""
            raise RuntimeError(f"X API 拒绝访问，请检查权限或订阅状态{suffix}")
        if response.status_code == 429:
            raise RuntimeError("X API 调用被限流，请稍后再试。")
        if response.status_code >= 400:
            detail = _extract_api_error(payload)
            suffix = f"：{detail}" if detail else ""
            raise RuntimeError(f"X API 调用失败，HTTP {response.status_code}{suffix}")
        return payload


class _XWatchTask:
    """
    内部 X 推文监控任务。
    """

    def __init__(
        self,
        username: str,
        x_user_id: str,
        interval: float,
        limit_per_cycle: int,
        group_id: Optional[int],
        user_id: Optional[int],
        fetcher: Callable[[Optional[str]], Sequence[XPostResult]],
        formatter: Callable[[Sequence[XPostResult], str], str],
        notify: Callable[[str], None],
        notify_media: Optional[
            Callable[[str, Sequence[XPostResult], str], None]
        ] = None,
    ) -> None:
        """
        初始化监控任务。

        Args:
            username (str): X 用户名。
            x_user_id (str): X 用户 ID。
            interval (float): 轮询间隔秒数。
            limit_per_cycle (int): 单轮推送上限。
            group_id (Optional[int]): 触发命令的群号。
            user_id (Optional[int]): 触发命令的用户号。
            fetcher (Callable[[Optional[str]], Sequence[XPostResult]]): 数据获取函数。
            formatter (Callable[[Sequence[XPostResult], str], str]): 文本格式化函数。
            notify (Callable[[str], None]): 纯文本通知回调。
            notify_media (Optional[Callable[[str, Sequence[XPostResult], str], None]]):
                携带图片的通知回调。

        Raises:
            AssertionError: 当任务参数非法时抛出。
        """
        assert username and username.strip(), "username 不能为空"
        assert x_user_id and x_user_id.strip(), "x_user_id 不能为空"
        assert interval > 0, "interval 必须大于 0"
        assert limit_per_cycle > 0, "limit_per_cycle 必须大于 0"
        self._username = username.strip()
        self._x_user_id = x_user_id.strip()
        self._interval = interval
        self._limit = limit_per_cycle
        self._group_id = group_id
        self._user_id = user_id
        self._fetcher = fetcher
        self._formatter = formatter
        self._notify = notify
        self._notify_media = notify_media
        self._stop_event = Event()
        self._thread = Thread(target=self._run, name="x-monitor", daemon=True)
        self._seen: set[str] = set()
        self._since_id: Optional[str] = None
        self._first_cycle = True

    def start(self) -> None:
        """
        启动后台线程。

        Returns:
            None: 无返回值。

        Raises:
            RuntimeError: 当线程启动失败时由 threading 抛出。
        """
        self._thread.start()

    def stop(self) -> None:
        """
        请求停止并等待线程退出。

        Returns:
            None: 无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._interval + 1)

    def alive(self) -> bool:
        """
        判断线程是否仍在运行。

        Returns:
            bool: True 表示线程存活。

        Raises:
            None: 本方法不抛出异常。
        """
        return self._thread.is_alive()

    def matches(self, username: str, group_id: Optional[int] = None) -> bool:
        """
        判断任务是否匹配指定用户名与群号。

        Args:
            username (str): X 用户名。
            group_id (Optional[int]): 目标群号。为空时只匹配用户名。

        Returns:
            bool: True 表示匹配。

        Raises:
            AssertionError: 当用户名格式非法时抛出。
        """
        normalized = _normalize_username(username).lower()
        if self._username.lower() != normalized:
            return False
        if group_id is None:
            return True
        return self._group_id == group_id

    def to_state(self) -> dict[str, object]:
        """
        导出可持久化的任务配置。

        Returns:
            dict[str, object]: 任务配置。

        Raises:
            None: 本方法不抛出异常。
        """
        return {
            "username": self._username,
            "x_user_id": self._x_user_id,
            "interval": self._interval,
            "limit_per_cycle": self._limit,
            "group_id": self._group_id,
            "user_id": self._user_id,
        }

    def _run(self) -> None:
        """
        轮询获取数据并推送新增推文。

        Returns:
            None: 无返回值。

        Raises:
            None: 后台异常会被记录并进入下一轮。
        """
        while not self._stop_event.is_set():
            try:
                results = list(self._fetcher(self._since_id))
                if self._first_cycle:
                    self._refresh_seen(results)
                    self._first_cycle = False
                else:
                    new_items = [
                        item for item in results if item.post_id not in self._seen
                    ]
                    if new_items:
                        self._handle_new_items(new_items)
                    self._refresh_seen(results)
            except Exception as err:
                print(f"[XMonitor] 监控轮询异常: {err}", flush=True)
            self._stop_event.wait(self._interval)

    def _refresh_seen(self, items: Sequence[XPostResult]) -> None:
        """
        更新已见推文集合与 since_id。

        Args:
            items (Sequence[XPostResult]): 本轮拉取的推文。

        Returns:
            None: 无返回值。

        Raises:
            ValueError: 当推文 ID 不是数字字符串时抛出。
        """
        if not items:
            return
        for item in items:
            self._seen.add(item.post_id)
        newest = max(items, key=lambda item: _post_id_key(item.post_id))
        if self._since_id is None:
            self._since_id = newest.post_id
        elif _post_id_key(newest.post_id) > _post_id_key(self._since_id):
            self._since_id = newest.post_id
        if len(self._seen) > 400:
            self._seen = {item.post_id for item in items}

    def _handle_new_items(self, items: Sequence[XPostResult]) -> None:
        """
        处理新推文并触发通知。

        Args:
            items (Sequence[XPostResult]): 新发现的推文列表。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当格式化结果为空时抛出。
        """
        subset = list(items)[: self._limit]
        message = self._formatter(subset, "NEW")
        assert message.strip(), "通知内容不能为空"
        if _env_flag(NEW_POST_NOTICE_ENV):
            self._notify(self._format_new_post_notice(subset))
        if self._notify_media is not None:
            self._notify_media(message, subset, "NEW")
        else:
            self._notify(message)
        print(
            f"[XMonitor] 发现新推文 {len(subset)}/{len(items)} 条，用户=@{self._username}",
            flush=True,
        )

    def _format_new_post_notice(self, items: Sequence[XPostResult]) -> str:
        """
        生成新推文批次的前置文字通知。

        Args:
            items (Sequence[XPostResult]): 本轮将要推送的推文列表。

        Returns:
            str: 前置文字通知。

        Raises:
            AssertionError: 当推文列表为空时抛出。
        """
        assert items, "推文列表不能为空"
        display_name = (items[0].display_name or self._username).strip()
        assert display_name, "用户显示名不能为空"
        return f"[NEW] {display_name} 更新了推文"


class XMonitorManager:
    """
    X 推文监控管理器。
    """

    def __init__(
        self,
        client: Optional[XAPIClient] = None,
        store_path: str = DEFAULT_STORE_PATH,
        browser_client: Optional[object] = None,
    ) -> None:
        """
        初始化管理器。

        Args:
            client (Optional[XAPIClient]): 可选的自定义 X API 客户端。
            store_path (str): 监控任务持久化 JSON 路径。
            browser_client (Optional[object]): 可选的浏览器模式客户端。

        Raises:
            None: 客户端按需初始化，因此构造函数不读取密钥。
        """
        self._client = client
        self._browser_client = browser_client
        self._client_lock = Lock()
        self._browser_client_lock = Lock()
        self._lock = Lock()
        self._watch_tasks: list[_XWatchTask] = []
        self._store_path = Path(store_path)

    def latest(self, username: str, limit: int = DEFAULT_LIMIT) -> list[XPostResult]:
        """
        获取指定账号最新推文。

        Args:
            username (str): X 用户名。
            limit (int): 返回数量上限。

        Returns:
            list[XPostResult]: 最新推文列表。

        Raises:
            AssertionError: 当参数非法时抛出。
            RuntimeError: 当 X API 调用失败时抛出。
        """
        assert limit > 0, "limit 必须大于 0"
        if self._use_browser_source():
            return self._get_browser_client().latest(username, limit=limit)
        profile = self._get_client().get_user_profile(username)
        page_size = max(DEFAULT_LIMIT, min(100, limit))
        return self._fetch_results(
            profile.username, profile.user_id, limit=page_size
        )[:limit]

    def fetch_link(self, url: str) -> XPostResult:
        """
        按 X/Twitter 推文链接获取单条推文结果。

        Args:
            url (str): X/Twitter 推文链接。

        Returns:
            XPostResult: 单条推文结果，包含图片渲染所需 source_payload。

        Raises:
            AssertionError: 当链接非法或响应中缺少目标推文时抛出。
            RuntimeError: 当 X API 调用失败时抛出。
            ValueError: 当 API 返回字段格式非法时抛出。
        """
        if self._use_browser_source():
            return self._get_browser_client().fetch_link(url)
        link = _parse_tweet_link(url)
        payload = self._get_client().get_tweet_by_id(link.tweet_id)
        results = self._parse_posts(link.username or "unknown", payload)
        for item in results:
            if item.post_id == link.tweet_id:
                return item
        raise AssertionError("未获取到目标推文")

    def list_watch_tasks(self) -> list[dict[str, object]]:
        """
        返回当前所有监控任务快照。

        Returns:
            list[dict[str, object]]: 每个任务的配置。

        Raises:
            None: 本方法不主动抛出异常。
        """
        with self._lock:
            self._prune_dead_watch_tasks()
            return [task.to_state() for task in self._watch_tasks]

    def start_watch(
        self,
        username: str,
        interval: float,
        limit_per_cycle: int = DEFAULT_LIMIT,
        group_id: Optional[int] = None,
        user_id: Optional[int] = None,
        notify: Optional[Callable[[str], None]] = None,
        notify_media: Optional[
            Callable[[str, Sequence[XPostResult], str], None]
        ] = None,
        persist: bool = True,
        x_user_id: Optional[str] = None,
    ) -> None:
        """
        启动账号推文监控。

        Args:
            username (str): X 用户名。
            interval (float): 轮询间隔秒数。
            limit_per_cycle (int): 单轮推送上限。
            group_id (Optional[int]): 触发命令的群号。
            user_id (Optional[int]): 触发命令的用户号。
            notify (Optional[Callable[[str], None]]): 文本通知回调。
            notify_media (Optional[Callable[[str, Sequence[XPostResult], str], None]]):
                携带图片的通知回调。
            persist (bool): 是否写入持久化存储。
            x_user_id (Optional[str]): 已持久化的 X 用户 ID。

        Returns:
            None: 无返回值。

        Raises:
            AssertionError: 当参数非法时抛出。
            RuntimeError: 当任务超过上限或 API 调用失败时抛出。
        """
        normalized = _normalize_username(username)
        assert interval > 0, "interval 必须大于 0"
        assert limit_per_cycle > 0, "limit_per_cycle 必须大于 0"
        assert notify is not None, "notify 回调不能为空"
        resolved_x_user_id = str(x_user_id or "").strip()
        resolved_username = normalized
        if not resolved_x_user_id:
            if self._use_browser_source():
                resolved_x_user_id = f"browser:{normalized.lower()}"
            else:
                profile = self._get_client().get_user_profile(normalized)
                resolved_username = profile.username
                resolved_x_user_id = profile.user_id
        with self._lock:
            self._prune_dead_watch_tasks()
            if len(self._watch_tasks) >= MAX_WATCH_TASKS:
                raise RuntimeError(f"最多支持 {MAX_WATCH_TASKS} 个 X 监控任务，请先关闭后再启动。")
            if self._use_browser_source():
                fetcher = lambda since_id: self._get_browser_client().latest(
                    resolved_username, limit=max(DEFAULT_LIMIT, limit_per_cycle)
                )
            else:
                fetcher = lambda since_id: self._fetch_results(
                    resolved_username, resolved_x_user_id, since_id=since_id
                )
            formatter = lambda items, tag: self.format_lines(items, tag)
            task = _XWatchTask(
                username=resolved_username,
                x_user_id=resolved_x_user_id,
                interval=interval,
                limit_per_cycle=limit_per_cycle,
                group_id=group_id,
                user_id=user_id,
                fetcher=fetcher,
                formatter=formatter,
                notify=notify,
                notify_media=notify_media,
            )
            self._watch_tasks.append(task)
            task.start()
            if persist:
                self._persist_locked()

    def stop_watch(self, username: str, group_id: Optional[int] = None) -> int:
        """
        停止匹配用户名的监控任务。

        Args:
            username (str): X 用户名。
            group_id (Optional[int]): 群号。为空时停止所有同名任务。

        Returns:
            int: 停止的任务数量。

        Raises:
            AssertionError: 当用户名格式非法时抛出。
        """
        _normalize_username(username)
        with self._lock:
            self._prune_dead_watch_tasks()
            remaining: list[_XWatchTask] = []
            stopped = 0
            for task in self._watch_tasks:
                if task.matches(username, group_id=group_id):
                    task.stop()
                    stopped += 1
                else:
                    remaining.append(task)
            if stopped:
                self._watch_tasks = remaining
                self._persist_locked()
            return stopped

    def restore_tasks(
        self,
        build_callbacks: Callable[
            [Optional[int], Optional[int], dict[str, object]],
            Sequence[Optional[Callable[..., object]]],
        ],
    ) -> int:
        """
        从持久化文件恢复监控任务。

        Args:
            build_callbacks (Callable): 根据持久化记录构造通知回调的工厂。

        Returns:
            int: 成功恢复的任务数量。

        Raises:
            None: 单条恢复失败会记录日志并继续处理下一条。
        """
        records = self._load_records()
        if not records:
            return 0
        restored = 0
        for record in records:
            try:
                username = str(record.get("username") or "").strip()
                x_user_id = str(record.get("x_user_id") or "").strip()
                interval = float(record.get("interval") or 0)
                limit_per_cycle = int(record.get("limit_per_cycle") or DEFAULT_LIMIT)
                group_id = record.get("group_id")
                user_id = record.get("user_id")
                if not username or not x_user_id or interval <= 0 or limit_per_cycle <= 0:
                    raise AssertionError("持久化记录缺少必要字段")
                callbacks = list(build_callbacks(group_id, user_id, record))
                if not callbacks:
                    raise AssertionError("缺少通知回调")
                notify_cb = callbacks[0]
                notify_media_cb = callbacks[1] if len(callbacks) >= 2 else None
                assert callable(notify_cb), "notify 回调不可为空"
                self.start_watch(
                    username=username,
                    interval=interval,
                    limit_per_cycle=limit_per_cycle,
                    group_id=int(group_id) if group_id is not None else None,
                    user_id=int(user_id) if user_id is not None else None,
                    notify=notify_cb,  # type: ignore[arg-type]
                    notify_media=notify_media_cb,  # type: ignore[arg-type]
                    persist=False,
                    x_user_id=x_user_id,
                )
                restored += 1
            except Exception as err:
                print(f"[XMonitor] 恢复任务失败：{record} -> {err}", flush=True)
        return restored

    def _get_client(self) -> XAPIClient:
        """
        获取 X API 客户端，必要时延迟初始化。

        Returns:
            XAPIClient: 可用的客户端实例。

        Raises:
            AssertionError: 当环境变量缺失时抛出。
        """
        with self._client_lock:
            if self._client is None:
                self._client = XAPIClient()
            return self._client

    def _use_browser_source(self) -> bool:
        """
        判断是否启用浏览器公开页面数据源。

        Returns:
            bool: X_MONITOR_SOURCE 为 browser/playwright/headless 时返回 True。

        Raises:
            None: 本方法不抛出异常。
        """
        source = os.environ.get(SOURCE_ENV, "api").strip().lower()
        return source in BROWSER_SOURCE_ALIASES and (
            self._client is None or self._browser_client is not None
        )

    def _get_browser_client(self) -> object:
        """
        获取浏览器模式客户端，必要时延迟初始化。

        Returns:
            object: 支持 latest/fetch_link 的浏览器客户端。

        Raises:
            RuntimeError: 当 Playwright 不可用时由客户端初始化路径抛出。
        """
        with self._browser_client_lock:
            if self._browser_client is None:
                from src.x_browser_monitor import XBrowserClient

                self._browser_client = XBrowserClient()
            return self._browser_client

    def _fetch_results(
        self,
        username: str,
        x_user_id: str,
        limit: int = DEFAULT_LIMIT,
        since_id: Optional[str] = None,
    ) -> list[XPostResult]:
        """
        调用接口并转换为标准推文结果。

        Args:
            username (str): X 用户名。
            x_user_id (str): X 用户 ID。
            limit (int): API 拉取数量。
            since_id (Optional[str]): 仅拉取该 ID 之后的推文。

        Returns:
            list[XPostResult]: 转换后的结果。

        Raises:
            AssertionError: 当 API 返回结构异常时抛出。
            RuntimeError: 当 X API 调用失败时抛出。
        """
        payload = self._get_client().get_user_posts(
            x_user_id, max_results=limit, since_id=since_id
        )
        return self._parse_posts(username, payload)

    def _parse_posts(
        self, username: str, payload: dict[str, object]
    ) -> list[XPostResult]:
        """
        将 X API 时间线响应转换为推文结果。

        Args:
            username (str): X 用户名。
            payload (dict[str, object]): X API 原始响应。

        Returns:
            list[XPostResult]: 推文结果列表。

        Raises:
            AssertionError: 当 API 返回结构异常时抛出。
            ValueError: 当时间字段格式非法时抛出。
        """
        media_by_key = self._build_media_map(payload)
        user_by_id = self._build_user_map(payload)
        data = payload.get("data") or []
        if not data:
            return []
        assert isinstance(data, list), "X API 推文 data 必须为列表"
        results: list[XPostResult] = []
        for raw_post in data:
            assert isinstance(raw_post, dict), "X API 单条推文必须为对象"
            post_id = str(raw_post.get("id") or "").strip()
            assert post_id, "X API 推文缺少 id"
            text = str(raw_post.get("text") or "").strip()
            created_label = _format_created_at(raw_post.get("created_at"))
            image_urls = _collect_post_image_urls(raw_post, media_by_key, limit=4)
            post_username = self._post_username(raw_post, user_by_id, username)
            display_name = self._post_display_name(
                raw_post, user_by_id, post_username
            )
            results.append(
                XPostResult(
                    username=post_username,
                    post_id=post_id,
                    text=text,
                    created_label=created_label,
                    url=POST_URL.format(username=post_username, post_id=post_id),
                    image_urls=image_urls,
                    source_payload=payload,
                    display_name=display_name,
                )
            )
        return results

    def _post_username(
        self,
        post: dict[str, object],
        user_by_id: dict[str, dict[str, object]],
        default_name: str,
    ) -> str:
        """
        从推文作者 expansion 中提取用户名。

        Args:
            post (dict[str, object]): 单条推文对象。
            user_by_id (dict[str, dict[str, object]]): 用户 ID 到用户对象的映射。
            default_name (str): 缺少作者 expansion 时使用的名称。

        Returns:
            str: 推文作者用户名。

        Raises:
            AssertionError: 当无法确定用户名时抛出。
        """
        author_id = str(post.get("author_id") or "").strip()
        raw_user = user_by_id.get(author_id) if author_id else None
        username = ""
        if raw_user is not None:
            username = str(raw_user.get("username") or "").strip()
        resolved = username or default_name.strip()
        assert resolved, "推文作者用户名不能为空"
        return resolved

    def _post_display_name(
        self,
        post: dict[str, object],
        user_by_id: dict[str, dict[str, object]],
        default_name: str,
    ) -> str:
        """
        从推文作者 expansion 中提取显示名。

        Args:
            post (dict[str, object]): 单条推文对象。
            user_by_id (dict[str, dict[str, object]]): 用户 ID 到用户对象的映射。
            default_name (str): 缺少作者 expansion 时使用的名称。

        Returns:
            str: 作者显示名。

        Raises:
            None: 本方法不主动抛出异常。
        """
        author_id = str(post.get("author_id") or "").strip()
        raw_user = user_by_id.get(author_id) if author_id else None
        if raw_user is None:
            return default_name.strip()
        name = str(raw_user.get("name") or "").strip()
        if name:
            return name
        return str(raw_user.get("username") or default_name).strip()

    def _build_media_map(
        self, payload: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        """
        从响应 includes 中构造媒体映射。

        Args:
            payload (dict[str, object]): X API 原始响应。

        Returns:
            dict[str, dict[str, object]]: media_key 到媒体对象的映射。

        Raises:
            AssertionError: 当 includes.media 类型异常时抛出。
        """
        includes = payload.get("includes") or {}
        if not includes:
            return {}
        assert isinstance(includes, dict), "X API includes 必须为对象"
        media_items = includes.get("media") or []
        if not media_items:
            return {}
        assert isinstance(media_items, list), "X API includes.media 必须为列表"
        media_by_key: dict[str, dict[str, object]] = {}
        for raw_media in media_items:
            assert isinstance(raw_media, dict), "X API 单条 media 必须为对象"
            media_key = str(raw_media.get("media_key") or "").strip()
            if media_key:
                media_by_key[media_key] = raw_media
        return media_by_key

    def _build_user_map(
        self, payload: dict[str, object]
    ) -> dict[str, dict[str, object]]:
        """
        从响应 includes 中构造用户映射。

        Args:
            payload (dict[str, object]): X API 原始响应。

        Returns:
            dict[str, dict[str, object]]: user_id 到用户对象的映射。

        Raises:
            AssertionError: 当 includes.users 类型异常时抛出。
        """
        includes = payload.get("includes") or {}
        if not includes:
            return {}
        assert isinstance(includes, dict), "X API includes 必须为对象"
        users = includes.get("users") or []
        if not users:
            return {}
        assert isinstance(users, list), "X API includes.users 必须为列表"
        user_by_id: dict[str, dict[str, object]] = {}
        for raw_user in users:
            assert isinstance(raw_user, dict), "X API 单条 user 必须为对象"
            user_id = str(raw_user.get("id") or "").strip()
            if user_id:
                user_by_id[user_id] = raw_user
        return user_by_id

    def _prune_dead_watch_tasks(self) -> None:
        """
        清理已停止的监控任务。

        Returns:
            None: 无返回值。

        Raises:
            None: 本方法不主动抛出异常。
        """
        alive_tasks = []
        for task in self._watch_tasks:
            if task.alive():
                alive_tasks.append(task)
        self._watch_tasks = alive_tasks

    def _load_records(self) -> list[dict[str, object]]:
        """
        读取持久化任务列表。

        Returns:
            list[dict[str, object]]: 持久化记录集合。

        Raises:
            None: 读取失败会记录日志并返回空列表。
        """
        if not self._store_path.exists():
            return []
        try:
            with self._store_path.open("r", encoding="utf-8") as f:
                raw = f.read().strip()
                if not raw:
                    return []
                data = json.loads(raw)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
        except Exception as err:
            print(f"[XMonitor] 读取持久化任务失败：{err}", flush=True)
        return []

    def _persist_locked(self) -> None:
        """
        在持锁状态下写入当前任务列表。

        Returns:
            None: 无返回值。

        Raises:
            None: 写入失败会记录日志。
        """
        try:
            records = [task.to_state() for task in self._watch_tasks]
            if not self._store_path.parent.exists():
                os.makedirs(self._store_path.parent, exist_ok=True)
            with self._store_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        except Exception as err:
            print(f"[XMonitor] 保存持久化任务失败：{err}", flush=True)

    @staticmethod
    def format_lines(items: Sequence[XPostResult], tag: str) -> str:
        """
        将推文结果格式化为多行文本。

        Args:
            items (Sequence[XPostResult]): 推文列表。
            tag (str): 标签前缀。

        Returns:
            str: 合并后的多行文本。

        Raises:
            None: 本方法不主动抛出异常。
        """
        if not items:
            return ""
        username = items[0].username
        lines = [
            f"[X {tag}] | @{username}",
            "检测到新推文！",
            "=========================",
        ]
        for idx, item in enumerate(items, 1):
            created = item.created_label or "未知"
            lines.extend(
                [
                    f"#{idx}",
                    item.text or "(无正文)",
                    f"时间：{created}",
                    f"链接：{item.url}",
                ]
            )
        return "\n".join(lines)
