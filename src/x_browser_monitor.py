"""
Browser-backed X public page monitor.

This backend uses Playwright to read public X pages without logging in. It does
not attempt to bypass login walls, CAPTCHA, private accounts, or paid API gates.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from src.x_monitor import (
    POST_URL,
    XPostResult,
    _format_created_at,
    _normalize_username,
    _parse_tweet_link,
)

BROWSER_BASE_URL_ENV = "X_MONITOR_BROWSER_BASE_URL"
BROWSER_TIMEOUT_ENV = "X_MONITOR_BROWSER_TIMEOUT_MS"
BROWSER_WAIT_ENV = "X_MONITOR_BROWSER_WAIT_MS"
BROWSER_HEADLESS_ENV = "X_MONITOR_BROWSER_HEADLESS"
BROWSER_STORAGE_STATE_ENV = "X_MONITOR_BROWSER_STORAGE_STATE"
DEFAULT_BROWSER_BASE_URL = "https://x.com"
DEFAULT_BROWSER_TIMEOUT_MS = 30000
DEFAULT_BROWSER_WAIT_MS = 2500
TWEET_ARTICLE_SELECTOR = 'article[data-testid="tweet"]'
STATUS_LINK_PATTERN = re.compile(r"/([^/?#]+)/status/(\d+)")
BLOCKED_HINTS = (
    "sign in to x",
    "log in to x",
    "create account",
    "something went wrong",
    "try reloading",
    "captcha",
    "unusual activity",
)


def _env_int(name: str, default: int) -> int:
    """
    Read a positive integer environment variable.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    """
    Read a boolean environment variable.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BrowserTweetDraft:
    """
    Parsed tweet data from a public page article.
    """

    username: str
    post_id: str
    text: str
    created_at: str
    display_name: str
    image_urls: tuple[str, ...]
    profile_image_url: str = ""

    def to_result(self) -> XPostResult:
        """
        Convert the draft into the monitor's normal result shape.
        """
        payload = build_browser_source_payload(self)
        return XPostResult(
            username=self.username,
            post_id=self.post_id,
            text=self.text,
            created_label=_format_created_at(self.created_at),
            url=POST_URL.format(username=self.username, post_id=self.post_id),
            image_urls=self.image_urls,
            source_payload=payload,
            display_name=self.display_name,
        )


def build_browser_source_payload(draft: BrowserTweetDraft) -> dict[str, object]:
    """
    Build a minimal X API-like payload for the existing image renderer.
    """
    author_id = f"browser:{draft.username.lower()}"
    media: list[dict[str, object]] = []
    media_keys: list[str] = []
    for idx, url in enumerate(draft.image_urls):
        key = f"browser_{draft.post_id}_{idx}"
        media_keys.append(key)
        media.append(
            {
                "media_key": key,
                "type": "photo",
                "url": url,
            }
        )

    post: dict[str, object] = {
        "id": draft.post_id,
        "text": draft.text,
        "author_id": author_id,
    }
    if draft.created_at:
        post["created_at"] = draft.created_at
    if media_keys:
        post["attachments"] = {"media_keys": media_keys}

    includes: dict[str, object] = {
        "users": [
            {
                "id": author_id,
                "name": draft.display_name or draft.username,
                "username": draft.username,
                "profile_image_url": draft.profile_image_url,
            }
        ]
    }
    if media:
        includes["media"] = media

    return {"data": [post], "includes": includes}


class XBrowserClient:
    """
    Playwright-backed reader for public X pages.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        wait_ms: Optional[int] = None,
        headless: Optional[bool] = None,
        storage_state: Optional[str] = None,
    ) -> None:
        """
        Initialize browser settings.
        """
        configured_base = base_url or os.environ.get(BROWSER_BASE_URL_ENV, "")
        self._base_url = (configured_base or DEFAULT_BROWSER_BASE_URL).rstrip("/")
        self._timeout_ms = timeout_ms or _env_int(
            BROWSER_TIMEOUT_ENV, DEFAULT_BROWSER_TIMEOUT_MS
        )
        self._wait_ms = wait_ms or _env_int(BROWSER_WAIT_ENV, DEFAULT_BROWSER_WAIT_MS)
        self._headless = (
            headless
            if headless is not None
            else _env_bool(BROWSER_HEADLESS_ENV, True)
        )
        configured_storage_state = (
            storage_state
            if storage_state is not None
            else os.environ.get(BROWSER_STORAGE_STATE_ENV, "")
        )
        self._storage_state = configured_storage_state.strip()

    def latest(self, username: str, limit: int = 5) -> list[XPostResult]:
        """
        Read latest visible public tweets from a profile page.
        """
        assert limit > 0, "limit 必须大于 0"
        normalized = _normalize_username(username)
        url = f"{self._base_url}/{normalized}"
        drafts = self._read_page(url, normalized, limit)
        return [draft.to_result() for draft in drafts[:limit]]

    def fetch_link(self, url: str) -> XPostResult:
        """
        Read a single public tweet page.
        """
        link = _parse_tweet_link(url)
        username = link.username or "unknown"
        target_url = f"{self._base_url}/{username}/status/{link.tweet_id}"
        drafts = self._read_page(target_url, username, limit=8)
        for draft in drafts:
            if draft.post_id == link.tweet_id:
                return draft.to_result()
        raise AssertionError("未在公开页面中获取到目标推文")

    def _read_page(
        self, url: str, fallback_username: str, limit: int
    ) -> list[BrowserTweetDraft]:
        """
        Open a public X page and parse visible tweet articles.
        """
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("浏览器模式需要安装 playwright。") from exc

        try:
            context_options = self._browser_context_options()
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=self._headless)
                context = None
                try:
                    context = browser.new_context(**context_options)
                    page = context.new_page()
                    page.set_default_timeout(self._timeout_ms)
                    page.goto(
                        url,
                        wait_until="domcontentloaded",
                        timeout=self._timeout_ms,
                    )
                    page.wait_for_timeout(self._wait_ms)
                    try:
                        page.locator(TWEET_ARTICLE_SELECTOR).first.wait_for(
                            timeout=min(self._timeout_ms, 10000)
                        )
                    except PlaywrightTimeoutError:
                        self._raise_page_blocked(page)
                    return self._extract_tweets(page, fallback_username, limit)
                finally:
                    if context is not None:
                        context.close()
                    browser.close()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"读取 X 公开页面失败：{exc}") from exc

    def _extract_tweets(
        self, page: object, fallback_username: str, limit: int
    ) -> list[BrowserTweetDraft]:
        """
        Extract tweet drafts from visible articles.
        """
        articles = page.locator(TWEET_ARTICLE_SELECTOR)
        count = articles.count()
        if count <= 0:
            self._raise_page_blocked(page)
        results: list[BrowserTweetDraft] = []
        seen: set[str] = set()
        for idx in range(count):
            if len(results) >= limit:
                break
            article = articles.nth(idx)
            draft = self._article_to_draft(article, fallback_username)
            if draft is None or draft.post_id in seen:
                continue
            seen.add(draft.post_id)
            results.append(draft)
        if not results:
            self._raise_page_blocked(page)
        return results

    def _article_to_draft(
        self, article: object, fallback_username: str
    ) -> Optional[BrowserTweetDraft]:
        """
        Parse one X tweet article into a draft.
        """
        username = _normalize_username(fallback_username) if fallback_username else ""
        post_id = ""
        links = article.locator('a[href*="/status/"]')
        for idx in range(links.count()):
            href = str(links.nth(idx).get_attribute("href") or "").strip()
            match = STATUS_LINK_PATTERN.search(href)
            if match is None:
                continue
            raw_username, raw_post_id = match.groups()
            if raw_post_id.isdigit():
                post_id = raw_post_id
                if raw_username and _is_valid_username(raw_username):
                    username = raw_username
                break
        if not post_id or not username:
            return None

        text = _safe_inner_text(article.locator('[data-testid="tweetText"]').first)
        if not text:
            text = _compact_article_text(_safe_inner_text(article))

        created_at = str(
            article.locator("time").first.get_attribute("datetime") or ""
        ).strip()
        display_name = _extract_display_name(article, username)
        image_urls = _extract_image_urls(article)
        profile_image_url = _extract_profile_image_url(article)
        return BrowserTweetDraft(
            username=username,
            post_id=post_id,
            text=text.strip(),
            created_at=created_at,
            display_name=display_name,
            image_urls=image_urls,
            profile_image_url=profile_image_url,
        )

    def _raise_page_blocked(self, page: object) -> None:
        """
        Raise a human-readable error when public page reading is unavailable.
        """
        body = _safe_inner_text(page.locator("body"))
        lower = body.lower()
        if any(hint in lower for hint in BLOCKED_HINTS):
            raise RuntimeError(
                "X 公开页面当前不可读：页面要求登录、验证码或返回错误。"
                "浏览器监控不会绕过这些限制。"
            )
        raise RuntimeError("X 公开页面中没有找到可读取的推文。")

    def _browser_context_options(self) -> dict[str, object]:
        """
        Return browser context options, including an optional login state file.
        """
        if not self._storage_state:
            return {}
        state_path = os.path.abspath(self._storage_state)
        if not os.path.exists(state_path):
            raise RuntimeError(
                f"X 浏览器登录态文件不存在：{state_path}。"
                f"请先导出登录态，或清空 {BROWSER_STORAGE_STATE_ENV}。"
            )
        return {"storage_state": state_path}


def _is_valid_username(value: str) -> bool:
    """
    Return whether a path segment can be an X username.
    """
    try:
        _normalize_username(value)
    except AssertionError:
        return False
    return True


def _safe_inner_text(locator: object) -> str:
    """
    Read inner_text from a Playwright locator without leaking exceptions.
    """
    try:
        return str(locator.inner_text(timeout=1000) or "").strip()
    except Exception:
        return ""


def _compact_article_text(value: str) -> str:
    """
    Keep a readable fallback text body from an article's full text.
    """
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    ignored = {"reply", "repost", "like", "view", "show more", "translate post"}
    kept = [line for line in lines if line.lower() not in ignored]
    return "\n".join(kept[:8])


def _extract_display_name(article: object, username: str) -> str:
    """
    Extract the visible display name from the article header.
    """
    raw = _safe_inner_text(article.locator('[data-testid="User-Name"]').first)
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("@"):
            continue
        return cleaned
    return username


def _extract_image_urls(article: object) -> tuple[str, ...]:
    """
    Extract public media image URLs from an article.
    """
    images = article.locator('img[src*="twimg.com/media"]')
    urls: list[str] = []
    for idx in range(images.count()):
        src = str(images.nth(idx).get_attribute("src") or "").strip()
        if src.startswith("http") and src not in urls:
            urls.append(src)
        if len(urls) >= 4:
            break
    return tuple(urls)


def _extract_profile_image_url(article: object) -> str:
    """
    Extract the author's public profile image URL from an article.
    """
    images = article.locator('img[src*="profile_images"]')
    for idx in range(images.count()):
        src = str(images.nth(idx).get_attribute("src") or "").strip()
        if src.startswith("http"):
            return src
    return ""
