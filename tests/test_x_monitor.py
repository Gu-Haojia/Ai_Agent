"""
X 推文监控解析与图文消息拼装单元测试。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from threading import Event
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

from src.x_monitor import (
    DEFAULT_LIMIT,
    NEW_POST_NOTICE_ENV,
    RESTORE_MODE_ENV,
    XAPIClient,
    XMonitorManager,
    XPostResult,
    _collect_post_image_urls,
    _normalize_username,
    _parse_tweet_link,
    _XWatchTask,
)
from src.x_monitor_media import (
    XPostImagePayloadBuilder,
    _send_group_msg,
    compose_x_media_message,
)
from src.x_monitor_render import (
    BrowserRenderConfig,
    BrowserTweetRenderer,
    XTweetPayloadParser,
    render_tweet_html,
)
from src.x_monitor_translate import (
    GeminiTweetTranslator,
    TRANSLATION_MODEL,
    TRANSLATION_MODE_ENV,
    XRenderedTweetTextTranslator,
    XTweetTranslationMode,
)
import qq_group_bot
from qq_group_bot import QQBotHandler


def build_render_payload(
    post_id: str = "1",
    text: str = "hello #XMonitor",
    image_url: str = "https://example.com/1.jpg",
) -> dict[str, object]:
    """
    构造用于推文图片渲染测试的 X API payload。

    Args:
        post_id (str): 推文 ID。
        text (str): 推文正文。
        image_url (str): 推文图片 URL。

    Returns:
        dict[str, object]: X API 风格响应。

    Raises:
        None: 本函数不主动抛出异常。
    """
    return {
        "data": [
            {
                "id": post_id,
                "text": text,
                "author_id": "10",
                "created_at": "2026-05-05T01:02:03.000Z",
                "attachments": {"media_keys": ["3_1"]},
            }
        ],
        "includes": {
            "users": [
                {
                    "id": "10",
                    "name": "Kana Hanaiwa",
                    "username": "kana_hanaiwa",
                    "profile_image_url": "https://example.com/avatar_normal.jpg",
                }
            ],
            "media": [
                {
                    "media_key": "3_1",
                    "type": "photo",
                    "url": image_url,
                    "width": 1200,
                    "height": 900,
                }
            ],
        },
    }


class FakeXMonitorManager(XMonitorManager):
    """
    用于命令测试的 X 监控管理器替身。
    """

    def __init__(self) -> None:
        """
        初始化调用记录。

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        self.latest_calls: list[tuple[str, int]] = []
        self.fetch_link_calls: list[str] = []

    def latest(self, username: str, limit: int = 5) -> list[XPostResult]:
        """
        返回固定的最新推文。

        Args:
            username (str): X 用户名。
            limit (int): 返回数量上限。

        Returns:
            list[XPostResult]: 固定推文列表。

        Raises:
            None
        """
        self.latest_calls.append((username, limit))
        return [
            XPostResult(
                username="kana_hanaiwa",
                post_id="1",
                text="latest post",
                created_label="05-05 10:02",
                url="https://x.com/kana_hanaiwa/status/1",
                image_urls=("https://example.com/1.jpg",),
                source_payload=build_render_payload(text="latest post"),
            )
        ]

    def fetch_link(self, url: str) -> XPostResult:
        """
        返回固定的链接推文。

        Args:
            url (str): X 推文链接。

        Returns:
            XPostResult: 固定推文。

        Raises:
            None
        """
        self.fetch_link_calls.append(url)
        return XPostResult(
            username="kana_hanaiwa",
            post_id="1",
            text="linked post",
            created_label="05-05 10:02",
            url="https://x.com/kana_hanaiwa/status/1",
            image_urls=("https://example.com/1.jpg",),
            source_payload=build_render_payload(text="linked post"),
        )


class XMonitorParseTests(unittest.TestCase):
    """
    验证 X API 响应解析逻辑。
    """

    def test_normalize_username_accepts_at_prefix(self) -> None:
        """
        用户名允许带 @ 前缀，并会被清理为裸用户名。
        """
        self.assertEqual(_normalize_username("@kana_hanaiwa"), "kana_hanaiwa")

    def test_parse_tweet_link_reads_status_id(self) -> None:
        """
        推文链接解析应支持带查询参数的 x.com 链接。
        """
        link = _parse_tweet_link(
            "https://x.com/kana_hanaiwa/status/1919610000000000001?s=20"
        )

        self.assertEqual(link.username, "kana_hanaiwa")
        self.assertEqual(link.tweet_id, "1919610000000000001")

    def test_parse_tweet_link_rejects_invalid_host(self) -> None:
        """
        非 X/Twitter 域名不应被当作推文链接处理。
        """
        with self.assertRaises(AssertionError):
            _parse_tweet_link("https://example.com/kana/status/1")

    def test_collect_post_image_urls_reads_photo_and_preview(self) -> None:
        """
        应从 photo 读取 url，并从 video 读取 preview_image_url。
        """
        post = {"attachments": {"media_keys": ["3_1", "7_2"]}}
        media_by_key = {
            "3_1": {
                "media_key": "3_1",
                "type": "photo",
                "url": "https://example.com/photo.jpg",
            },
            "7_2": {
                "media_key": "7_2",
                "type": "video",
                "preview_image_url": "https://example.com/video.jpg",
            },
        }
        urls = _collect_post_image_urls(post, media_by_key, limit=4)
        self.assertEqual(
            urls,
            (
                "https://example.com/photo.jpg",
                "https://example.com/video.jpg",
            ),
        )

    def test_parse_posts_with_media_expansion(self) -> None:
        """
        应将 X API timeline 响应转换为 XPostResult。
        """
        payload = {
            "data": [
                {
                    "id": "1919610000000000001",
                    "text": "hello\nworld",
                    "author_id": "10",
                    "created_at": "2026-05-05T01:02:03.000Z",
                    "attachments": {"media_keys": ["3_1"]},
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "10",
                        "name": "Kana Hanaiwa",
                        "username": "kana_hanaiwa",
                    }
                ],
                "media": [
                    {
                        "media_key": "3_1",
                        "type": "photo",
                        "url": "https://example.com/p.jpg",
                    }
                ]
            },
        }
        manager = XMonitorManager()
        results = manager._parse_posts("kana_hanaiwa", payload)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].username, "kana_hanaiwa")
        self.assertEqual(results[0].post_id, "1919610000000000001")
        self.assertEqual(results[0].image_urls, ("https://example.com/p.jpg",))
        self.assertEqual(results[0].display_name, "Kana Hanaiwa")
        self.assertEqual(
            results[0].url,
            "https://x.com/kana_hanaiwa/status/1919610000000000001",
        )
        self.assertIs(results[0].source_payload, payload)

    def test_get_tweet_by_id_uses_lookup_fields(self) -> None:
        """
        单推文拉取应请求图片渲染所需字段和 expansions。
        """

        class FakeClient(XAPIClient):
            """
            记录 X API 请求参数的客户端。
            """

            def __init__(self) -> None:
                """
                初始化请求记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.calls: list[dict[str, object]] = []

            def _request_json(
                self, url: str, params: dict[str, str] | None = None
            ) -> dict[str, object]:
                """
                记录请求并返回空响应。

                Args:
                    url (str): 请求 URL。
                    params (dict[str, str] | None): 查询参数。

                Returns:
                    dict[str, object]: 空 X API 响应。

                Raises:
                    None
                """
                self.calls.append({"url": url, "params": params or {}})
                return {"data": []}

        client = FakeClient()
        payload = client.get_tweet_by_id("1919610000000000001")

        self.assertEqual(payload, {"data": []})
        self.assertEqual(client.calls[0]["url"], "https://api.x.com/2/tweets")
        params = client.calls[0]["params"]
        self.assertIsInstance(params, dict)
        self.assertEqual(params["ids"], "1919610000000000001")
        self.assertIn("note_tweet", params["tweet.fields"])
        self.assertIn("referenced_tweets.id", params["expansions"])
        self.assertIn("profile_image_url", params["user.fields"])

    def test_get_user_profile_requests_most_recent_tweet_id(self) -> None:
        """
        按用户名查询资料时应请求最新推文 ID 字段。
        """

        class FakeClient(XAPIClient):
            """
            记录用户查询请求的客户端。
            """

            def __init__(self) -> None:
                """
                初始化请求记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.calls: list[dict[str, object]] = []

            def _request_json(
                self, url: str, params: dict[str, str] | None = None
            ) -> dict[str, object]:
                """
                记录请求并返回固定用户响应。

                Args:
                    url (str): 请求 URL。
                    params (dict[str, str] | None): 查询参数。

                Returns:
                    dict[str, object]: X API 用户响应。

                Raises:
                    None
                """
                self.calls.append({"url": url, "params": params or {}})
                return {
                    "data": {
                        "id": "10",
                        "name": "Kana Hanaiwa",
                        "username": "kana_hanaiwa",
                        "most_recent_tweet_id": "12345",
                    }
                }

        client = FakeClient()
        profile = client.get_user_profile("@kana_hanaiwa")

        self.assertEqual(
            client.calls[0]["url"],
            "https://api.x.com/2/users/by/username/kana_hanaiwa",
        )
        params = client.calls[0]["params"]
        self.assertIsInstance(params, dict)
        self.assertIn("most_recent_tweet_id", params["user.fields"])
        self.assertEqual(profile.most_recent_tweet_id, "12345")

    def test_get_user_profile_by_id_requests_most_recent_tweet_id(self) -> None:
        """
        按用户 ID 查询资料时应请求最新推文 ID 字段。
        """

        class FakeClient(XAPIClient):
            """
            记录按 ID 查询用户请求的客户端。
            """

            def __init__(self) -> None:
                """
                初始化请求记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.calls: list[dict[str, object]] = []

            def _request_json(
                self, url: str, params: dict[str, str] | None = None
            ) -> dict[str, object]:
                """
                记录请求并返回固定用户响应。

                Args:
                    url (str): 请求 URL。
                    params (dict[str, str] | None): 查询参数。

                Returns:
                    dict[str, object]: X API 用户响应。

                Raises:
                    None
                """
                self.calls.append({"url": url, "params": params or {}})
                return {
                    "data": {
                        "id": "10",
                        "name": "Kana Hanaiwa",
                        "username": "kana_hanaiwa",
                        "most_recent_tweet_id": "67890",
                    }
                }

        client = FakeClient()
        profile = client.get_user_profile_by_id("10")

        self.assertEqual(client.calls[0]["url"], "https://api.x.com/2/users/10")
        params = client.calls[0]["params"]
        self.assertIsInstance(params, dict)
        self.assertIn("most_recent_tweet_id", params["user.fields"])
        self.assertEqual(profile.username, "kana_hanaiwa")
        self.assertEqual(profile.most_recent_tweet_id, "67890")

    def test_get_user_posts_excludes_retweets_and_replies(self) -> None:
        """
        拉取用户时间线时应同时排除转推与回复。
        """

        class FakeClient(XAPIClient):
            """
            记录用户时间线请求的客户端。
            """

            def __init__(self) -> None:
                """
                初始化请求记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.calls: list[dict[str, object]] = []

            def _request_json(
                self, url: str, params: dict[str, str] | None = None
            ) -> dict[str, object]:
                """
                记录请求并返回空时间线响应。

                Args:
                    url (str): 请求 URL。
                    params (dict[str, str] | None): 查询参数。

                Returns:
                    dict[str, object]: 空 X API 响应。

                Raises:
                    None
                """
                self.calls.append({"url": url, "params": params or {}})
                return {"data": []}

        client = FakeClient()
        payload = client.get_user_posts("10", max_results=DEFAULT_LIMIT, since_id="99")

        self.assertEqual(payload, {"data": []})
        self.assertEqual(client.calls[0]["url"], "https://api.x.com/2/users/10/tweets")
        params = client.calls[0]["params"]
        self.assertIsInstance(params, dict)
        self.assertEqual(params["exclude"], "retweets,replies")
        self.assertEqual(params["since_id"], "99")

    def test_fetch_link_builds_single_post_result(self) -> None:
        """
        管理器应按链接拉取单条推文并使用 API 作者信息构造结果。
        """

        class FakeClient:
            """
            返回固定单推文响应的客户端。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.calls: list[str] = []

            def get_tweet_by_id(self, tweet_id: str) -> dict[str, object]:
                """
                返回固定推文 payload。

                Args:
                    tweet_id (str): 推文 ID。

                Returns:
                    dict[str, object]: X API 风格响应。

                Raises:
                    None
                """
                self.calls.append(tweet_id)
                return {
                    "data": [
                        {
                            "id": tweet_id,
                            "text": "linked post",
                            "author_id": "10",
                            "created_at": "2026-05-05T01:02:03.000Z",
                        }
                    ],
                    "includes": {
                        "users": [
                            {
                                "id": "10",
                                "name": "Kana Hanaiwa",
                                "username": "kana_hanaiwa",
                            }
                        ]
                    },
                }

        client = FakeClient()
        manager = XMonitorManager(client=client)
        result = manager.fetch_link(
            "https://x.com/someone/status/1919610000000000001?s=20"
        )

        self.assertEqual(client.calls, ["1919610000000000001"])
        self.assertEqual(result.username, "kana_hanaiwa")
        self.assertEqual(result.display_name, "Kana Hanaiwa")
        self.assertEqual(
            result.url,
            "https://x.com/kana_hanaiwa/status/1919610000000000001",
        )
        self.assertIsNotNone(result.source_payload)
        self.assertRegex(result.created_label, r"\d{2}-\d{2} \d{2}:\d{2}")

    def test_latest_uses_default_limit_as_minimum_page_size(self) -> None:
        """
        latest 请求少量推文时，API 拉取数量应使用统一默认值 5。
        """

        class FakeClient:
            """
            记录 X API 调用参数的客户端替身。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.max_results_calls: list[int] = []

            def get_user_profile(self, username: str) -> SimpleNamespace:
                """
                返回固定用户资料。

                Args:
                    username (str): X 用户名。

                Returns:
                    SimpleNamespace: 含 user_id 与 username 的资料对象。

                Raises:
                    None
                """
                return SimpleNamespace(
                    user_id="10",
                    username=username.lstrip("@"),
                    most_recent_tweet_id="12345",
                )

            def get_user_posts(
                self,
                user_id: str,
                max_results: int = DEFAULT_LIMIT,
                since_id: str | None = None,
            ) -> dict[str, object]:
                """
                返回固定推文响应并记录 max_results。

                Args:
                    user_id (str): X 用户 ID。
                    max_results (int): API 拉取数量。
                    since_id (str | None): 增量拉取起点。

                Returns:
                    dict[str, object]: X API 风格响应。

                Raises:
                    None
                """
                self.max_results_calls.append(max_results)
                return {
                    "data": [
                        {
                            "id": str(100 + idx),
                            "text": f"post {idx}",
                            "created_at": "2026-05-05T01:02:03.000Z",
                        }
                        for idx in range(max_results)
                    ]
                }

        client = FakeClient()
        manager = XMonitorManager(client=client)
        results = manager.latest("@kana_hanaiwa", limit=1)

        self.assertEqual(client.max_results_calls, [DEFAULT_LIMIT])
        self.assertEqual(len(results), 1)


class XMonitorRenderTests(unittest.TestCase):
    """
    验证 XMonitor 风格的 payload 解析与 HTML 渲染。
    """

    def test_renderer_uses_configured_device_scale_factor(self) -> None:
        """
        截图渲染器应使用配置中的设备缩放因子。
        """
        payload = build_render_payload()
        tweet = XTweetPayloadParser().parse(payload)[0]
        locator = mock.MagicMock()
        locator.screenshot.return_value = b"png"
        page = mock.MagicMock()
        page.locator.return_value = locator
        browser = mock.MagicMock()
        browser.new_page.return_value = page
        playwright = SimpleNamespace(
            chromium=SimpleNamespace(launch=mock.Mock(return_value=browser))
        )
        context = mock.MagicMock()
        context.__enter__.return_value = playwright
        context.__exit__.return_value = None

        with mock.patch("playwright.sync_api.sync_playwright", return_value=context):
            data = BrowserTweetRenderer(
                BrowserRenderConfig(device_scale_factor=1.25)
            ).render_to_png_bytes(tweet)

        self.assertEqual(data, b"png")
        self.assertEqual(
            browser.new_page.call_args.kwargs["device_scale_factor"],
            1.25,
        )

    def test_parse_payload_reads_author_media_and_text_entities(self) -> None:
        """
        应解析作者、媒体，并在 HTML 中标记正文 entity。
        """
        payload = build_render_payload(text="hello @official #XMonitor")
        tweets = XTweetPayloadParser().parse(payload)
        self.assertEqual(len(tweets), 1)
        self.assertEqual(tweets[0].author.name, "Kana Hanaiwa")
        self.assertEqual(tweets[0].media[0].best_url, "https://example.com/1.jpg")

        html = render_tweet_html(tweets[0], BrowserRenderConfig(width=720))
        self.assertIn("Kana Hanaiwa", html)
        self.assertIn("@kana_hanaiwa", html)
        self.assertIn('<span class="mention">@official</span>', html)
        self.assertIn('<span class="hashtag">#XMonitor</span>', html)
        self.assertIn("https://example.com/1.jpg", html)
        self.assertIn('"Noto Sans CJK JP"', html)
        self.assertIn('"Noto Color Emoji"', html)
        self.assertIn(
            '<div class="agent-footer">本推文由 筱泽广Agent 提供</div>',
            html,
        )
        self.assertIn(".agent-footer", html)

    def test_render_quote_only_expands_one_level(self) -> None:
        """
        引用推文内部的再次引用不应继续递归渲染。
        """
        payload = {
            "data": [
                {
                    "id": "1",
                    "text": "top body",
                    "author_id": "10",
                    "referenced_tweets": [{"type": "quoted", "id": "2"}],
                }
            ],
            "includes": {
                "users": [
                    {"id": "10", "name": "Top User", "username": "top_user"},
                    {"id": "20", "name": "Quote User", "username": "quote_user"},
                    {"id": "30", "name": "Nested User", "username": "nested_user"},
                ],
                "tweets": [
                    {
                        "id": "2",
                        "text": "quote body",
                        "author_id": "20",
                        "referenced_tweets": [{"type": "quoted", "id": "3"}],
                    },
                    {
                        "id": "3",
                        "text": "nested body",
                        "author_id": "30",
                    },
                ],
            },
        }
        tweets = XTweetPayloadParser().parse(payload)

        html = render_tweet_html(tweets[0], BrowserRenderConfig(width=720))

        self.assertIn("Top User", html)
        self.assertIn("Quote User", html)
        self.assertIn("quote body", html)
        self.assertNotIn("Nested User", html)
        self.assertNotIn("nested body", html)
        self.assertEqual(html.count('<div class="quote">'), 1)

    def test_render_text_entities_does_not_mark_email_at(self) -> None:
        """
        邮箱中间的 @ 不应被识别为 X mention。
        """
        payload = build_render_payload(text="mail user@example.com @valid_user")
        tweets = XTweetPayloadParser().parse(payload)
        html = render_tweet_html(tweets[0], BrowserRenderConfig(width=720))
        self.assertIn("user@example.com", html)
        self.assertIn('<span class="mention">@valid_user</span>', html)
        self.assertNotIn('<span class="mention">@example</span>', html)

    def test_parse_payload_keeps_short_expanded_url(self) -> None:
        """
        URL entity 的真实链接较短时应保留 expanded_url。
        """
        payload = build_render_payload(text="read https://t.co/short")
        payload["data"][0]["entities"] = {
            "urls": [
                {
                    "url": "https://t.co/short",
                    "expanded_url": "https://fanbox.cc/@kana/posts/12345",
                    "display_url": "fanbox.cc/@kana/posts...",
                }
            ]
        }

        tweets = XTweetPayloadParser().parse(payload)

        self.assertIn("https://fanbox.cc/@kana/posts/12345", tweets[0].text)
        self.assertNotIn("fanbox.cc/@kana/posts...", tweets[0].text)

    def test_parse_payload_uses_display_url_for_long_expanded_url(self) -> None:
        """
        URL entity 的真实链接过长时应使用 display_url。
        """
        expanded_url = "https://fanbox.cc/@kana/posts/" + "1234567890" * 6
        payload = build_render_payload(text="read https://t.co/long")
        payload["data"][0]["entities"] = {
            "urls": [
                {
                    "url": "https://t.co/long",
                    "expanded_url": expanded_url,
                    "display_url": "fanbox.cc/@kana/posts...",
                }
            ]
        }

        tweets = XTweetPayloadParser().parse(payload)

        self.assertIn("fanbox.cc/@kana/posts...", tweets[0].text)
        self.assertNotIn(expanded_url, tweets[0].text)

    def test_parse_payload_uses_note_tweet_url_entities(self) -> None:
        """
        长文正文来自 note_tweet 时应同步使用 note_tweet 的 URL entity。
        """
        payload = build_render_payload(text="short body https://t.co/photo")
        payload["data"][0]["entities"] = {
            "urls": [
                {
                    "url": "https://t.co/photo",
                    "expanded_url": "https://x.com/kana/status/1/photo/1",
                    "display_url": "pic.x.com/photo",
                    "media_key": "3_1",
                }
            ]
        }
        payload["data"][0]["note_tweet"] = {
            "text": "event detail\nhttps://t.co/live",
            "entities": {
                "urls": [
                    {
                        "url": "https://t.co/live",
                        "expanded_url": "http://livepocket.jp/e/c3pwb",
                        "display_url": "livepocket.jp/e/c3pwb",
                    }
                ]
            },
        }

        tweets = XTweetPayloadParser().parse(payload)

        self.assertIn("http://livepocket.jp/e/c3pwb", tweets[0].text)
        self.assertNotIn("https://t.co/live", tweets[0].text)
        self.assertNotIn("pic.x.com/photo", tweets[0].text)

    def test_parse_payload_unescapes_html_entities_before_rendering(self) -> None:
        """
        X API 正文中的 HTML entity 应还原为原始字符。
        """
        payload = build_render_payload(text="&lt;#いずみんち &amp; next")
        tweets = XTweetPayloadParser().parse(payload)
        self.assertEqual(tweets[0].text, "<#いずみんち & next")

        html = render_tweet_html(tweets[0], BrowserRenderConfig(width=720))
        self.assertIn("&lt;<span class=\"hashtag\">#いずみんち</span> &amp; next", html)
        self.assertNotIn("&amp;lt;#いずみんち", html)

    def test_render_translation_container_with_divider_and_chinese_font(self) -> None:
        """
        对照模式的简体中文译文应使用分隔线容器和中文字体。
        """
        payload = build_render_payload(text="hello")
        tweets = XTweetPayloadParser().parse(payload)
        tweets[0].translation_text = "你好 @official #话题"

        html = render_tweet_html(tweets[0], BrowserRenderConfig(width=720))

        self.assertIn('<section class="translation" lang="zh-CN">', html)
        self.assertIn(".translation::before", html)
        self.assertIn("width: 90%", html)
        self.assertIn("background: var(--border)", html)
        self.assertNotIn("简中翻译", html)
        self.assertIn('"Noto Sans CJK SC"', html)
        self.assertIn('"Apple Color Emoji"', html)
        self.assertIn('"Noto Color Emoji"', html)
        self.assertIn('<span class="mention">@official</span>', html)
        self.assertIn('<span class="hashtag">#话题</span>', html)


class XMonitorTranslationTests(unittest.TestCase):
    """
    验证 XMonitor 推文正文翻译模式。
    """

    def test_translation_mode_accepts_expected_aliases(self) -> None:
        """
        翻译模式应支持不翻译、仅翻译和对照三类配置。
        """
        self.assertEqual(
            XTweetTranslationMode.normalize("不翻译"),
            XTweetTranslationMode.NONE,
        )
        self.assertEqual(
            XTweetTranslationMode.normalize("仅翻译"),
            XTweetTranslationMode.TRANSLATED,
        )
        self.assertEqual(
            XTweetTranslationMode.normalize("对照"),
            XTweetTranslationMode.BILINGUAL,
        )

    def test_gemini_translator_uses_official_client_model(self) -> None:
        """
        Gemini 翻译器应调用指定模型并解析 JSON 译文。
        """

        class FakeModels:
            """
            模拟 Google GenAI models API。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.calls: list[dict[str, object]] = []

            def generate_content(
                self, model: str, contents: object, config: object
            ) -> SimpleNamespace:
                """
                记录生成请求并返回固定 JSON。

                Args:
                    model (str): 模型名称。
                    contents (object): 请求内容。
                    config (object): 生成配置。

                Returns:
                    SimpleNamespace: 模拟响应。

                Raises:
                    None
                """
                self.calls.append(
                    {"model": model, "contents": contents, "config": config}
                )
                return SimpleNamespace(text='{"translations":["你好"]}')

        fake_models = FakeModels()
        client = SimpleNamespace(models=fake_models)

        result = GeminiTweetTranslator(client=client).translate_texts(["hello"])

        self.assertEqual(result, ["你好"])
        self.assertEqual(fake_models.calls[0]["model"], TRANSLATION_MODEL)
        self.assertIn("hello", str(fake_models.calls[0]["contents"]))

    def test_rendered_tweet_translator_can_replace_text_only(self) -> None:
        """
        仅翻译模式应使用简体中文译文替换原文。
        """

        class FakeTranslator:
            """
            返回固定译文的翻译器。
            """

            def translate_texts(self, texts: Sequence[str]) -> list[str]:
                """
                返回与输入数量一致的固定译文。

                Args:
                    texts (Sequence[str]): 原文列表。

                Returns:
                    list[str]: 固定译文列表。

                Raises:
                    None
                """
                return ["你好" for _ in texts]

        tweet = XTweetPayloadParser().parse(build_render_payload(text="hello"))[0]
        translator = XRenderedTweetTextTranslator(FakeTranslator())

        translator.apply(tweet, XTweetTranslationMode.TRANSLATED)

        self.assertEqual(tweet.text, "你好")
        self.assertIsNone(tweet.translation_text)
        self.assertTrue(tweet.is_translated_text)
        html = render_tweet_html(tweet, BrowserRenderConfig(width=720))
        self.assertIn('<div class="text translated-text">你好</div>', html)
        self.assertIn(".text.translated-text", html)

    def test_rendered_tweet_translator_can_render_bilingual_text(self) -> None:
        """
        对照模式应保留原文，并将简体中文译文放入独立字段。
        """

        class FakeTranslator:
            """
            返回固定译文的翻译器。
            """

            def translate_texts(self, texts: Sequence[str]) -> list[str]:
                """
                返回与输入数量一致的固定译文。

                Args:
                    texts (Sequence[str]): 原文列表。

                Returns:
                    list[str]: 固定译文列表。

                Raises:
                    None
                """
                return ["你好" for _ in texts]

        tweet = XTweetPayloadParser().parse(build_render_payload(text="hello"))[0]
        translator = XRenderedTweetTextTranslator(FakeTranslator())

        translator.apply(tweet, XTweetTranslationMode.BILINGUAL)

        self.assertEqual(tweet.text, "hello")
        self.assertEqual(tweet.translation_text, "你好")
        self.assertFalse(tweet.is_translated_text)


class XMonitorMediaComposeTests(unittest.TestCase):
    """
    验证 X 推文 OneBot 图文消息拼装。
    """

    def test_send_group_msg_uses_sixty_second_timeout(self) -> None:
        """
        OneBot 发送应等待 60 秒，减少图片消息响应慢导致的误判超时。
        """

        class FakeResponse:
            """
            模拟 OneBot HTTP 响应。
            """

            status = 200

            def __enter__(self) -> "FakeResponse":
                """
                进入上下文管理器。

                Returns:
                    FakeResponse: 当前响应对象。

                Raises:
                    None: 本方法不主动抛出异常。
                """
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                """
                退出上下文管理器。

                Args:
                    exc_type (object): 异常类型。
                    exc (object): 异常实例。
                    traceback (object): 异常栈。

                Returns:
                    None: 无返回值。

                Raises:
                    None: 本方法不主动抛出异常。
                """
                return None

        with mock.patch(
            "src.x_monitor_media.urlopen", return_value=FakeResponse()
        ) as mocked:
            _send_group_msg("http://onebot", 123, "hello", "token")

        self.assertEqual(mocked.call_args.kwargs["timeout"], 60)

    def test_image_builder_uses_translation_mode_env(self) -> None:
        """
        推文图片构建器应在渲染前按环境变量翻译正文。
        """

        class FakeRenderer:
            """
            记录渲染时收到的推文文本。
            """

            def __init__(self) -> None:
                """
                初始化捕获字段。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.captured_text = ""

            def render_to_png_bytes(self, tweet: object) -> bytes:
                """
                捕获推文文本并返回固定 PNG 字节。

                Args:
                    tweet (object): 可渲染推文。

                Returns:
                    bytes: 固定 PNG 字节。

                Raises:
                    None
                """
                self.captured_text = str(getattr(tweet, "text"))
                return b"png"

        class FakeTranslator:
            """
            返回固定译文的翻译器。
            """

            def translate_texts(self, texts: Sequence[str]) -> list[str]:
                """
                返回与输入数量一致的固定译文。

                Args:
                    texts (Sequence[str]): 原文列表。

                Returns:
                    list[str]: 固定译文列表。

                Raises:
                    None
                """
                return ["你好" for _ in texts]

        item = XPostResult(
            username="kana_hanaiwa",
            post_id="1",
            text="hello",
            created_label="05-05 10:02",
            url="https://x.com/kana_hanaiwa/status/1",
            source_payload=build_render_payload(text="hello"),
        )
        fake_renderer = FakeRenderer()
        builder = XPostImagePayloadBuilder(
            renderer=fake_renderer,
            text_translator=XRenderedTweetTextTranslator(FakeTranslator()),
        )

        with mock.patch.dict("os.environ", {TRANSLATION_MODE_ENV: "only"}):
            b64, mime = builder.render(item)

        self.assertEqual(b64, "cG5n")
        self.assertEqual(mime, "image/png")
        self.assertEqual(fake_renderer.captured_text, "你好")

    def test_compose_renders_tweet_image_by_default(self) -> None:
        """
        默认应发送解析后的推文截图，不直接附带文本。
        """
        items = [
            XPostResult(
                username="kana_hanaiwa",
                post_id="1",
                text="post",
                created_label="05-05 10:02",
                url="https://x.com/kana_hanaiwa/status/1",
                image_urls=("https://example.com/1.jpg",),
                source_payload=build_render_payload(),
            )
        ]

        def fake_render(item: XPostResult) -> tuple[str, str]:
            """
            返回固定渲染图。

            Args:
                item (XPostResult): 推文结果。

            Returns:
                tuple[str, str]: base64 与 MIME。

            Raises:
                None: 本函数不抛出异常。
            """
            self.assertEqual(item.post_id, "1")
            return "Zg==", "image/png"

        payload = compose_x_media_message("hello", items, renderer=fake_render)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["type"], "image")
        self.assertTrue(payload[0]["data"]["file"].startswith("base64://Zg=="))
        self.assertEqual(payload[0]["data"]["cache"], "0")

    def test_compose_can_include_text_when_enabled(self) -> None:
        """
        开启文本选项时，应生成 text + rendered image 消息段。
        """
        items = [
            XPostResult(
                username="kana_hanaiwa",
                post_id="1",
                text="post",
                created_label="05-05 10:02",
                url="https://x.com/kana_hanaiwa/status/1",
                image_urls=("https://example.com/1.jpg",),
                source_payload=build_render_payload(),
            )
        ]

        def fake_render(item: XPostResult) -> tuple[str, str]:
            """
            返回固定渲染图。

            Args:
                item (XPostResult): 推文结果。

            Returns:
                tuple[str, str]: base64 与 MIME。

            Raises:
                None: 本函数不抛出异常。
            """
            self.assertEqual(item.username, "kana_hanaiwa")
            return "Zg==", "image/png"

        payload = compose_x_media_message(
            "hello", items, renderer=fake_render, include_text=True
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)
        self.assertEqual(
            [segment["type"] for segment in payload],
            ["text", "image"],
        )
        self.assertEqual(payload[0]["data"]["text"], "hello")

    def test_legacy_env_uses_original_text_and_media(self) -> None:
        """
        旧版环境变量开启时，应沿用文本 + 原图发送模式。
        """
        items = [
            XPostResult(
                username="kana_hanaiwa",
                post_id="1",
                text="multi",
                created_label="05-05 10:02",
                url="https://x.com/kana_hanaiwa/status/1",
                image_urls=(
                    "https://example.com/1.jpg",
                    "https://example.com/2.jpg",
                ),
                source_payload=build_render_payload(),
            )
        ]
        fetched: list[str] = []

        def fake_fetch(url: str) -> tuple[str, str]:
            """
            记录被下载的图片 URL。

            Args:
                url (str): 图片 URL。

            Returns:
                tuple[str, str]: base64 与 MIME。

            Raises:
                None: 本函数不抛出异常。
            """
            fetched.append(url)
            return "Zg==", "image/jpeg"

        with mock.patch.dict("os.environ", {"X_MONITOR_LEGACY_MEDIA": "1"}):
            payload = compose_x_media_message("hello", items, fetcher=fake_fetch)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 3)
        self.assertEqual(payload[0]["type"], "text")
        self.assertEqual(payload[1]["type"], "image")
        self.assertEqual(
            fetched,
            [
                "https://example.com/1.jpg",
                "https://example.com/2.jpg",
            ],
        )


class XMonitorWatchTaskTests(unittest.TestCase):
    """
    验证 X 推文监控任务的新推文通知行为。
    """

    def test_new_post_notice_env_sends_one_text_before_media(self) -> None:
        """
        开启环境变量时，一批新推文应先发送一条关注用户文字。
        """
        events: list[tuple[str, object]] = []
        task = _XWatchTask(
            username="kana_hanaiwa",
            x_user_id="10",
            interval=60,
            limit_per_cycle=5,
            group_id=123,
            user_id=456,
            initial_since_id="100",
            fetcher=lambda since_id: [],
            formatter=lambda items, tag: "formatted",
            notify=lambda text: events.append(("text", text)),
            notify_media=lambda text, items, tag: events.append(
                ("media", [item.post_id for item in items])
            ),
        )
        items = [
            XPostResult(
                username="kana_hanaiwa",
                post_id="1",
                text="first",
                created_label="05-05 10:02",
                url="https://x.com/kana_hanaiwa/status/1",
                display_name="Kana Hanaiwa",
            ),
            XPostResult(
                username="kana_hanaiwa",
                post_id="2",
                text="second",
                created_label="05-05 10:03",
                url="https://x.com/kana_hanaiwa/status/2",
                display_name="Kana Hanaiwa",
            ),
        ]

        with mock.patch.dict("os.environ", {NEW_POST_NOTICE_ENV: "1"}):
            task._handle_new_items(items)

        self.assertEqual(
            events,
            [
                ("text", "[NEW] Kana Hanaiwa 更新了推文"),
                ("media", ["1", "2"]),
            ],
        )

    def test_new_post_notice_env_disabled_keeps_media_only(self) -> None:
        """
        未开启环境变量时，不应发送额外前置文字。
        """
        events: list[tuple[str, object]] = []
        task = _XWatchTask(
            username="kana_hanaiwa",
            x_user_id="10",
            interval=60,
            limit_per_cycle=5,
            group_id=123,
            user_id=456,
            initial_since_id="100",
            fetcher=lambda since_id: [],
            formatter=lambda items, tag: "formatted",
            notify=lambda text: events.append(("text", text)),
            notify_media=lambda text, items, tag: events.append(
                ("media", [item.post_id for item in items])
            ),
        )
        item = XPostResult(
            username="kana_hanaiwa",
            post_id="1",
            text="first",
            created_label="05-05 10:02",
            url="https://x.com/kana_hanaiwa/status/1",
            display_name="Kana Hanaiwa",
        )

        with mock.patch.dict("os.environ", {NEW_POST_NOTICE_ENV: ""}):
            task._handle_new_items([item])

        self.assertEqual(events, [("media", ["1"])])

    def test_task_waits_one_interval_before_first_poll_by_default(self) -> None:
        """
        监控任务默认应等待一个 interval 后再首次请求 timeline。
        """
        fetch_calls: list[str] = []

        def fake_fetch(since_id: str) -> list[XPostResult]:
            """
            记录增量拉取起点。

            Args:
                since_id (str): 增量拉取起点。

            Returns:
                list[XPostResult]: 空推文列表。

            Raises:
                None
            """
            fetch_calls.append(since_id)
            return []

        task = _XWatchTask(
            username="kana_hanaiwa",
            x_user_id="10",
            interval=0.2,
            limit_per_cycle=5,
            group_id=123,
            user_id=456,
            initial_since_id="100",
            fetcher=fake_fetch,
            formatter=lambda items, tag: "formatted",
            notify=lambda text: None,
        )

        task.start()
        time.sleep(0.05)
        task.stop()

        self.assertEqual(fetch_calls, [])

    def test_task_first_actual_poll_sends_new_items(self) -> None:
        """
        首次实际轮询拿到 since_id 之后的新推文时应正常发送。
        """
        fetch_calls: list[str] = []
        events: list[str] = []
        fetched = Event()

        def fake_fetch(since_id: str) -> list[XPostResult]:
            """
            返回一条比初始 since_id 更新的推文。

            Args:
                since_id (str): 增量拉取起点。

            Returns:
                list[XPostResult]: 固定新推文列表。

            Raises:
                None
            """
            fetch_calls.append(since_id)
            fetched.set()
            return [
                XPostResult(
                    username="kana_hanaiwa",
                    post_id="101",
                    text="new post",
                    created_label="05-05 10:05",
                    url="https://x.com/kana_hanaiwa/status/101",
                    display_name="Kana Hanaiwa",
                )
            ]

        task = _XWatchTask(
            username="kana_hanaiwa",
            x_user_id="10",
            interval=60,
            limit_per_cycle=5,
            group_id=123,
            user_id=456,
            initial_since_id="100",
            fetcher=fake_fetch,
            formatter=lambda items, tag: ",".join(item.post_id for item in items),
            notify=lambda text: events.append(text),
            wait_initial_interval=False,
        )

        with mock.patch.dict("os.environ", {NEW_POST_NOTICE_ENV: ""}):
            task.start()
            self.assertTrue(fetched.wait(timeout=1))
            task.stop()

        self.assertEqual(fetch_calls[:1], ["100"])
        self.assertEqual(events, ["101"])

    def test_task_state_contains_since_id_and_updates_callback(self) -> None:
        """
        任务状态应包含 since_id，并在水位线前进时触发回调。
        """
        changed: list[str] = []
        task = _XWatchTask(
            username="kana_hanaiwa",
            x_user_id="10",
            interval=60,
            limit_per_cycle=5,
            group_id=123,
            user_id=456,
            initial_since_id="100",
            fetcher=lambda since_id: [],
            formatter=lambda items, tag: "formatted",
            notify=lambda text: None,
            on_state_change=lambda: changed.append("changed"),
        )
        item = XPostResult(
            username="kana_hanaiwa",
            post_id="101",
            text="new post",
            created_label="05-05 10:05",
            url="https://x.com/kana_hanaiwa/status/101",
            display_name="Kana Hanaiwa",
        )

        self.assertEqual(task.to_state()["since_id"], "100")
        task._refresh_seen([item])

        self.assertEqual(task.to_state()["since_id"], "101")
        self.assertEqual(changed, ["changed"])

    def test_start_watch_uses_profile_since_id_without_immediate_timeline(self) -> None:
        """
        新建监控应使用用户资料里的最新推文 ID，并默认不立刻拉 timeline。
        """

        class FakeClient:
            """
            记录监控启动阶段 API 调用的客户端替身。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.profile_calls: list[str] = []
                self.profile_by_id_calls: list[str] = []
                self.post_calls: list[tuple[str, str | None]] = []

            def get_user_profile(self, username: str) -> SimpleNamespace:
                """
                返回带最新推文 ID 的固定用户资料。

                Args:
                    username (str): X 用户名。

                Returns:
                    SimpleNamespace: 用户资料。

                Raises:
                    None
                """
                self.profile_calls.append(username)
                return SimpleNamespace(
                    user_id="10",
                    username="kana_hanaiwa",
                    most_recent_tweet_id="12345",
                )

            def get_user_profile_by_id(self, user_id: str) -> SimpleNamespace:
                """
                记录不应发生的按 ID 查询。

                Args:
                    user_id (str): X 用户 ID。

                Returns:
                    SimpleNamespace: 用户资料。

                Raises:
                    AssertionError: 本测试不期望调用该方法。
                """
                self.profile_by_id_calls.append(user_id)
                raise AssertionError("不应按 ID 查询")

            def get_user_posts(
                self,
                user_id: str,
                max_results: int = DEFAULT_LIMIT,
                since_id: str | None = None,
            ) -> dict[str, object]:
                """
                记录 timeline 拉取调用。

                Args:
                    user_id (str): X 用户 ID。
                    max_results (int): API 拉取数量。
                    since_id (str | None): 增量拉取起点。

                Returns:
                    dict[str, object]: 空 timeline 响应。

                Raises:
                    None
                """
                self.post_calls.append((user_id, since_id))
                return {"data": []}

        client = FakeClient()
        manager = XMonitorManager(client=client)  # type: ignore[arg-type]

        manager.start_watch(
            "@kana_hanaiwa",
            interval=0.2,
            notify=lambda text: None,
            persist=False,
        )
        time.sleep(0.05)
        for task in list(manager._watch_tasks):
            task.stop()

        self.assertEqual(client.profile_calls, ["kana_hanaiwa"])
        self.assertEqual(client.profile_by_id_calls, [])
        self.assertEqual(client.post_calls, [])

    def test_start_watch_persists_initial_since_id(self) -> None:
        """
        新建监控写入持久化文件时应包含初始 since_id。
        """

        class FakeClient:
            """
            返回固定用户资料的客户端替身。
            """

            def get_user_profile(self, username: str) -> SimpleNamespace:
                """
                返回带最新推文 ID 的固定用户资料。

                Args:
                    username (str): X 用户名。

                Returns:
                    SimpleNamespace: 用户资料。

                Raises:
                    None
                """
                return SimpleNamespace(
                    user_id="10",
                    username=username,
                    most_recent_tweet_id="12345",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "x_monitor.json")
            manager = XMonitorManager(
                client=FakeClient(),  # type: ignore[arg-type]
                store_path=store_path,
            )

            manager.start_watch(
                "kana_hanaiwa",
                interval=60,
                notify=lambda text: None,
            )
            with open(store_path, encoding="utf-8") as f:
                records = json.loads(f.read())
            for task in list(manager._watch_tasks):
                task.stop()

        self.assertEqual(records[0]["since_id"], "12345")
        self.assertNotIn("limit_per_cycle", records[0])

    def test_restore_tasks_latest_mode_uses_user_id_profile_since_id(self) -> None:
        """
        恢复已有监控时应按 user id 获取最新推文 ID 初始化 since_id。
        """

        class FakeClient:
            """
            记录恢复监控时 API 调用的客户端替身。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.profile_calls: list[str] = []
                self.profile_by_id_calls: list[str] = []
                self.post_calls: list[tuple[str, str | None]] = []

            def get_user_profile(self, username: str) -> SimpleNamespace:
                """
                记录不应发生的用户名查询。

                Args:
                    username (str): X 用户名。

                Returns:
                    SimpleNamespace: 用户资料。

                Raises:
                    AssertionError: 本测试不期望调用该方法。
                """
                self.profile_calls.append(username)
                raise AssertionError("恢复任务不应按用户名查询")

            def get_user_profile_by_id(self, user_id: str) -> SimpleNamespace:
                """
                返回带最新推文 ID 的固定用户资料。

                Args:
                    user_id (str): X 用户 ID。

                Returns:
                    SimpleNamespace: 用户资料。

                Raises:
                    None
                """
                self.profile_by_id_calls.append(user_id)
                return SimpleNamespace(
                    user_id=user_id,
                    username="kana_hanaiwa",
                    most_recent_tweet_id="67890",
                )

            def get_user_posts(
                self,
                user_id: str,
                max_results: int = DEFAULT_LIMIT,
                since_id: str | None = None,
            ) -> dict[str, object]:
                """
                记录 timeline 拉取调用。

                Args:
                    user_id (str): X 用户 ID。
                    max_results (int): API 拉取数量。
                    since_id (str | None): 增量拉取起点。

                Returns:
                    dict[str, object]: 空 timeline 响应。

                Raises:
                    None
                """
                self.post_calls.append((user_id, since_id))
                return {"data": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "x_monitor.json")
            with open(store_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "username": "old_name",
                            "x_user_id": "10",
                            "interval": 60,
                            "group_id": 123,
                            "user_id": 456,
                            "since_id": "100",
                        }
                    ],
                    f,
                )
            client = FakeClient()
            manager = XMonitorManager(
                client=client,  # type: ignore[arg-type]
                store_path=store_path,
            )

            with mock.patch.dict("os.environ", {RESTORE_MODE_ENV: "latest"}):
                restored = manager.restore_tasks(
                    lambda group_id, user_id, rec: [lambda text: None]
                )
            states = [task.to_state() for task in manager._watch_tasks]
            for task in list(manager._watch_tasks):
                task.stop()

        self.assertEqual(restored, 1)
        self.assertEqual(client.profile_calls, [])
        self.assertEqual(client.profile_by_id_calls, ["10"])
        self.assertEqual(client.post_calls, [])
        self.assertEqual(states[0]["username"], "kana_hanaiwa")
        self.assertEqual(states[0]["since_id"], "67890")

    def test_restore_tasks_timeline_mode_advances_since_id_only(self) -> None:
        """
        timeline 恢复模式应只用旧 since_id 推进水位线，不发送停机推文。
        """

        class FakeClient:
            """
            记录 timeline 恢复调用的客户端替身。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.profile_by_id_calls: list[str] = []
                self.post_calls: list[tuple[str, str | None]] = []

            def get_user_profile_by_id(self, user_id: str) -> SimpleNamespace:
                """
                记录不应发生的 profile 查询。

                Args:
                    user_id (str): X 用户 ID。

                Returns:
                    SimpleNamespace: 用户资料。

                Raises:
                    AssertionError: 本测试不期望调用该方法。
                """
                self.profile_by_id_calls.append(user_id)
                raise AssertionError("timeline 模式不应查询用户资料")

            def get_user_posts(
                self,
                user_id: str,
                max_results: int = DEFAULT_LIMIT,
                since_id: str | None = None,
            ) -> dict[str, object]:
                """
                返回停机期间的新推文用于推进 since_id。

                Args:
                    user_id (str): X 用户 ID。
                    max_results (int): API 拉取数量。
                    since_id (str | None): 增量拉取起点。

                Returns:
                    dict[str, object]: X API 风格响应。

                Raises:
                    None
                """
                self.post_calls.append((user_id, since_id))
                return {
                    "data": [
                        {"id": "101", "text": "old downtime post"},
                        {"id": "105", "text": "new downtime post"},
                    ]
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "x_monitor.json")
            with open(store_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "username": "kana_hanaiwa",
                            "x_user_id": "10",
                            "interval": 60,
                            "group_id": 123,
                            "user_id": 456,
                            "since_id": "100",
                        }
                    ],
                    f,
                )
            client = FakeClient()
            events: list[str] = []
            manager = XMonitorManager(
                client=client,  # type: ignore[arg-type]
                store_path=store_path,
            )

            with mock.patch.dict("os.environ", {RESTORE_MODE_ENV: "timeline"}):
                restored = manager.restore_tasks(
                    lambda group_id, user_id, rec: [events.append]
                )
            states = [task.to_state() for task in manager._watch_tasks]
            for task in list(manager._watch_tasks):
                task.stop()

        self.assertEqual(restored, 1)
        self.assertEqual(client.profile_by_id_calls, [])
        self.assertEqual(client.post_calls, [("10", "100")])
        self.assertEqual(states[0]["since_id"], "105")
        self.assertEqual(events, [])

    def test_restore_tasks_timeline_mode_keeps_empty_since_id(self) -> None:
        """
        timeline 恢复模式无新推文时应沿用旧 since_id。
        """

        class FakeClient:
            """
            返回空 timeline 响应的客户端替身。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.post_calls: list[tuple[str, str | None]] = []

            def get_user_posts(
                self,
                user_id: str,
                max_results: int = DEFAULT_LIMIT,
                since_id: str | None = None,
            ) -> dict[str, object]:
                """
                返回空 timeline 响应。

                Args:
                    user_id (str): X 用户 ID。
                    max_results (int): API 拉取数量。
                    since_id (str | None): 增量拉取起点。

                Returns:
                    dict[str, object]: 空 X API 响应。

                Raises:
                    None
                """
                self.post_calls.append((user_id, since_id))
                return {"data": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "x_monitor.json")
            with open(store_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "username": "kana_hanaiwa",
                            "x_user_id": "10",
                            "interval": 60,
                            "group_id": 123,
                            "user_id": 456,
                            "since_id": "100",
                        }
                    ],
                    f,
                )
            client = FakeClient()
            manager = XMonitorManager(
                client=client,  # type: ignore[arg-type]
                store_path=store_path,
            )

            with mock.patch.dict("os.environ", {RESTORE_MODE_ENV: "timeline"}):
                restored = manager.restore_tasks(
                    lambda group_id, user_id, rec: [lambda text: None]
                )
            states = [task.to_state() for task in manager._watch_tasks]
            for task in list(manager._watch_tasks):
                task.stop()

        self.assertEqual(restored, 1)
        self.assertEqual(client.post_calls, [("10", "100")])
        self.assertEqual(states[0]["since_id"], "100")

    def test_restore_tasks_timeline_mode_persists_advanced_since_id(self) -> None:
        """
        恢复任务时 timeline 模式推进的 since_id 应写回持久化文件。
        """

        class FakeClient:
            """
            为恢复任务返回停机期间新推文的客户端替身。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.post_calls: list[tuple[str, str | None]] = []

            def get_user_posts(
                self,
                user_id: str,
                max_results: int = DEFAULT_LIMIT,
                since_id: str | None = None,
            ) -> dict[str, object]:
                """
                返回停机期间的新推文。

                Args:
                    user_id (str): X 用户 ID。
                    max_results (int): API 拉取数量。
                    since_id (str | None): 增量拉取起点。

                Returns:
                    dict[str, object]: X API 风格响应。

                Raises:
                    None
                """
                self.post_calls.append((user_id, since_id))
                return {"data": [{"id": "108", "text": "downtime post"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "x_monitor.json")
            with open(store_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "username": "kana_hanaiwa",
                            "x_user_id": "10",
                            "interval": 60,
                            "group_id": 123,
                            "user_id": 456,
                            "since_id": "100",
                        }
                    ],
                    f,
                )
            client = FakeClient()
            manager = XMonitorManager(
                client=client,  # type: ignore[arg-type]
                store_path=store_path,
            )

            with mock.patch.dict("os.environ", {RESTORE_MODE_ENV: "timeline"}):
                restored = manager.restore_tasks(
                    lambda group_id, user_id, rec: [lambda text: None]
                )
            with open(store_path, encoding="utf-8") as f:
                records = json.loads(f.read())
            for task in list(manager._watch_tasks):
                task.stop()

        self.assertEqual(restored, 1)
        self.assertEqual(client.post_calls, [("10", "100")])
        self.assertEqual(records[0]["since_id"], "108")

    def test_start_watch_requires_most_recent_tweet_id(self) -> None:
        """
        用户资料缺少最新推文 ID 时应直接失败，不回退到 timeline 初始化。
        """

        class FakeClient:
            """
            返回缺少最新推文 ID 的用户资料。
            """

            def __init__(self) -> None:
                """
                初始化调用记录。

                Args:
                    None

                Returns:
                    None

                Raises:
                    None
                """
                self.post_calls: list[str] = []

            def get_user_profile(self, username: str) -> SimpleNamespace:
                """
                返回没有最新推文 ID 的用户资料。

                Args:
                    username (str): X 用户名。

                Returns:
                    SimpleNamespace: 用户资料。

                Raises:
                    None
                """
                return SimpleNamespace(
                    user_id="10",
                    username=username,
                    most_recent_tweet_id="",
                )

            def get_user_posts(
                self,
                user_id: str,
                max_results: int = DEFAULT_LIMIT,
                since_id: str | None = None,
            ) -> dict[str, object]:
                """
                记录不应发生的 timeline 拉取。

                Args:
                    user_id (str): X 用户 ID。
                    max_results (int): API 拉取数量。
                    since_id (str | None): 增量拉取起点。

                Returns:
                    dict[str, object]: 空 timeline 响应。

                Raises:
                    None
                """
                self.post_calls.append(user_id)
                return {"data": []}

        client = FakeClient()
        manager = XMonitorManager(client=client)  # type: ignore[arg-type]

        with self.assertRaises(AssertionError):
            manager.start_watch(
                "kana_hanaiwa",
                interval=60,
                notify=lambda text: None,
                persist=False,
            )

        self.assertEqual(client.post_calls, [])


class QQBotXMonitorCommandTests(unittest.TestCase):
    """
    验证 QQBot 中隐藏的 X 推文监控测试命令。
    """

    def test_xtrans_command_cycles_translation_mode(self) -> None:
        """
        /xtrans 应循环切换 XMonitor 翻译模式环境变量。
        """
        handler = QQBotHandler.__new__(QQBotHandler)
        handler.bot_cfg = SimpleNamespace(
            api_base="http://onebot",
            access_token="token",
            cmd_allowed_users=(),
        )

        with mock.patch.dict("os.environ", {TRANSLATION_MODE_ENV: "none"}):
            with mock.patch.object(qq_group_bot, "_send_group_msg") as send_text:
                handled = handler._handle_commands(123, 456, "/xtrans")
                handled_second = handler._handle_commands(123, 456, "/xtrans")
                handled_third = handler._handle_commands(123, 456, "/xtrans")

            self.assertTrue(handled)
            self.assertTrue(handled_second)
            self.assertTrue(handled_third)
            self.assertEqual(os.environ[TRANSLATION_MODE_ENV], "none")
            self.assertEqual(send_text.call_count, 3)
            self.assertIn("不翻译 -> 仅翻译", send_text.call_args_list[0].args[2])
            self.assertIn("仅翻译 -> 对照", send_text.call_args_list[1].args[2])
            self.assertIn("对照 -> 不翻译", send_text.call_args_list[2].args[2])

    def test_hidden_test_command_sends_latest_tweet_with_media(self) -> None:
        """
        /xmonitor 用户 test 应拉取最新一条推文并走图文发送路径。
        """
        handler = QQBotHandler.__new__(QQBotHandler)
        old_cfg = getattr(QQBotHandler, "bot_cfg", None)
        old_monitor = getattr(QQBotHandler, "x_monitor", None)
        monitor = FakeXMonitorManager()
        sent: list[dict[str, object]] = []

        def fake_send_x_message_with_images(
            api_base: str,
            group_id: int,
            access_token: str,
            text: str,
            items: list[XPostResult],
        ) -> None:
            """
            记录 X 图文发送参数。

            Args:
                api_base (str): OneBot API 基地址。
                group_id (int): 群号。
                access_token (str): API Token。
                text (str): 文本内容。
                items (list[XPostResult]): 推文列表。

            Returns:
                None

            Raises:
                None
            """
            sent.append(
                {
                    "api_base": api_base,
                    "group_id": group_id,
                    "access_token": access_token,
                    "text": text,
                    "items": items,
                }
            )

        try:
            QQBotHandler.bot_cfg = SimpleNamespace(
                api_base="http://onebot", access_token="token"
            )
            QQBotHandler.x_monitor = monitor
            with mock.patch.object(
                qq_group_bot,
                "send_x_message_with_images",
                side_effect=fake_send_x_message_with_images,
            ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_text:
                handled = handler._handle_x_monitor_commands(
                    123, 456, ["/xmonitor", "@kana_hanaiwa", "test"]
                )
            self.assertTrue(handled)
            self.assertEqual(monitor.latest_calls, [("@kana_hanaiwa", 1)])
            send_text.assert_not_called()
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0]["api_base"], "http://onebot")
            self.assertEqual(sent[0]["group_id"], 123)
            self.assertIn("[X NEW] | @kana_hanaiwa", str(sent[0]["text"]))
            self.assertIn("latest post", str(sent[0]["text"]))
            self.assertEqual(len(sent[0]["items"]), 1)
        finally:
            QQBotHandler.x_monitor = old_monitor
            if old_cfg is not None:
                QQBotHandler.bot_cfg = old_cfg

    def test_xlink_command_sends_target_tweet_with_media(self) -> None:
        """
        /xlink 推文链接应拉取目标推文并走图文发送路径。
        """
        handler = QQBotHandler.__new__(QQBotHandler)
        old_cfg = getattr(QQBotHandler, "bot_cfg", None)
        old_monitor = getattr(QQBotHandler, "x_monitor", None)
        monitor = FakeXMonitorManager()
        sent: list[dict[str, object]] = []

        def fake_send_x_message_with_images(
            api_base: str,
            group_id: int,
            access_token: str,
            text: str,
            items: list[XPostResult],
        ) -> None:
            """
            记录 X 图文发送参数。

            Args:
                api_base (str): OneBot API 基地址。
                group_id (int): 群号。
                access_token (str): API Token。
                text (str): 文本内容。
                items (list[XPostResult]): 推文列表。

            Returns:
                None

            Raises:
                None
            """
            sent.append(
                {
                    "api_base": api_base,
                    "group_id": group_id,
                    "access_token": access_token,
                    "text": text,
                    "items": items,
                }
            )

        try:
            QQBotHandler.bot_cfg = SimpleNamespace(
                api_base="http://onebot", access_token="token"
            )
            QQBotHandler.x_monitor = monitor
            with mock.patch.object(
                qq_group_bot,
                "send_x_message_with_images",
                side_effect=fake_send_x_message_with_images,
            ), mock.patch.object(qq_group_bot, "_send_group_msg") as send_text:
                handled = handler._handle_commands(
                    123,
                    456,
                    "/xlink https://x.com/kana_hanaiwa/status/1?s=20",
                )

            self.assertTrue(handled)
            self.assertEqual(
                monitor.fetch_link_calls,
                ["https://x.com/kana_hanaiwa/status/1?s=20"],
            )
            send_text.assert_not_called()
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0]["api_base"], "http://onebot")
            self.assertEqual(sent[0]["group_id"], 123)
            self.assertIn("[X LINK] | @kana_hanaiwa", str(sent[0]["text"]))
            self.assertIn("linked post", str(sent[0]["text"]))
            self.assertEqual(len(sent[0]["items"]), 1)
        finally:
            QQBotHandler.x_monitor = old_monitor
            if old_cfg is not None:
                QQBotHandler.bot_cfg = old_cfg

    def test_xlink_command_reports_usage_when_url_missing(self) -> None:
        """
        /xlink 缺少链接时应返回用法提示。
        """
        handler = QQBotHandler.__new__(QQBotHandler)
        old_cfg = getattr(QQBotHandler, "bot_cfg", None)
        old_monitor = getattr(QQBotHandler, "x_monitor", None)
        monitor = FakeXMonitorManager()

        try:
            QQBotHandler.bot_cfg = SimpleNamespace(
                api_base="http://onebot", access_token="token"
            )
            QQBotHandler.x_monitor = monitor
            with mock.patch.object(qq_group_bot, "_send_group_msg") as send_text:
                handled = handler._handle_commands(123, 456, "/xlink")

            self.assertTrue(handled)
            self.assertEqual(monitor.fetch_link_calls, [])
            self.assertIn("用法: /xlink 推文链接", send_text.call_args.args[2])
        finally:
            QQBotHandler.x_monitor = old_monitor
            if old_cfg is not None:
                QQBotHandler.bot_cfg = old_cfg


if __name__ == "__main__":
    unittest.main()
