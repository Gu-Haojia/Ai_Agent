"""
X 推文监控解析与图文消息拼装单元测试。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.x_monitor import (
    DEFAULT_LIMIT,
    NEW_POST_NOTICE_ENV,
    XMonitorManager,
    XPostResult,
    _collect_post_image_urls,
    _normalize_username,
    _XWatchTask,
)
from src.x_monitor_media import _send_group_msg, compose_x_media_message
from src.x_monitor_render import (
    BrowserRenderConfig,
    BrowserTweetRenderer,
    XTweetPayloadParser,
    render_tweet_html,
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


class XMonitorParseTests(unittest.TestCase):
    """
    验证 X API 响应解析逻辑。
    """

    def test_normalize_username_accepts_at_prefix(self) -> None:
        """
        用户名允许带 @ 前缀，并会被清理为裸用户名。
        """
        self.assertEqual(_normalize_username("@kana_hanaiwa"), "kana_hanaiwa")

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
        self.assertRegex(results[0].created_label, r"\d{2}-\d{2} \d{2}:\d{2}")

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
                return SimpleNamespace(user_id="10", username=username.lstrip("@"))

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
            '<div class="agent-footer">本推文由筱泽广Agent提供</div>',
            html,
        )
        self.assertIn(".agent-footer", html)

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
                ("text", "关注的Kana Hanaiwa发表了新的推文。"),
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


class QQBotXMonitorCommandTests(unittest.TestCase):
    """
    验证 QQBot 中隐藏的 X 推文监控测试命令。
    """

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


if __name__ == "__main__":
    unittest.main()
