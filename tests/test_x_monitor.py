"""
X 推文监控解析与图文消息拼装单元测试。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.x_monitor import (
    XMonitorManager,
    XPostResult,
    _collect_post_image_urls,
    _normalize_username,
)
from src.x_monitor_media import compose_x_media_message
import qq_group_bot
from qq_group_bot import QQBotHandler


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
                    "created_at": "2026-05-05T01:02:03.000Z",
                    "attachments": {"media_keys": ["3_1"]},
                }
            ],
            "includes": {
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
        self.assertEqual(
            results[0].url,
            "https://x.com/kana_hanaiwa/status/1919610000000000001",
        )
        self.assertRegex(results[0].created_label, r"\d{2}-\d{2} \d{2}:\d{2}")


class XMonitorMediaComposeTests(unittest.TestCase):
    """
    验证 X 推文 OneBot 图文消息拼装。
    """

    def test_compose_with_images(self) -> None:
        """
        有图片时应生成 text + image 消息段。
        """
        items = [
            XPostResult(
                username="kana_hanaiwa",
                post_id="1",
                text="post",
                created_label="05-05 10:02",
                url="https://x.com/kana_hanaiwa/status/1",
                image_urls=("https://example.com/1.jpg",),
            )
        ]

        def fake_fetch(url: str) -> tuple[str, str]:
            """
            返回固定图片内容。

            Args:
                url (str): 图片 URL。

            Returns:
                tuple[str, str]: base64 与 MIME。

            Raises:
                None: 本函数不抛出异常。
            """
            return "Zg==", "image/webp"

        payload = compose_x_media_message(
            "hello", items, fetcher=fake_fetch, max_images=1
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["type"], "text")
        self.assertEqual(payload[1]["type"], "image")
        self.assertTrue(payload[1]["data"]["file"].startswith("base64://Zg=="))

    def test_compose_sends_all_images_from_same_post(self) -> None:
        """
        同一条推文包含多张图时，应全部生成 image 段。
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
                    "https://example.com/3.jpg",
                    "https://example.com/4.jpg",
                ),
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

        payload = compose_x_media_message("hello", items, fetcher=fake_fetch)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 5)
        self.assertEqual(
            [segment["type"] for segment in payload],
            ["text", "image", "image", "image", "image"],
        )
        self.assertEqual(
            fetched,
            [
                "https://example.com/1.jpg",
                "https://example.com/2.jpg",
                "https://example.com/3.jpg",
                "https://example.com/4.jpg",
            ],
        )

    def test_compose_without_images_returns_text_segment(self) -> None:
        """
        无图片时应只生成文本消息段。
        """
        items = [
            XPostResult(
                username="kana_hanaiwa",
                post_id="2",
                text="plain",
                created_label="05-05 10:03",
                url="https://x.com/kana_hanaiwa/status/2",
                image_urls=(),
            )
        ]

        def fail_fetch(url: str) -> tuple[str, str]:
            """
            禁止在无图片场景触发下载。

            Args:
                url (str): 图片 URL。

            Returns:
                tuple[str, str]: 永远不会返回。

            Raises:
                AssertionError: 一旦被调用即抛出。
            """
            raise AssertionError("不应触发下载")

        payload = compose_x_media_message("plain", items, fetcher=fail_fetch)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["type"], "text")
        self.assertEqual(payload[0]["data"]["text"], "plain")


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
