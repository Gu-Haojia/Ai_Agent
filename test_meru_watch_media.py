"""
Meru 监控图片抓取与消息拼装的单元测试。
"""

from __future__ import annotations

import unittest

from qq_group_bot import _compose_meru_watch_message
from src.meru_monitor import MeruSearchResult, _collect_image_urls


class MeruMonitorImageExtractionTests(unittest.TestCase):
    """
    验证图片 URL 抽取逻辑的健壮性。

    Args:
        unittest.TestCase: 单元测试基类。

    Returns:
        None

    Raises:
        None
    """

    def test_collect_image_urls_prefers_thumbnails_then_photos(self) -> None:
        """
        确保优先使用缩略图，其次补充 photos 字段。

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: 当抽取结果不符合预期时抛出。
        """

        item = {
            "thumbnails": [
                "https://example.com/thumb1.webp",
                "https://example.com/thumb2.webp",
            ],
            "photos": [
                {"uri": "https://example.com/photo1.jpg"},
            ],
        }
        urls = _collect_image_urls(item, limit=3)
        self.assertEqual(
            urls,
            (
                "https://example.com/thumb1.webp",
                "https://example.com/thumb2.webp",
                "https://example.com/photo1.jpg",
            ),
        )

    def test_collect_image_urls_filters_invalid_and_limits(self) -> None:
        """
        确保会过滤非法 URL 并遵守数量上限。

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: 当返回值未过滤或未裁剪时抛出。
        """

        item = {
            "thumbnails": [
                "",
                "https://example.com/valid1.jpg",
                "https://example.com/valid1.jpg",
            ],
            "photos": [
                "invalid",
                {"uri": "https://example.com/valid2.jpg"},
                {"url": "https://example.com/valid3.jpg"},
            ],
        }
        urls = _collect_image_urls(item, limit=2)
        self.assertEqual(
            urls,
            (
                "https://example.com/valid1.jpg",
                "https://example.com/valid2.jpg",
            ),
        )


class MeruWatchMessageComposeTests(unittest.TestCase):
    """
    验证 Meru 新品通知消息的拼装与附图行为。

    Args:
        unittest.TestCase: 单元测试基类。

    Returns:
        None

    Raises:
        None
    """

    def test_compose_meru_watch_message_with_at_and_images(self) -> None:
        """
        当提供图片与 @ 对象时，应生成文本 + 图片段。

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: 当消息段缺失或数量不符时抛出。
        """

        items = [
            MeruSearchResult(
                keyword="k",
                item_id="1",
                name="Item1",
                price=100,
                created_label="01-01 00:00",
                url="https://example.com/1",
                image_urls=("https://example.com/img1.jpg",),
            ),
            MeruSearchResult(
                keyword="k",
                item_id="2",
                name="Item2",
                price=200,
                created_label="01-01 00:01",
                url="https://example.com/2",
                image_urls=(
                    "https://example.com/img2.jpg",
                    "https://example.com/img3.jpg",
                ),
            ),
        ]

        def fake_fetcher(url: str) -> tuple[str, str]:
            return ("Zg==", "image/png")

        payload = _compose_meru_watch_message(
            "hello", items, at_qq=123456, fetcher=fake_fetcher, max_images=2
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 4)
        self.assertEqual(payload[0]["type"], "at")
        self.assertEqual(payload[0]["data"]["qq"], "123456")
        self.assertEqual(payload[1]["type"], "text")
        self.assertEqual(payload[1]["data"]["text"], "hello")
        self.assertEqual(payload[2]["type"], "image")
        self.assertTrue(payload[2]["data"]["file"].startswith("base64://Zg=="))
        self.assertTrue(payload[2]["data"]["name"].endswith(".png"))
        self.assertEqual(payload[3]["type"], "image")

    def test_compose_meru_watch_message_without_images_returns_text_segment(
        self,
    ) -> None:
        """
        当没有可用图片时，仅发送文本段。

        Args:
            None

        Returns:
            None

        Raises:
            AssertionError: 当仍试图附加图片或文本缺失时抛出。
        """

        items = [
            MeruSearchResult(
                keyword="k",
                item_id="3",
                name="Item3",
                price=None,
                created_label="01-01 00:02",
                url="https://example.com/3",
                image_urls=(),
            )
        ]

        def fail_fetcher(url: str) -> tuple[str, str]:
            raise AssertionError(f"fetcher should not be called for {url}")

        payload = _compose_meru_watch_message(
            "plain text", items, at_qq=None, fetcher=fail_fetcher
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["type"], "text")
        self.assertEqual(payload[0]["data"]["text"], "plain text")


if __name__ == "__main__":
    unittest.main()
