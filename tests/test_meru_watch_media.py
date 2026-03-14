"""
Meru 监控图片抓取与消息拼装单元测试。
"""

from __future__ import annotations

import unittest

from src.meru_monitor import MeruSearchResult, _collect_image_urls
from src.meru_watch_media import compose_meru_media_message


class MeruImageExtractTests(unittest.TestCase):
    """
    验证接口返回的图片 URL 抽取逻辑。
    """

    def test_collect_image_urls_prefers_thumbnails(self) -> None:
        """
        应优先返回 thumbnails，数量不足再补 photos。
        """

        item = {
            "thumbnails": [
                "https://example.com/t1.webp",
                "https://example.com/t2.webp",
            ],
            "photos": [{"uri": "https://example.com/p1.jpg"}],
        }
        urls = _collect_image_urls(item, limit=3)
        self.assertEqual(
            urls,
            (
                "https://example.com/t1.webp",
                "https://example.com/t2.webp",
                "https://example.com/p1.jpg",
            ),
        )

    def test_collect_image_urls_filters_invalid(self) -> None:
        """
        应过滤空字符串与非 http 链接，并遵守上限。
        """

        item = {
            "thumbnails": ["", "https://example.com/valid1.jpg", "https://example.com/valid1.jpg"],
            "photos": ["invalid", {"uri": "https://example.com/valid2.jpg"}],
        }
        urls = _collect_image_urls(item, limit=2)
        self.assertEqual(
            urls,
            (
                "https://example.com/valid1.jpg",
                "https://example.com/valid2.jpg",
            ),
        )


class MeruMediaComposeTests(unittest.TestCase):
    """
    验证拼装 OneBot 消息段的行为。
    """

    def test_compose_with_images_and_at(self) -> None:
        """
        提供图片与 @ 时，应生成 at + text + image 段。
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
                image_urls=("https://example.com/img2.jpg",),
            ),
        ]

        def fake_fetch(url: str) -> tuple[str, str]:
            return ("Zg==", "image/png")

        payload = compose_meru_media_message(
            "hello", items, at_qq=10000, fetcher=fake_fetch, max_images=2
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(payload[0]["type"], "at")
        self.assertEqual(payload[0]["data"]["qq"], "10000")
        self.assertEqual(payload[1]["type"], "text")
        self.assertEqual(len(payload), 4)
        self.assertTrue(
            payload[2]["data"]["file"].startswith("base64://Zg==")
        )

    def test_compose_without_images_returns_text_only(self) -> None:
        """
        无图片时只返回文本段。
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

        def fail_fetch(url: str) -> tuple[str, str]:
            raise AssertionError("不应触发下载")

        payload = compose_meru_media_message(
            "plain", items, at_qq=None, fetcher=fail_fetch
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["type"], "text")
        self.assertEqual(payload[0]["data"]["text"], "plain")


if __name__ == "__main__":
    unittest.main()
