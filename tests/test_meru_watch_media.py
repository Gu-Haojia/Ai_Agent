"""
Meru 监控图片抓取与消息拼装单元测试。
"""

from __future__ import annotations

import base64
import io
import unittest
from unittest import mock

from PIL import Image

from src.meru_monitor import MeruMonitorManager, MeruSearchResult, _collect_image_urls
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

    def test_watch_text_omits_links_and_search_keeps_links(self) -> None:
        """监控消息应直接移除链接行，搜索结果仍保留商品链接。"""
        item = MeruSearchResult(
            keyword="k",
            item_id="1",
            name="Item1",
            price=100,
            created_label="01-01 00:00",
            url="https://example.com/1",
            previous_price=200,
        )
        for tag in ("NEW", "PRICE<= 200", "PRICE_DROP", "PRICE_DROP<= 200"):
            with self.subTest(tag=tag):
                text = MeruMonitorManager.format_lines([item], tag)
                self.assertNotIn(item.url, text)
                self.assertNotIn("链接：", text)
                self.assertNotIn("二维码", text)
                self.assertNotIn("扫码", text)
                self.assertIn("Item1", text)
                self.assertIn("价格：¥100", text)
                self.assertIn("时间：01-01 00:00", text)
        self.assertIn(item.url, MeruMonitorManager.format_lines([item], "SEARCH"))

    def test_compose_with_images_and_at(self) -> None:
        """
        共用图片的不同商品仍应各自生成二维码图片，并保留 @ 和文本。
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
                image_urls=("https://example.com/img1.jpg",),
            ),
        ]

        source = io.BytesIO()
        Image.new("RGB", (300, 300), (200, 220, 240)).save(source, format="PNG")
        fetch = mock.Mock(return_value=source.getvalue())

        payload = compose_meru_media_message(
            "hello", items, at_qq=10000, fetcher=fetch, max_images=2
        )
        self.assertIsInstance(payload, list)
        self.assertEqual(payload[0]["type"], "at")
        self.assertEqual(payload[0]["data"]["qq"], "10000")
        self.assertEqual(payload[1]["type"], "text")
        self.assertEqual(payload[1]["data"]["text"], "hello")
        self.assertEqual(len(payload), 4)
        self.assertEqual(
            fetch.call_args_list,
            [mock.call("https://example.com/img1.jpg")] * 2,
        )
        self.assertNotEqual(payload[2]["data"]["file"], payload[3]["data"]["file"])
        for segment in payload[2:]:
            self.assertEqual(segment["type"], "image")
            self.assertTrue(segment["data"]["name"].endswith(".jpg"))
            image_data = base64.b64decode(segment["data"]["file"].removeprefix("base64://"))
            with Image.open(io.BytesIO(image_data)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (640, 640))
                for actual, expected in zip(image.getpixel((0, 0)), (200, 220, 240)):
                    self.assertLessEqual(abs(actual - expected), 2)
                self.assertEqual(image.convert("L").getextrema(), (0, 255))

    def test_compose_without_images_raises_before_download(self) -> None:
        """
        缺少商品图时应明确报错，避免发送没有访问入口的商品。
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

        fetch = mock.Mock(side_effect=AssertionError("不应触发下载"))
        with self.assertRaisesRegex(AssertionError, "缺少图片"):
            compose_meru_media_message("plain", items, fetcher=fetch)
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
