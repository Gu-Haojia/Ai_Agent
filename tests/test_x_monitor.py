"""
X 推文监控解析与图文消息拼装单元测试。
"""

from __future__ import annotations

import unittest

from src.x_monitor import (
    XMonitorManager,
    XPostResult,
    _collect_post_image_urls,
    _normalize_username,
)
from src.x_monitor_media import compose_x_media_message


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


if __name__ == "__main__":
    unittest.main()
