import base64
import tempfile
import unittest
from pathlib import Path

from image_storage import ImageStorageManager, StoredImage

try:
    from qq_group_bot import QQBotHandler, _extract_cq_images, _parse_message_and_at

    _QQ_MODULE_AVAILABLE = True
except ModuleNotFoundError:
    QQBotHandler = None
    _extract_cq_images = None
    _parse_message_and_at = None
    _QQ_MODULE_AVAILABLE = False


class MultimodalUnitTest(unittest.TestCase):
    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")
    def test_extract_cq_images_parses_multiple_segments(self) -> None:
        raw = (
            "[CQ:at,qq=10086][CQ:image,file=foo.png,url=http://example.com/foo.png]"
            "some text"
            "[CQ:image,file=bar.jpg,url=https://example.com/bar.jpg]"
        )
        images = _extract_cq_images(raw)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].url, "http://example.com/foo.png")
        self.assertEqual(images[0].filename, "foo.png")
        self.assertEqual(images[1].url, "https://example.com/bar.jpg")

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")
    def test_parse_message_and_at_handles_array_with_image(self) -> None:
        event = {
            "self_id": 20000,
            "message": [
                {"type": "at", "data": {"qq": "20000"}},
                {"type": "text", "data": {"text": "请看看"}},
                {
                    "type": "image",
                    "data": {
                        "url": "https://example.com/test.png",
                        "file": "test.png",
                        "name": "test.png",
                    },
                },
            ],
        }
        parsed = _parse_message_and_at(event)
        self.assertTrue(parsed.at_me)
        self.assertEqual(parsed.text, "请看看")
        self.assertEqual(len(parsed.images), 1)
        self.assertEqual(parsed.images[0].url, "https://example.com/test.png")

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")
    def test_multimodal_content_builder_includes_text_and_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stored = StoredImage(
                path=Path(tmp_dir) / "stored.png",
                mime_type="image/png",
                base64_data=base64.b64encode(b"dummy").decode("ascii"),
            )
            content = QQBotHandler._build_multimodal_content("test", [stored])
            self.assertEqual(content[0]["text"], "test")
            self.assertTrue(any(item.get("type") == "image_url" for item in content))

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")
    def test_compose_group_message_appends_cq_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "gen.png"
            image_path.write_bytes(b"img")
            message = QQBotHandler._compose_group_message("hello", [image_path])
            self.assertIn("[CQ:image,file=file://", message)
            self.assertTrue(message.startswith("hello"))

    def test_save_generated_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            png_base64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wwAAuMB9oK08QAAAABJRU5ErkJggg=="
            )
            result = manager.save_generated_image(png_base64, "prompt", "image/png")
            self.assertTrue(result.path.exists())
            self.assertEqual(result.path.suffix, ".png")
            self.assertEqual(result.mime_type, "image/png")


if __name__ == "__main__":
    unittest.main()
