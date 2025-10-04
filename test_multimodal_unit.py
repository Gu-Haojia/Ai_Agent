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
            self.assertTrue(
                any(
                    item.get("type") == "text" and "stored.png" in item.get("text", "")
                    for item in content
                )
            )

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")
    def test_compose_group_message_appends_cq_codes(self) -> None:
        message = QQBotHandler._compose_group_message(
            "hello", [("ZGF0YQ==", "image/png")]
        )
        self.assertIn("hello", message)
        self.assertIn("[CQ:image,file=base64://", message)

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

    def test_save_base64_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            data = base64.b64encode(b"test").decode("ascii")
            stored = manager.save_base64_image(data, "image/jpeg")
            self.assertTrue(stored.path.exists())
            self.assertTrue(stored.base64_data)

    def test_load_stored_image_by_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            data = base64.b64encode(b"reference").decode("ascii")
            stored = manager.save_base64_image(data, "image/png")
            loaded = manager.load_stored_image(stored.path.name)
            self.assertEqual(loaded.path, stored.path)
            self.assertEqual(loaded.mime_type, "image/png")
            self.assertTrue(loaded.base64_data)
            with self.assertRaises(AssertionError):
                manager.load_stored_image("missing.png")

    def test_generate_url_candidates_prefers_twitter_orig(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            url = "https://pbs.twimg.com/profile_images/12345/test_400x400.jpg"
            candidates = manager._generate_url_candidates(url)
            self.assertTrue(any("name=orig" in c for c in candidates[:3]))
            self.assertEqual(candidates[-1], url)

    def test_generate_url_candidates_strip_imageview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            url = "https://example.com/image.png?imageView=1&thumbnail=400x400"
            candidates = manager._generate_url_candidates(url)
            self.assertTrue(any("imageView" not in c and "thumbnail" not in c for c in candidates[:-1]))

    def test_is_generated_path_handles_generated_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            png_base64 = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wwAAuMB9oK08QAAAABJRU5ErkJggg=="
            )
            generated = manager.save_generated_image(png_base64, "prompt", "image/png")
            self.assertTrue(manager.is_generated_path(str(generated.path)))
            self.assertTrue(manager.is_generated_path(f"file://{generated.path}"))
            other_path = Path(tmp_dir) / "outer.png"
            other_path.write_bytes(base64.b64decode(png_base64))
            self.assertFalse(manager.is_generated_path(str(other_path)))

    def test_normalize_aspect_ratio_accepts_alias(self) -> None:
        ratio = ImageStorageManager._normalize_aspect_ratio(None, "square")
        self.assertEqual(ratio, "1:1")

    def test_normalize_aspect_ratio_from_size(self) -> None:
        ratio = ImageStorageManager._normalize_aspect_ratio(None, "800x600")
        self.assertEqual(ratio, "800:600")

    def test_normalize_aspect_ratio_invalid_size(self) -> None:
        with self.assertRaises(AssertionError):
            ImageStorageManager._normalize_aspect_ratio(None, "invalid")


if __name__ == "__main__":
    unittest.main()
