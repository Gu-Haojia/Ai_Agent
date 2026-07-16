import base64
from io import BytesIO
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import image_storage
from image_storage import ImageStorageManager, StoredImage, StoredVideo
from PIL import Image

try:
    from qq_group_bot import QQBotHandler, _extract_cq_images, _parse_message_and_at

    _QQ_MODULE_AVAILABLE = True
except ModuleNotFoundError:
    QQBotHandler = None
    _extract_cq_images = None
    _parse_message_and_at = None
    _QQ_MODULE_AVAILABLE = False


class MultimodalUnitTest(unittest.TestCase):
    @staticmethod
    def _make_image_base64(width: int, height: int, image_format: str) -> str:
        """
        生成测试用图片 Base64 字符串。

        Args:
            width (int): 图片宽度。
            height (int): 图片高度。
            image_format (str): Pillow 图片格式，例如 ``PNG`` 或 ``JPEG``。

        Returns:
            str: 图片二进制内容的 Base64 字符串。

        Raises:
            AssertionError: 当尺寸或格式非法时抛出。
        """
        assert width > 0 and height > 0, "测试图片尺寸必须为正数"
        assert image_format.strip(), "测试图片格式不能为空"
        buffer = BytesIO()
        Image.new("RGB", (width, height), color=(128, 160, 192)).save(
            buffer, format=image_format
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

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
    def test_multimodal_content_builder_includes_datetime_reminder(self) -> None:
        """
        动态时间提醒应作为当前用户消息中的额外文本块。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 断言失败时由 unittest 报告。
        """
        content = QQBotHandler._build_multimodal_content(
            "用户原消息",
            [],
            include_datetime_system_reminder=True,
        )

        self.assertEqual(len(content), 2)
        self.assertEqual(content[0], {"type": "text", "text": "用户原消息"})
        self.assertTrue(
            re.fullmatch(
                r"<system_reminder>Current datetime: "
                r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} \(JST\), "
                r"Weekday: [A-Za-z]+</system_reminder>",
                str(content[1]["text"]),
            )
        )
        self.assertEqual(
            content[1]["type"],
            "text",
        )

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 解析逻辑测试")
    def test_format_video_part_preserves_duration_for_token_count(self) -> None:
        """
        视频消息块应保存时长，同时保持 Gemini media 输入结构。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 断言失败时由 unittest 报告。
        """
        stored = StoredVideo(
            path=Path("video.mp4"),
            mime_type="video/mp4",
            base64_data=base64.b64encode(b"video").decode("ascii"),
            duration_seconds=12.4,
        )

        part = QQBotHandler._format_video_part(
            stored, "google_genai:gemini-3.5-flash"
        )

        self.assertEqual(part["type"], "media")
        self.assertEqual(part["mime_type"], "video/mp4")
        self.assertEqual(part["data"], b"video")
        self.assertEqual(part["duration_seconds"], 12.4)

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

    def test_infer_mime_prefers_detected_image_type(self) -> None:
        """应通过文件头识别真实图片类型并覆盖错误的备用 MIME。"""

        image_data = base64.b64decode(self._make_image_base64(1, 1, "JPEG"))

        result = ImageStorageManager._infer_mime(image_data, "image/png")

        self.assertEqual(result, "image/jpeg")

    def test_infer_mime_uses_fallback_for_unknown_data(self) -> None:
        """无法从文件头识别图片时应继续使用调用方提供的备用 MIME。"""

        result = ImageStorageManager._infer_mime(b"unknown-data", "image/webp")

        self.assertEqual(result, "image/webp")

    def test_generate_image_via_openai_uses_generate_without_reference(self) -> None:
        """确认未传参考图时调用 OpenAI 图像生成接口。"""
        fake_response = SimpleNamespace(data=[SimpleNamespace(b64_json="ZmFrZQ==")])
        fake_images = SimpleNamespace(
            generate=mock.Mock(return_value=fake_response),
            edit=mock.Mock(),
        )
        fake_client = SimpleNamespace(images=fake_images)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            with mock.patch("openai.OpenAI", return_value=fake_client):
                result = manager.generate_image_via_openai(" draw a cat ")

        self.assertIs(result, fake_response)
        fake_images.generate.assert_called_once_with(
            model="gpt-image-2",
            prompt="draw a cat",
        )
        fake_images.edit.assert_not_called()

    def test_generate_image_via_openai_uses_edit_with_reference(self) -> None:
        """确认传入参考图时调用 OpenAI 图像编辑接口。"""
        fake_response = SimpleNamespace(data=[SimpleNamespace(b64_json="ZmFrZQ==")])
        fake_images = SimpleNamespace(
            generate=mock.Mock(),
            edit=mock.Mock(return_value=fake_response),
        )
        fake_client = SimpleNamespace(images=fake_images)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            reference_path = Path(tmp_dir) / "reference.png"
            reference_path.write_bytes(b"fake-reference")
            with mock.patch("openai.OpenAI", return_value=fake_client):
                result = manager.generate_image_via_openai(" edit this ", reference_path)

        self.assertIs(result, fake_response)
        fake_images.generate.assert_not_called()
        fake_images.edit.assert_called_once_with(
            model="gpt-image-2",
            image=reference_path.resolve(),
            prompt="edit this",
        )

    def test_save_base64_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            data = self._make_image_base64(1, 1, "JPEG")
            stored = manager.save_base64_image(data, "image/jpeg")
            self.assertTrue(stored.path.exists())
            self.assertTrue(stored.base64_data)

    def test_load_stored_image_by_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            data = self._make_image_base64(1, 1, "PNG")
            stored = manager.save_base64_image(data, "image/png")
            loaded = manager.load_stored_image(stored.path.name)
            self.assertEqual(loaded.path, stored.path)
            self.assertEqual(loaded.mime_type, "image/png")
            self.assertTrue(loaded.base64_data)
            with self.assertRaises(AssertionError):
                manager.load_stored_image("missing.png")

    def test_save_base64_image_resizes_when_pixels_exceed_limit(self) -> None:
        """确认超过像素上限的入站图片会自动缩放后保存。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            data = self._make_image_base64(20, 20, "PNG")
            with mock.patch.object(image_storage, "MAX_STORED_IMAGE_PIXELS", 100):
                stored = manager.save_base64_image(data, "image/png")

            self.assertEqual(stored.mime_type, "image/jpeg")
            with Image.open(stored.path) as saved:
                self.assertLessEqual(saved.size[0] * saved.size[1], 100)

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

    def test_save_remote_image_uses_validated_user_agent(self) -> None:
        """确认最终图片下载使用与搜索验证一致的稳定 UA。"""
        image_data = base64.b64decode(self._make_image_base64(1, 1, "PNG"))
        response = SimpleNamespace(
            status_code=200,
            content=image_data,
            headers={"Content-Type": "image/png"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            with mock.patch.object(
                image_storage.requests,
                "get",
                return_value=response,
            ) as get_mock:
                stored = manager.save_remote_image("https://example.com/image.png")

        self.assertIsNotNone(stored)
        self.assertEqual(
            get_mock.call_args.kwargs["headers"]["User-Agent"],
            "Mozilla/5.0",
        )

    def test_save_remote_video_records_probed_duration(self) -> None:
        """
        远程视频保存后应记录 ffprobe 返回的时长。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 断言失败时由 unittest 报告。
        """
        response = mock.Mock()
        response.status_code = 200
        response.headers = {"Content-Type": "video/mp4"}
        response.iter_content.return_value = [b"fake-video"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            with (
                mock.patch.object(
                    image_storage.requests, "get", return_value=response
                ),
                mock.patch.object(
                    manager, "_guess_video_mime", return_value="video/mp4"
                ),
                mock.patch.object(
                    manager, "_probe_video_duration_seconds", return_value=9.6
                ) as probe_duration,
            ):
                stored = manager.save_remote_video(
                    "https://example.com/video.mp4"
                )

        self.assertEqual(stored.duration_seconds, 9.6)
        probe_duration.assert_called_once_with(stored.path)
        response.close.assert_called_once_with()

    def test_probe_video_duration_uses_ffprobe(self) -> None:
        """
        视频时长探测应调用 ffprobe 并解析秒数。

        Returns:
            None: 测试无返回值。

        Raises:
            None: 断言失败时由 unittest 报告。
        """
        completed = SimpleNamespace(returncode=0, stdout="7.25\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "video.mp4"
            video_path.write_bytes(b"fake-video")
            with mock.patch.object(
                image_storage.subprocess, "run", return_value=completed
            ) as run_mock:
                duration = ImageStorageManager._probe_video_duration_seconds(
                    video_path
                )

        self.assertEqual(duration, 7.25)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], "ffprobe")
        self.assertEqual(command[-1], str(video_path))

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

    def test_generate_image_via_gemini_retries_429_with_backoff(self) -> None:
        class FakeClientError(Exception):
            def __init__(self, status_code: int) -> None:
                super().__init__(f"HTTP {status_code}")
                self.status_code = status_code

        class FakeModels:
            def __init__(self) -> None:
                self.calls = 0

            def generate_content(self, model: str, contents: object, config: object) -> object:
                del model, contents, config
                self.calls += 1
                if self.calls <= 3:
                    raise FakeClientError(429)
                inline_data = SimpleNamespace(data=b"fake-image", mime_type="image/png")
                part = SimpleNamespace(text=None, inline_data=inline_data)
                content = SimpleNamespace(parts=[part])
                candidate = SimpleNamespace(
                    content=content,
                    finish_reason=None,
                    finish_message=None,
                )
                return SimpleNamespace(candidates=[candidate], prompt_feedback=None)

        fake_models = FakeModels()
        fake_client = SimpleNamespace(models=fake_models)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            with (
                mock.patch.object(image_storage.genai, "Client", return_value=fake_client),
                mock.patch.object(image_storage.errors, "ClientError", FakeClientError),
                mock.patch.object(image_storage.time, "sleep") as sleep_mock,
            ):
                result = manager.generate_image_via_gemini("draw a cat")
            self.assertTrue(result.path.exists())

        self.assertEqual([call.args[0] for call in sleep_mock.call_args_list], [5, 10, 30])
        self.assertEqual(fake_models.calls, 4)
        self.assertEqual(result.mime_type, "image/png")
        self.assertEqual(result.prompt, "draw a cat")


if __name__ == "__main__":
    unittest.main()
