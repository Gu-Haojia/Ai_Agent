import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from yt_dlp import YoutubeDL

from image_storage import ImageStorageManager
from src.yt_dlp_downloader import DownloadedVideo, YtDlpVideoDownloader
from src.video_validation import VideoMetadata

try:
    from qq_group_bot import QQBotHandler

    _QQ_MODULE_AVAILABLE = True
except ModuleNotFoundError:
    QQBotHandler = None
    _QQ_MODULE_AVAILABLE = False


class FakeYoutubeDL:
    """用于单元测试的 yt-dlp 替身。"""

    last_options: dict[str, object] = {}
    last_url: str = ""
    file_bytes: bytes = b"fake-video"

    def __init__(self, options: dict[str, object]) -> None:
        self._options = options
        FakeYoutubeDL.last_options = options

    def __enter__(self) -> "FakeYoutubeDL":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        del exc_type, exc, tb
        return False

    def extract_info(self, url: str, download: bool = True) -> dict[str, object]:
        """
        模拟 yt-dlp 下载返回值。

        Args:
            url (str): 原始下载链接。
            download (bool): 是否执行下载。

        Returns:
            dict[str, object]: 模拟的下载结果。

        Raises:
            AssertionError: 当 download 为 False 时抛出。
        """
        assert download is True, "测试中必须执行 download=True"
        FakeYoutubeDL.last_url = url
        template = str(self._options["outtmpl"]["default"])
        file_path = (
            template.replace("%(extractor)s", "twitter")
            .replace("%(id)s", "123456")
            .replace("%(ext)s", "mp4")
        )
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(FakeYoutubeDL.file_bytes)
        return {
            "id": "123456",
            "title": "测试视频",
            "extractor": "twitter",
            "requested_downloads": [{"filepath": str(target)}],
        }


class YtDlpDownloaderTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeYoutubeDL.file_bytes = b"fake-video"

    def test_cookie_file_is_optional_and_detected_on_next_download(self) -> None:
        """缺少文件时保持原行为，新增文件后无需重建下载器即可加载。

        Returns:
            None: 验证不存在时不创建文件，存在时正确加载并限定域名。

        Raises:
            AssertionError: 当自动检测或 Cookie 域名隔离失效时抛出。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cookie_path = Path(tmp_dir) / "secret" / "bilibili_cookies.txt"
            downloader = YtDlpVideoDownloader(Path(tmp_dir) / "videos")
            with mock.patch.object(YtDlpVideoDownloader, "_COOKIE_FILE", cookie_path):
                options = downloader._build_options()
                self.assertNotIn("cookiefile", options)
                with YoutubeDL(options) as ydl:
                    self.assertEqual(len(ydl.cookiejar), 0)
                self.assertFalse(cookie_path.parent.exists())
                cookie_path.parent.mkdir()
                cookie_path.write_text(
                    "# Netscape HTTP Cookie File\n"
                    ".bilibili.com\tTRUE\t/\tTRUE\t0\tSESSDATA\ttest-session\n",
                    encoding="utf-8",
                )
                options = downloader._build_options()
                self.assertEqual(options["cookiefile"], str(cookie_path))
                self.assertNotIn("cookiesfrombrowser", options)
                with YoutubeDL(options) as ydl:
                    self.assertEqual(
                        ydl.cookiejar.get_cookie_header("https://www.bilibili.com/"),
                        "SESSDATA=test-session",
                    )
                    self.assertIsNone(ydl.cookiejar.get_cookie_header("https://x.com/"))

    def test_cookie_directory_is_rejected(self) -> None:
        """固定路径被误建为目录时明确失败。

        Returns:
            None: 验证错误路径不会被忽略。

        Raises:
            AssertionError: 当目录被接受为 Cookie 文件时抛出。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = YtDlpVideoDownloader(tmp_dir)
            with mock.patch.object(YtDlpVideoDownloader, "_COOKIE_FILE", Path(tmp_dir)):
                with self.assertRaisesRegex(AssertionError, "必须为普通文件"):
                    downloader._build_options()

    def test_empty_cookie_file_is_rejected(self) -> None:
        """拒绝空 Cookie 文件，避免悄悄发起未登录请求。

        Returns:
            None: 验证空文件检查。

        Raises:
            AssertionError: 当空文件未被拒绝时抛出。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cookie_path = Path(tmp_dir) / "cookies.txt"
            cookie_path.touch()
            downloader = YtDlpVideoDownloader(tmp_dir)
            with mock.patch.object(YtDlpVideoDownloader, "_COOKIE_FILE", cookie_path):
                with self.assertRaisesRegex(AssertionError, "不能为空"):
                    downloader._build_options()

    def test_unreadable_cookie_file_is_rejected(self) -> None:
        """拦截不可读文件，防止 yt-dlp 静默忽略登录信息。

        Returns:
            None: 验证权限错误在请求前暴露。

        Raises:
            AssertionError: 当不可读文件被静默忽略时抛出。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cookie_path = Path(tmp_dir) / "cookies.txt"
            cookie_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            downloader = YtDlpVideoDownloader(tmp_dir)
            with (
                mock.patch.object(YtDlpVideoDownloader, "_COOKIE_FILE", cookie_path),
                mock.patch("src.yt_dlp_downloader.os.access", return_value=False),
            ):
                with self.assertRaisesRegex(AssertionError, "不可读"):
                    downloader._build_options()

    def test_from_image_storage_uses_same_video_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            downloader = YtDlpVideoDownloader.from_image_storage(manager)
            self.assertEqual(
                downloader._video_dir,
                (Path(tmp_dir) / "incoming" / "video").resolve(),
            )

    def test_download_rejects_corrupt_small_video_with_zero_probe_exit(self) -> None:
        """拦截 ffprobe 返回成功但报告非法帧大小的小视频。

        Returns:
            None: 验证下载失败且保留诊断文件。

        Raises:
            AssertionError: 当损坏文件被当作下载成功时抛出。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = YtDlpVideoDownloader(tmp_dir)
            probe_result = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="{}",
                stderr="Sample size 3199971767 is too large",
            )
            with (
                mock.patch("src.yt_dlp_downloader.YoutubeDL", FakeYoutubeDL),
                mock.patch("subprocess.run", return_value=probe_result),
            ):
                with self.assertRaisesRegex(AssertionError, "3199971767"):
                    downloader.download("https://x.com/example/status/123456")
            self.assertEqual(len(list(Path(tmp_dir).glob("*.mp4"))), 1)

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖")
    def test_dl_rejects_corrupt_video_before_sending(self) -> None:
        """通过真实命令处理链确认坏文件只产生失败提示。

        Returns:
            None: 验证不会构造视频消息或删除诊断文件。

        Raises:
            AssertionError: 当命令仍发送视频或删除坏文件时抛出。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            handler = object.__new__(QQBotHandler)
            handler.bot_cfg = SimpleNamespace(
                api_base="http://127.0.0.1:3000", access_token="",
                cmd_allowed_users=(),
            )
            manager = ImageStorageManager(tmp_dir)
            with (
                mock.patch.object(QQBotHandler, "_require_image_storage", return_value=manager),
                mock.patch("src.yt_dlp_downloader.YoutubeDL", FakeYoutubeDL),
                mock.patch("subprocess.run", return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="{}",
                    stderr="Sample size 3199971767 is too large",
                )),
                mock.patch("qq_group_bot._send_group_msg") as send_mock,
                mock.patch("qq_group_bot._delete_local_video_after_delay") as cleanup_mock,
            ):
                self.assertTrue(handler._handle_commands(
                    10001, 20002, "/dl https://x.com/example/status/123456"
                ))
            send_mock.assert_called_once()
            self.assertIsInstance(send_mock.call_args.args[2], str)
            self.assertIn("下载失败", send_mock.call_args.args[2])
            cleanup_mock.assert_not_called()

    def test_download_uses_auto_extractor_for_x_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = YtDlpVideoDownloader(Path(tmp_dir) / "incoming" / "video")
            with (
                mock.patch("src.yt_dlp_downloader.YoutubeDL", FakeYoutubeDL),
                mock.patch.object(downloader._validator, "validate", return_value=
                    VideoMetadata(10.0, 320, 180, "h264", "yuv420p", False)),
            ):
                result = downloader.download(
                    "https://x.com/example/status/1234567890?s=20"
                )
            self.assertTrue(result.path.is_file())
        self.assertEqual(result.extractor, "twitter")
        self.assertEqual(result.title, "测试视频")
        self.assertNotIn("allowed_extractors", FakeYoutubeDL.last_options)
        self.assertEqual(
            FakeYoutubeDL.last_options.get("format"),
            YtDlpVideoDownloader._DEFAULT_FORMAT,
        )
        self.assertEqual(
            FakeYoutubeDL.last_options.get("merge_output_format"),
            "mp4",
        )
        self.assertTrue(YtDlpVideoDownloader._DEFAULT_FORMAT.endswith("/best"))

    def test_download_compresses_video_when_size_exceeds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            FakeYoutubeDL.file_bytes = b"x" * 10
            downloader = YtDlpVideoDownloader(Path(tmp_dir) / "incoming" / "video")

            def _write_compressed(
                _input_path: Path,
                output_path: Path,
                _video_bitrate_kbps: int,
                _audio_bitrate_kbps: int,
            ) -> None:
                output_path.write_bytes(b"compressed")

            with (
                mock.patch("src.yt_dlp_downloader.YoutubeDL", FakeYoutubeDL),
                mock.patch.object(downloader._validator, "validate", return_value=
                    VideoMetadata(10.0, 320, 180, "h264", "yuv420p", False)),
                mock.patch.object(
                    downloader,
                    "_calculate_target_bitrates",
                    return_value=(1000, 64),
                ),
                mock.patch.object(
                    downloader,
                    "_run_ffmpeg_compression",
                    side_effect=_write_compressed,
                ),
                mock.patch.object(downloader, "_COMPRESS_TRIGGER_BYTES", 1),
                mock.patch.object(downloader, "_MAX_SENDABLE_BYTES", 1024),
            ):
                result = downloader.download("https://x.com/example/status/1234567890?s=20")
                self.assertTrue(result.path.is_file())
                self.assertEqual(result.path.name, "twitter_123456_compressed.mp4")
                self.assertEqual(result.path.read_bytes(), b"compressed")

    def test_download_raises_when_compressed_video_still_exceeds_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            FakeYoutubeDL.file_bytes = b"x" * 10
            downloader = YtDlpVideoDownloader(Path(tmp_dir) / "incoming" / "video")

            def _write_oversized(
                _input_path: Path,
                output_path: Path,
                _video_bitrate_kbps: int,
                _audio_bitrate_kbps: int,
            ) -> None:
                output_path.write_bytes(b"y" * 40)

            with (
                mock.patch("src.yt_dlp_downloader.YoutubeDL", FakeYoutubeDL),
                mock.patch.object(downloader._validator, "validate", return_value=
                    VideoMetadata(10.0, 320, 180, "h264", "yuv420p", False)),
                mock.patch.object(
                    downloader,
                    "_calculate_target_bitrates",
                    return_value=(1000, 64),
                ),
                mock.patch.object(
                    downloader,
                    "_run_ffmpeg_compression",
                    side_effect=_write_oversized,
                ),
                mock.patch.object(downloader, "_COMPRESS_TRIGGER_BYTES", 1),
                mock.patch.object(downloader, "_MAX_SENDABLE_BYTES", 20),
            ):
                with self.assertRaisesRegex(AssertionError, "压制后视频仍超过 70MB"):
                    downloader.download("https://x.com/example/status/1234567890?s=20")

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 命令测试")
    def test_handle_commands_dl_sends_video_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            handler = object.__new__(QQBotHandler)
            handler.bot_cfg = SimpleNamespace(
                api_base="http://127.0.0.1:3000",
                access_token="",
                cmd_allowed_users=(),
            )
            manager = ImageStorageManager(tmp_dir)
            video_path = Path(tmp_dir) / "incoming" / "video" / "twitter_1.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"video")
            downloaded = DownloadedVideo(
                path=video_path,
                source_url="https://x.com/example/status/1",
                extractor="twitter",
                title="测试视频",
            )
            with (
                mock.patch.object(
                    QQBotHandler, "_require_image_storage", return_value=manager
                ),
                mock.patch(
                    "qq_group_bot.YtDlpVideoDownloader.from_image_storage"
                ) as factory_mock,
                mock.patch("qq_group_bot._send_group_msg") as send_mock,
                mock.patch(
                    "qq_group_bot._delete_local_video_after_delay"
                ) as cleanup_mock,
            ):
                factory_mock.return_value.download.return_value = downloaded
                handled = handler._handle_commands(
                    10001, 20002, "/dl https://x.com/example/status/1"
                )
        self.assertTrue(handled)
        factory_mock.return_value.download.assert_called_once_with(
            "https://x.com/example/status/1"
        )
        send_mock.assert_called_once()
        cleanup_mock.assert_called_once_with(video_path)
        payload = send_mock.call_args.args[2]
        self.assertEqual(payload[0]["type"], "video")
        self.assertEqual(
            payload[0]["data"]["file"],
            "base64://dmlkZW8=",
        )

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 命令测试")
    @mock.patch.dict(os.environ)
    def test_handle_commands_imageprovider_toggles_env(self) -> None:
        """确认 /imageprovider 默认在 Gemini 与 OpenAI 之间循环。"""
        old_provider = os.environ.pop("IMAGE_PROVIDER", None)
        old_next_provider = os.environ.pop("NEXT_IMAGE_PROVIDER", None)
        os.environ.pop("GEMINI_IMAGE_MODEL", None)
        try:
            handler = object.__new__(QQBotHandler)
            handler.bot_cfg = SimpleNamespace(
                api_base="http://127.0.0.1:3000",
                access_token="",
                cmd_allowed_users=(),
            )

            with (
                tempfile.TemporaryDirectory() as tmp_dir,
                mock.patch.object(
                    QQBotHandler, "image_storage", ImageStorageManager(tmp_dir)
                ),
                mock.patch("qq_group_bot._send_group_msg") as send_mock,
            ):
                handled = handler._handle_commands(10001, 20002, "/imageprovider")

            self.assertTrue(handled)
            self.assertEqual(os.environ.get("IMAGE_PROVIDER"), "openai")
            send_mock.assert_called_once()
            self.assertIn("gemini -> openai", send_mock.call_args.args[2])

            with mock.patch("qq_group_bot._send_group_msg") as send_mock:
                handled = handler._handle_commands(10001, 20002, "/imageprovider")

            self.assertTrue(handled)
            self.assertEqual(os.environ.get("IMAGE_PROVIDER"), "gemini")
            send_mock.assert_called_once()
            self.assertIn("openai -> gemini", send_mock.call_args.args[2])
            self.assertIn(
                "当前生图模型：gemini-2.5-flash-image。", send_mock.call_args.args[2]
            )
        finally:
            if old_provider is None:
                os.environ.pop("IMAGE_PROVIDER", None)
            else:
                os.environ["IMAGE_PROVIDER"] = old_provider
            if old_next_provider is None:
                os.environ.pop("NEXT_IMAGE_PROVIDER", None)
            else:
                os.environ["NEXT_IMAGE_PROVIDER"] = old_next_provider

    @unittest.skipUnless(_QQ_MODULE_AVAILABLE, "缺少 langgraph 依赖，跳过 QQ 命令测试")
    def test_handle_commands_imageprovider_uses_configured_provider(self) -> None:
        """确认 /imageprovider 使用环境变量配置循环目标。"""
        old_provider = os.environ.pop("IMAGE_PROVIDER", None)
        old_next_provider = os.environ.get("NEXT_IMAGE_PROVIDER")
        os.environ["NEXT_IMAGE_PROVIDER"] = "xai"
        try:
            handler = object.__new__(QQBotHandler)
            handler.bot_cfg = SimpleNamespace(
                api_base="http://127.0.0.1:3000",
                access_token="",
                cmd_allowed_users=(),
            )

            with mock.patch("qq_group_bot._send_group_msg") as send_mock:
                handled = handler._handle_commands(10001, 20002, "/imageprovider")

            self.assertTrue(handled)
            self.assertEqual(os.environ.get("IMAGE_PROVIDER"), "xai")
            send_mock.assert_called_once()
            self.assertIn("gemini -> xai", send_mock.call_args.args[2])

            with mock.patch("qq_group_bot._send_group_msg") as send_mock:
                handled = handler._handle_commands(10001, 20002, "/imageprovider")

            self.assertTrue(handled)
            self.assertEqual(os.environ.get("IMAGE_PROVIDER"), "gemini")
            send_mock.assert_called_once()
            self.assertIn("xai -> gemini", send_mock.call_args.args[2])
        finally:
            if old_provider is None:
                os.environ.pop("IMAGE_PROVIDER", None)
            else:
                os.environ["IMAGE_PROVIDER"] = old_provider
            if old_next_provider is None:
                os.environ.pop("NEXT_IMAGE_PROVIDER", None)
            else:
                os.environ["NEXT_IMAGE_PROVIDER"] = old_next_provider


if __name__ == "__main__":
    unittest.main()
