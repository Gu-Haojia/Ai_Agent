import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from image_storage import ImageStorageManager
from src.yt_dlp_downloader import DownloadedVideo, YtDlpVideoDownloader

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
        target.write_bytes(b"fake-video")
        return {
            "id": "123456",
            "title": "测试视频",
            "extractor": "twitter",
            "requested_downloads": [{"filepath": str(target)}],
        }


class YtDlpDownloaderTest(unittest.TestCase):
    def test_from_image_storage_uses_same_video_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ImageStorageManager(tmp_dir)
            downloader = YtDlpVideoDownloader.from_image_storage(manager)
            self.assertEqual(
                downloader._video_dir,
                (Path(tmp_dir) / "incoming" / "video").resolve(),
            )

    def test_download_uses_auto_extractor_for_x_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = YtDlpVideoDownloader(Path(tmp_dir) / "incoming" / "video")
            with mock.patch("src.yt_dlp_downloader.YoutubeDL", FakeYoutubeDL):
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


if __name__ == "__main__":
    unittest.main()
