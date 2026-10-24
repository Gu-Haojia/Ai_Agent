"""验证真实 yt-dlp 链路中的下载前提示、分轨合并及失败边界。"""

from copy import deepcopy
import io
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from yt_dlp import YoutubeDL
from yt_dlp.extractor.common import InfoExtractor
from yt_dlp.networking import Request, Response
from yt_dlp.networking.exceptions import HTTPError
from yt_dlp.utils import ExtractorError, format_bytes

from image_storage import ImageStorageManager
from qq_group_bot import QQBotHandler
from src.yt_dlp_downloader import DownloadStartNoticePP, YtDlpVideoDownloader


class NoticeFixtureIE(InfoExtractor):
    """提供确定的解析结果，媒体仍由真实 yt-dlp 下载和 FFmpeg 校验。"""

    _VALID_URL = r"https://fixture\.example/watch/(?P<id>\d+)"

    def __init__(self, info: dict[str, Any], events: list[str], failure: str) -> None:
        """保存当前测试的解析信息与事件列表。

        Args:
            info (dict[str, Any]): 合成视频的解析信息。
            events (list[str]): 用于验证执行顺序的共享列表。
            failure (str): 要模拟的失败阶段。

        Returns:
            None: 初始化提取器。

        Raises:
            None: 构造时不会主动抛出异常。
        """
        super().__init__()
        self._info = info
        self._events = events
        self._failure = failure

    def _real_extract(self, url: str) -> dict[str, Any]:
        """模拟成功解析或解析失败，不替换后续下载逻辑。

        Args:
            url (str): 合成视频页面地址。

        Returns:
            dict[str, Any]: 独立的视频信息副本。

        Raises:
            ExtractorError: 测试要求模拟解析失败时抛出。
        """
        del url
        self._events.append("parse")
        if self._failure == "parse":
            raise ExtractorError("测试解析失败", expected=True)
        return deepcopy(self._info)


@pytest.fixture(scope="module")
def notice_media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """生成一秒完整视频及其音视频分轨，用于真实下载和解码验证。

    Args:
        tmp_path_factory (pytest.TempPathFactory): pytest 临时目录工厂。

    Returns:
        dict[str, bytes]: 按请求路径索引的媒体内容。

    Raises:
        subprocess.CalledProcessError: 合成或分轨失败时抛出。
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("需要 FFmpeg 和 ffprobe")
    folder = tmp_path_factory.mktemp("download-notice-media")
    combined = folder / "combined.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=64x32:rate=6:duration=1", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", str(combined),
    ], check=True, capture_output=True)
    for filename, option in (("video.mp4", "-an"), ("audio.m4a", "-vn")):
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(combined), "-c", "copy",
            option, str(folder / filename),
        ], check=True, capture_output=True)
    return {path.name: path.read_bytes() for path in folder.iterdir()}


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("failure", ["", "parse", "download", "notice"])
def test_dl_notice_precedes_real_media_download(
    tmp_path: Path, notice_media: dict[str, bytes], split: bool, failure: str
) -> None:
    """覆盖命令处理、真实下载、分轨合并与全片校验的通知顺序。

    Args:
        tmp_path (Path): 本次测试输出目录。
        notice_media (dict[str, bytes]): 合成媒体文件。
        split (bool): 是否使用独立音视频分轨。
        failure (str): 要模拟的失败阶段，空字符串表示成功。

    Returns:
        None: 验证群提示先于媒体请求，失败时不发送视频。

    Raises:
        AssertionError: 通知时机、内容、次数或失败处理不符合约定时抛出。
    """
    events: list[str] = []
    filename = "video.mp4" if split else "combined.mp4"
    formats = [{
        "format_id": "video", "url": f"https://fixture.example/{filename}",
        "ext": "mp4", "vcodec": "h264", "acodec": "none" if split else "aac",
        "width": 64, "height": 32, "filesize": len(notice_media[filename]),
    }]
    if split:
        formats.append({
            "format_id": "audio", "url": "https://fixture.example/audio.m4a",
            "ext": "m4a", "vcodec": "none", "acodec": "aac",
            "filesize": len(notice_media["audio.m4a"]),
        })
    info = {
        "id": "1", "title": "测试视频 [CQ:at,qq=all]", "duration": 1.0,
        "width": 64, "height": 32, "uploader": "不展示的作者", "formats": formats,
    }

    def create_downloader(options: dict[str, Any]) -> YoutubeDL:
        """注册测试提取器并保留真实 yt-dlp 下载实现。

        Args:
            options (dict[str, Any]): 项目下载器的真实参数。

        Returns:
            YoutubeDL: 使用确定解析结果的真实下载会话。

        Raises:
            None: 构造时不主动抛出异常。
        """
        ydl = YoutubeDL(options, auto_init=False)
        ydl.add_info_extractor(NoticeFixtureIE(info, events, failure))
        return ydl

    def respond(request: Request) -> Response:
        """在 HTTP 边界提供合成媒体，记录媒体请求的实际时机。

        Args:
            request (Request): yt-dlp 的真实媒体请求。

        Returns:
            Response: 可下载并解码的媒体数据。

        Raises:
            HTTPError: 模拟媒体服务器拒绝下载时抛出。
        """
        events.append("media")
        if failure == "download":
            raise HTTPError(Response(io.BytesIO(b""), request.url, {}, status=403))
        content = notice_media[request.url.rsplit("/", 1)[-1]]
        return Response(io.BytesIO(content), request.url, {"Content-Length": str(len(content))})

    def record_message(
        api_base: str, group_id: int, message: str | list[dict[str, Any]], access_token: str
    ) -> None:
        """记录回群顺序，防止测试向真实群发送消息。

        Args:
            api_base (str): 合成 OneBot 地址。
            group_id (int): 目标测试群。
            message (str | list[dict[str, Any]]): 提示、错误文本或视频消息。
            access_token (str): 测试令牌。

        Returns:
            None: 仅记录调用，不执行外部发送。

        Raises:
            RuntimeError: 测试要求模拟提示发送失败时抛出。
        """
        del api_base, access_token
        assert group_id == 10001
        if isinstance(message, str):
            events.append("failure")
        elif message[0]["type"] == "text":
            events.append("notice")
            if failure == "notice":
                raise RuntimeError("测试提示发送失败")
        else:
            events.append("video")

    handler = object.__new__(QQBotHandler)
    handler.bot_cfg = SimpleNamespace(api_base="http://fixture.example", access_token="", cmd_allowed_users=())
    manager = ImageStorageManager(str(tmp_path))
    with (
        mock.patch.object(QQBotHandler, "_require_image_storage", return_value=manager),
        mock.patch.object(YtDlpVideoDownloader, "_COOKIE_FILE", tmp_path / "missing-cookies.txt"),
        mock.patch("src.yt_dlp_downloader.YoutubeDL", side_effect=create_downloader),
        mock.patch.object(YoutubeDL, "urlopen", side_effect=respond),
        mock.patch("qq_group_bot._send_group_msg", side_effect=record_message) as send,
        mock.patch("qq_group_bot._delete_local_video_after_delay") as cleanup,
    ):
        assert handler._handle_commands(10001, 20002, "/dl https://fixture.example/watch/1")
    if failure == "parse":
        assert events == ["parse", "failure"]
    elif failure == "notice":
        assert events == ["parse", "notice", "failure"]
    elif failure == "download":
        assert events[:2] == ["parse", "notice"]
        assert events[-1] == "failure"
        assert "video" not in events
    else:
        assert events == ["parse", "notice", *(["media"] * (2 if split else 1)), "video"]
        cleanup.assert_called_once()
        assert cleanup.call_args.args[0].is_file()
    if failure:
        cleanup.assert_not_called()
    if failure != "parse":
        notice = send.call_args_list[0].args[2]
        assert notice[0]["type"] == "text"
        text = notice[0]["data"]["text"]
        assert "开始下载" in text and "测试视频 [CQ:at,qq=all]" in text
        assert "时长：00:01" in text and "分辨率：64×32" in text
        size = sum(len(notice_media[name]) for name in ("video.mp4", "audio.m4a")) if split else len(notice_media[filename])
        assert f"大小：{'约 ' if split else ''}{format_bytes(size)}" in text
        assert "来源" not in text and "作者" not in text


@pytest.mark.parametrize(("metadata", "expected"), [
    ({"filesize": 1048576}, "1.00MiB"),
    ({"filesize_approx": 2097152}, "约 2.00MiB"),
    ({}, "未知"),
    ({"requested_formats": [{"filesize": 1048576}, {"filesize_approx": 1048576}]}, "约 2.00MiB"),
    ({"filesize_approx": 1048576, "requested_formats": [{"filesize": 1048576}, {}]}, "未知"),
])
def test_notice_size_and_single_notification(metadata: dict[str, Any], expected: str) -> None:
    """校验估算和缺失大小，并防止同一请求重复触发提示。

    Args:
        metadata (dict[str, Any]): 不同来源的大小信息。
        expected (str): 期望的展示文本。

    Returns:
        None: 验证提示内容与调用次数。

    Raises:
        AssertionError: 大小展示不准确或回调重复执行时抛出。
    """
    callback = mock.Mock()
    processor = DownloadStartNoticePP(callback)
    info = {"title": "测试视频", "duration": 3661.5, **metadata}
    processor.run(info)
    processor.run(info)
    callback.assert_called_once()
    assert f"大小：{expected}" in callback.call_args.args[0]
    assert "时长：1:01:01" in callback.call_args.args[0]
