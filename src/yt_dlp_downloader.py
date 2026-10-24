"""
yt-dlp 视频下载模块。

负责调用 yt-dlp 自动识别站点，
并将下载后的视频保存到 QQ Bot 既有的视频缓存目录。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from image_storage import ImageStorageManager
from src.video_validation import VideoMetadata, VideoValidator
from yt_dlp import YoutubeDL
from yt_dlp.postprocessor.common import PostProcessor
from yt_dlp.utils import format_bytes

__all__ = ["DownloadedVideo", "YtDlpVideoDownloader"]


@dataclass(frozen=True)
class DownloadedVideo:
    """已下载的视频文件信息。"""

    path: Path
    source_url: str
    extractor: str
    title: str


class DownloadStartNoticePP(PostProcessor):
    """在视频解析完成、媒体下载开始前发送一次基本信息提示。

    Args:
        notify (Callable[[str], None]): 接收提示文本的回调。

    Returns:
        None: 构造时不发送提示。

    Raises:
        None: 构造时不主动抛出异常。
    """

    def __init__(self, notify: Callable[[str], None]) -> None:
        """保存当前下载请求的通知回调。

        Args:
            notify (Callable[[str], None]): 在下载前同步执行的通知函数。

        Returns:
            None: 初始化通知状态。

        Raises:
            None: 构造时不主动抛出异常。
        """
        super().__init__()
        self._notify = notify
        self._notified = False

    def run(self, info: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        """依据已选中的下载格式发送信息，后续分轨或重试不重复通知。

        Args:
            info (dict[str, Any]): yt-dlp 已解析并选择格式的视频信息。

        Returns:
            tuple[list[str], dict[str, Any]]: 空删除列表和原视频信息。

        Raises:
            AssertionError: 当视频标题缺失、时长或大小无效时抛出。
            Exception: 通知回调失败时透传异常，停止下载。
        """
        if not self._notified:
            self._notify(self._format_message(info))
            self._notified = True
        return [], info

    @staticmethod
    def _format_message(info: dict[str, Any]) -> str:
        """格式化标题、时长、分辨率与下载大小。

        Args:
            info (dict[str, Any]): yt-dlp 视频信息。

        Returns:
            str: 视频基本信息与开始下载提示。

        Raises:
            AssertionError: 当标题缺失、时长或大小无效时抛出。
        """
        title = info.get("title")
        assert isinstance(title, str) and title.strip(), "解析结果缺少视频标题"
        lines = ["视频解析成功，开始下载…", f"标题：{title.strip()}"]
        duration = info.get("duration")
        if duration is not None:
            assert isinstance(duration, (int, float)) and duration >= 0, "视频时长无效"
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_text = f"{minutes:02d}:{seconds:02d}"
            if hours:
                time_text = f"{hours}:{time_text}"
            lines.append(f"时长：{time_text}")
        width, height = info.get("width"), info.get("height")
        if width and height:
            lines.append(f"分辨率：{width}×{height}")
        lines.append(f"大小：{DownloadStartNoticePP._format_size(info)}")
        return "\n".join(lines)

    @staticmethod
    def _format_size(info: dict[str, Any]) -> str:
        """显示选中格式的大小，分轨合计标为估算且不将部分大小当总量。

        Args:
            info (dict[str, Any]): 包含单文件或所选音视频分轨的视频信息。

        Returns:
            str: 实际大小、带“约”的估算大小，或“未知”。

        Raises:
            AssertionError: 当已提供的大小不是非负数时抛出。
        """
        selected_formats = info.get("requested_formats")
        if selected_formats:
            sizes: list[float] = []
            for item in selected_formats:
                size = item.get("filesize")
                if size is None:
                    size = item.get("filesize_approx")
                if size is None:
                    return "未知"
                assert isinstance(size, (int, float)) and size >= 0, "视频大小无效"
                sizes.append(size)
            return f"约 {format_bytes(sum(sizes))}"
        size = info.get("filesize")
        approximate = size is None
        if approximate:
            size = info.get("filesize_approx")
        if size is None:
            return "未知"
        assert isinstance(size, (int, float)) and size >= 0, "视频大小无效"
        return f"{'约 ' if approximate else ''}{format_bytes(size)}"


class YtDlpVideoDownloader:
    """
    使用 yt-dlp 下载视频到本地缓存目录。

    下载参数全部内置，不向外暴露格式、extractor 或缓存路径细节。
    """

    _DEFAULT_FORMAT = (
        "best[ext=mp4][acodec!=none][vcodec!=none]"
        "/best[acodec!=none][vcodec!=none]"
        "/bestvideo[ext=mp4]+bestaudio[ext=m4a]"
        "/bestvideo+bestaudio"
        "/best"
    )
    _MAX_FILESIZE_BYTES = 512 * 1024 * 1024
    _COMPRESS_TRIGGER_BYTES = 50 * 1024 * 1024
    _MAX_SENDABLE_BYTES = 70 * 1024 * 1024
    _TARGET_SIZE_RATIO = 0.92
    _DEFAULT_AUDIO_BITRATE_KBPS = 128
    _MIN_VIDEO_BITRATE_KBPS = 180
    _COOKIE_FILE = Path(__file__).resolve().parent.parent / "secret" / "bilibili_cookies.txt"
    _DEFAULT_HEADERS: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    def __init__(self, video_dir: Path | str) -> None:
        """
        初始化下载器。

        Args:
            video_dir (Path | str): 视频缓存目录。

        Returns:
            None: 本构造函数不返回值。

        Raises:
            AssertionError: 当目录参数为空时抛出。
        """
        assert video_dir, "video_dir 不能为空"
        self._video_dir = Path(video_dir).expanduser().resolve()
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._validator = VideoValidator()

    @staticmethod
    def _log(message: str) -> None:
        """
        输出下载工具的控制台日志。

        Args:
            message (str): 日志内容。

        Returns:
            None: 本方法不返回值。

        Raises:
            None: 本方法不会主动抛出异常。
        """
        timestamp = time.strftime("[%m-%d %H:%M:%S]", time.localtime())
        print(f"\033[94m{timestamp}\033[0m [yt-dlp] {message}", flush=True)

    @classmethod
    def from_image_storage(cls, manager: ImageStorageManager) -> "YtDlpVideoDownloader":
        """
        基于现有图像存储管理器创建下载器。

        Args:
            manager (ImageStorageManager): 已初始化的图像存储管理器。

        Returns:
            YtDlpVideoDownloader: 指向相同视频缓存目录的下载器实例。

        Raises:
            AssertionError: 当 manager 类型不符时抛出。
        """
        assert isinstance(manager, ImageStorageManager), "manager 必须为 ImageStorageManager 实例"
        video_dir = Path(getattr(manager, "_incoming_video_dir")).resolve()
        return cls(video_dir)

    def download(
        self, url: str, on_download_start: Callable[[str], None] | None = None
    ) -> DownloadedVideo:
        """
        下载给定 URL 对应的视频。

        Args:
            url (str): 视频页面链接。
            on_download_start (Callable[[str], None] | None): 解析成功后、媒体下载前
                接收基本信息提示的回调；未提供时不发送提示。

        Returns:
            DownloadedVideo: 下载完成后的本地文件信息。

        Raises:
            AssertionError: 当 URL、下载结果或视频完整性校验不符合要求时抛出。
            Exception: 当 yt-dlp 下载失败时透传原始异常。
        """
        assert isinstance(url, str) and url.strip(), "下载链接不能为空"
        target_url = url.strip()
        assert target_url.startswith(("http://", "https://")), "下载链接必须为 HTTP(S) 地址"
        self._log(f"使用 yt-dlp 自动识别 extractor: url={target_url}")
        self._log(f"开始下载到目录: {self._video_dir}")
        try:
            with YoutubeDL(self._build_options()) as ydl:
                if on_download_start is not None:
                    ydl.add_post_processor(
                        DownloadStartNoticePP(on_download_start), when="before_dl"
                    )
                info = ydl.extract_info(target_url, download=True)
            path = self._extract_downloaded_path(info)
            title = str(info.get("title") or path.stem)
            extractor = str(info.get("extractor") or "generic")
            duration = info.get("duration")
            expected_duration = None if duration is None else float(duration)
            self._log(f"开始视频完整性校验: path={path}")
            metadata = self._validator.validate(path, expected_duration)
            path = self._compress_if_needed(path, metadata)
        except Exception as exc:
            self._log(f"下载失败: {type(exc).__name__}: {exc}")
            raise
        self._log(
            f"下载完成: extractor={extractor} title={title} path={path} size={path.stat().st_size}"
        )
        return DownloadedVideo(
            path=path,
            source_url=target_url,
            extractor=extractor,
            title=title,
        )

    def _build_options(self) -> dict[str, object]:
        """
        构造 yt-dlp 下载参数，自动加载项目内已有的 Cookie 文件。

        Returns:
            dict[str, object]: yt-dlp 参数字典。

        Raises:
            AssertionError: 当 Cookie 路径不是文件、不可读或内容为空时抛出。
        """
        options: dict[str, object] = {
            "quiet": True,
            "no_warnings": False,
            "noprogress": True,
            "noplaylist": True,
            "overwrites": True,
            "socket_timeout": 30,
            "retries": 3,
            "file_access_retries": 3,
            "extractor_retries": 3,
            "format": self._DEFAULT_FORMAT,
            "merge_output_format": "mp4",
            "max_filesize": self._MAX_FILESIZE_BYTES,
            "http_headers": dict(self._DEFAULT_HEADERS),
            "outtmpl": {
                "default": str(self._video_dir / "%(extractor)s_%(id)s.%(ext)s")
            },
        }
        if self._COOKIE_FILE.exists():
            assert self._COOKIE_FILE.is_file(), "Cookie 路径必须为普通文件"
            assert os.access(self._COOKIE_FILE, os.R_OK), "Cookie 文件不可读"
            assert self._COOKIE_FILE.stat().st_size > 0, "Cookie 文件不能为空"
            options["cookiefile"] = str(self._COOKIE_FILE)
        return options

    def _compress_if_needed(self, video_path: Path, metadata: VideoMetadata) -> Path:
        """
        对超出阈值或编码不兼容的视频转码，校验完成后才替换原文件。

        Args:
            video_path (Path): 已下载完成的视频文件路径。
            metadata (VideoMetadata): 已通过完整性校验的视频信息。

        Returns:
            Path: 可直接用于发送的视频文件路径。

        Raises:
            AssertionError: 当压制后文件仍超过上限或参数非法时抛出。
            OSError: 当删除旧文件失败时抛出。
        """
        resolved = video_path.expanduser().resolve()
        assert resolved.is_file(), "待压制视频文件不存在"
        current_size = resolved.stat().st_size
        if current_size <= self._COMPRESS_TRIGGER_BYTES and not metadata.requires_transcode:
            return resolved
        self._log(
            f"开始视频压制或兼容性转换: path={resolved} size={current_size} "
            f"codec={metadata.video_codec} pixel_format={metadata.pixel_format}"
        )
        video_bitrate_kbps, audio_bitrate_kbps = self._calculate_target_bitrates(
            metadata.duration_seconds
        )
        if current_size <= self._COMPRESS_TRIGGER_BYTES:
            # 小文件只做兼容性转换，避免短片按 70MB 预算被设置成极高码率。
            source_bitrate_kbps = int(current_size * 8 / metadata.duration_seconds / 1000)
            video_bitrate_kbps = min(
                video_bitrate_kbps,
                max(self._MIN_VIDEO_BITRATE_KBPS, source_bitrate_kbps * 2),
            )
        compressed_path = resolved.with_name(f"{resolved.stem}_compressed.mp4")
        if compressed_path.exists():
            compressed_path.unlink()
        self._run_ffmpeg_compression(
            resolved,
            compressed_path,
            video_bitrate_kbps,
            audio_bitrate_kbps,
        )
        compressed_size = compressed_path.stat().st_size
        self._log(
            "压制完成: "
            f"path={compressed_path} size={compressed_size} "
            f"video_bitrate={video_bitrate_kbps}k audio_bitrate={audio_bitrate_kbps}k"
        )
        assert compressed_size <= self._MAX_SENDABLE_BYTES, "压制后视频仍超过 70MB"
        compressed_metadata = self._validator.validate(
            compressed_path, metadata.duration_seconds
        )
        assert not compressed_metadata.requires_transcode, "转码结果不符合 QQ 视频格式要求"
        resolved.unlink()
        return compressed_path

    def _calculate_target_bitrates(self, duration_seconds: float) -> tuple[int, int]:
        """
        根据目标大小计算压制码率。

        Args:
            duration_seconds (float): 视频时长，单位为秒。

        Returns:
            tuple[int, int]: 视频码率与音频码率，单位为 kbps。

        Raises:
            AssertionError: 当目标码率不足以完成压制时抛出。
        """
        assert duration_seconds > 0, "duration_seconds 必须大于 0"
        target_total_bits = int(self._MAX_SENDABLE_BYTES * 8 * self._TARGET_SIZE_RATIO)
        total_bitrate_kbps = max(1, int(target_total_bits / duration_seconds / 1000))
        audio_bitrate_kbps = min(self._DEFAULT_AUDIO_BITRATE_KBPS, max(64, total_bitrate_kbps // 5))
        video_bitrate_kbps = total_bitrate_kbps - audio_bitrate_kbps
        assert video_bitrate_kbps >= self._MIN_VIDEO_BITRATE_KBPS, "视频过长，无法压制到 70MB 内"
        return video_bitrate_kbps, audio_bitrate_kbps

    def _run_ffmpeg_compression(
        self,
        input_path: Path,
        output_path: Path,
        video_bitrate_kbps: int,
        audio_bitrate_kbps: int,
    ) -> None:
        """
        调用 ffmpeg 执行视频压制。

        Args:
            input_path (Path): 原始视频路径。
            output_path (Path): 压制后视频路径。
            video_bitrate_kbps (int): 视频码率，单位为 kbps。
            audio_bitrate_kbps (int): 音频码率，单位为 kbps。

        Returns:
            None: 本方法不返回值。

        Raises:
            AssertionError: 当 ffmpeg 执行失败或未产出文件时抛出。
            subprocess.TimeoutExpired: 当压制超过三十分钟时抛出。
        """
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=ceil(iw*sar/2)*2:ceil(ih/2)*2,setsar=1",
            "-preset",
            "veryfast",
            "-b:v",
            f"{video_bitrate_kbps}k",
            "-maxrate",
            f"{video_bitrate_kbps}k",
            "-bufsize",
            f"{video_bitrate_kbps * 2}k",
            "-c:a",
            "aac",
            "-b:a",
            f"{audio_bitrate_kbps}k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        stderr = completed.stderr.strip()
        assert completed.returncode == 0 and not stderr, (
            f"ffmpeg 压制失败: {stderr or completed.stdout.strip()}"
        )
        assert output_path.is_file(), "ffmpeg 未生成压制文件"

    def _extract_downloaded_path(self, info: dict[str, object]) -> Path:
        """
        从 yt-dlp 返回结果中提取最终文件路径。

        Args:
            info (dict[str, object]): yt-dlp 返回的信息字典。

        Returns:
            Path: 已下载的视频绝对路径。

        Raises:
            AssertionError: 当结果中不存在有效文件路径时抛出。
        """
        requested_downloads = info.get("requested_downloads")
        assert isinstance(requested_downloads, list) and requested_downloads, "yt-dlp 未返回下载结果"
        first_item = requested_downloads[0]
        assert isinstance(first_item, dict), "yt-dlp 下载结果结构异常"
        file_path = str(first_item.get("filepath") or first_item.get("_filename") or "").strip()
        assert file_path, "yt-dlp 未返回下载文件路径"
        resolved = Path(file_path).expanduser().resolve()
        assert resolved.is_file(), "下载结果文件不存在"
        return resolved
