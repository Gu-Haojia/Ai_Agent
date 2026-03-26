"""
yt-dlp 视频下载模块。

负责调用 yt-dlp 自动识别站点，
并将下载后的视频保存到 QQ Bot 既有的视频缓存目录。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

from image_storage import ImageStorageManager
from yt_dlp import YoutubeDL

__all__ = ["DownloadedVideo", "YtDlpVideoDownloader"]


@dataclass(frozen=True)
class DownloadedVideo:
    """已下载的视频文件信息。"""

    path: Path
    source_url: str
    extractor: str
    title: str


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

    def download(self, url: str) -> DownloadedVideo:
        """
        下载给定 URL 对应的视频。

        Args:
            url (str): 视频页面链接。

        Returns:
            DownloadedVideo: 下载完成后的本地文件信息。

        Raises:
            AssertionError: 当 URL 非法或下载结果缺少文件路径时抛出。
            Exception: 当 yt-dlp 下载失败时透传原始异常。
        """
        assert isinstance(url, str) and url.strip(), "下载链接不能为空"
        target_url = url.strip()
        assert target_url.startswith(("http://", "https://")), "下载链接必须为 HTTP(S) 地址"
        self._log(f"使用 yt-dlp 自动识别 extractor: url={target_url}")
        self._log(f"开始下载到目录: {self._video_dir}")
        try:
            with YoutubeDL(self._build_options()) as ydl:
                info = ydl.extract_info(target_url, download=True)
            path = self._extract_downloaded_path(info)
            title = str(info.get("title") or path.stem)
            extractor = str(info.get("extractor") or "generic")
            path = self._compress_if_needed(path)
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
        构造 yt-dlp 下载参数。

        Returns:
            dict[str, object]: yt-dlp 参数字典。

        Raises:
            None: 本方法不会主动抛出异常。
        """
        options: dict[str, object] = {
            "quiet": True,
            "no_warnings": True,
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
        return options

    def _compress_if_needed(self, video_path: Path) -> Path:
        """
        在文件超过阈值时压制视频到可发送大小。

        Args:
            video_path (Path): 已下载完成的视频文件路径。

        Returns:
            Path: 可直接用于发送的视频文件路径。

        Raises:
            AssertionError: 当压制后文件仍超过上限或参数非法时抛出。
            OSError: 当删除旧文件失败时抛出。
        """
        resolved = video_path.expanduser().resolve()
        assert resolved.is_file(), "待压制视频文件不存在"
        current_size = resolved.stat().st_size
        if current_size <= self._COMPRESS_TRIGGER_BYTES:
            return resolved
        self._log(f"文件超过压制阈值，开始压制: path={resolved} size={current_size}")
        duration_seconds = self._probe_duration_seconds(resolved)
        video_bitrate_kbps, audio_bitrate_kbps = self._calculate_target_bitrates(
            duration_seconds
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
        resolved.unlink()
        return compressed_path

    def _probe_duration_seconds(self, video_path: Path) -> float:
        """
        使用 ffprobe 读取视频时长。

        Args:
            video_path (Path): 视频文件路径。

        Returns:
            float: 视频时长，单位为秒。

        Raises:
            AssertionError: 当 ffprobe 执行失败或时长非法时抛出。
        """
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        stderr = completed.stderr.strip()
        assert completed.returncode == 0, f"ffprobe 读取时长失败: {stderr or completed.stdout.strip()}"
        duration_text = completed.stdout.strip()
        assert duration_text, "ffprobe 未返回视频时长"
        duration_seconds = float(duration_text)
        assert duration_seconds > 0, "视频时长必须大于 0"
        return duration_seconds

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
        """
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
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
        )
        stderr = completed.stderr.strip()
        assert completed.returncode == 0, f"ffmpeg 压制失败: {stderr or completed.stdout.strip()}"
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
