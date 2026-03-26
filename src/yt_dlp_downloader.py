"""
yt-dlp 视频下载模块。

负责调用 yt-dlp 自动识别站点，
并将下载后的视频保存到 QQ Bot 既有的视频缓存目录。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    )
    _MAX_FILESIZE_BYTES = 95 * 1024 * 1024
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
