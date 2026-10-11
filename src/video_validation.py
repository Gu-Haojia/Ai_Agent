"""校验下载视频的完整性，并判断是否需要转换为 QQ 兼容格式。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class VideoMetadata:
    """保存已通过校验的视频信息。

    Args:
        duration_seconds (float): 容器时长，单位为秒。
        width (int): 视频流宽度。
        height (int): 视频流高度。
        video_codec (str): 视频编码名称。
        pixel_format (str): 像素格式。
        requires_transcode (bool): 是否需要兼容性转码。

    Returns:
        None: 数据类初始化不返回值。

    Raises:
        None: 参数由 VideoValidator 校验。
    """

    duration_seconds: float
    width: int
    height: int
    video_codec: str
    pixel_format: str
    requires_transcode: bool


class VideoValidator:
    """通过元数据检查和全片解码拦截损坏或不完整的视频。

    Returns:
        None: 此类不维护外部状态。

    Raises:
        None: 构造时不执行校验。
    """

    def validate(
        self, video_path: Path, expected_duration: float | None = None
    ) -> VideoMetadata:
        """校验文件并返回发送前所需的视频信息。

        Args:
            video_path (Path): 下载或转码后的视频文件。
            expected_duration (float | None): 源站或转码前的时长；未知时为 None。

        Returns:
            VideoMetadata: 通过解析与全片解码校验的视频信息。

        Raises:
            AssertionError: 当文件为空、解析报错、时长不符或解码失败时抛出。
            OSError: 当无法执行 ffmpeg 或 ffprobe 时抛出。
            subprocess.TimeoutExpired: 当探测或全片解码超时时抛出。
            ValueError: 当探测结果不是合法 JSON 或数值时抛出。
        """
        path = video_path.expanduser().resolve()
        assert path.is_file() and path.stat().st_size > 0, "视频文件不存在或为空"
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_streams", "-show_format",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=False, timeout=60,
        )
        self._assert_success(probe, "视频解析")
        metadata = self._parse_metadata(path, probe.stdout, expected_duration)
        decoded = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
                "-xerror", "-err_detect", "explode", "-i", str(path),
                "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-",
            ],
            capture_output=True, text=True, check=False, timeout=600,
        )
        self._assert_success(decoded, "视频完整解码")
        return metadata

    @staticmethod
    def _assert_success(result: subprocess.CompletedProcess[str], stage: str) -> None:
        """同时检查退出码和 error 级别输出，拒绝伪成功。

        Args:
            result (subprocess.CompletedProcess[str]): 使用 error 日志级别的执行结果。
            stage (str): 用于错误提示的处理阶段。

        Returns:
            None: 无错误时返回。

        Raises:
            AssertionError: 当进程失败或 stderr 包含错误时抛出。
        """
        error = result.stderr.strip()
        assert result.returncode == 0 and not error, (
            f"{stage}失败，拒绝发送: {error[:2000] or f'退出码 {result.returncode}'}"
        )

    def _parse_metadata(
        self, path: Path, probe_json: str, expected_duration: float | None
    ) -> VideoMetadata:
        """解析必需字段，并检查时长与编码兼容性。

        Args:
            path (Path): 视频路径，用于检查容器扩展名。
            probe_json (str): ffprobe 的 JSON 输出。
            expected_duration (float | None): 期望时长。

        Returns:
            VideoMetadata: 已检查的视频元数据。

        Raises:
            AssertionError: 当缺少视频流、时长或尺寸无效时抛出。
            ValueError: 当 JSON 或数值非法时抛出。
            KeyError: 当探测结果缺少必需字段时抛出。
        """
        info = json.loads(probe_json)
        streams = info["streams"]
        assert isinstance(streams, list), "视频流信息结构异常"
        videos = [stream for stream in streams if stream["codec_type"] == "video"]
        assert videos, "下载文件没有视频流"
        video = videos[0]
        duration = float(info["format"]["duration"])
        self._assert_duration(duration, expected_duration)
        if "duration" in video:
            self._assert_duration(float(video["duration"]), expected_duration)
        width, height = int(video["width"]), int(video["height"])
        assert width > 0 and height > 0, "视频宽高必须大于 0"
        codec, pixel_format = str(video["codec_name"]), str(video["pix_fmt"])
        assert codec and pixel_format, "视频编码或像素格式为空"
        audio_codecs = [
            stream["codec_name"] for stream in streams if stream["codec_type"] == "audio"
        ]
        rotations = [
            float(side["rotation"]) for side in video.get("side_data_list", [])
            if "rotation" in side
        ]
        requires_transcode = (
            path.suffix.lower() != ".mp4"
            or codec != "h264"
            or pixel_format != "yuv420p"
            or any(codec != "aac" for codec in audio_codecs)
            or any(rotation % 360 != 0 for rotation in rotations)
            or video.get("sample_aspect_ratio") not in {None, "1:1", "0:1"}
        )
        return VideoMetadata(duration, width, height, codec, pixel_format, requires_transcode)

    @staticmethod
    def _assert_duration(duration: float, expected: float | None) -> None:
        """拒绝无效时长和明显截断的音视频结果。

        Args:
            duration (float): 探测到的容器或视频流时长。
            expected (float | None): 已知的期望时长。

        Returns:
            None: 时长符合要求时返回。

        Raises:
            AssertionError: 当时长非法或误差超过两秒及百分之二的较大值时抛出。
        """
        assert math.isfinite(duration) and duration > 0, "视频时长必须为有限正数"
        if expected is not None:
            assert math.isfinite(expected) and expected > 0, "源视频时长必须为有限正数"
            tolerance = max(2.0, expected * 0.02)
            assert abs(duration - expected) <= tolerance, (
                f"视频时长不符，拒绝发送: 实际 {duration:.3f} 秒，预期 {expected:.3f} 秒"
            )
