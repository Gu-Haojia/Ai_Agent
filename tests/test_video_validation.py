"""覆盖损坏 MP4、完整解码、截断视频和 QQ 兼容性转换。"""

import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from src.video_validation import VideoValidator
from src.yt_dlp_downloader import YtDlpVideoDownloader


class VideoValidatorTest(unittest.TestCase):
    """在进程边界模拟 FFmpeg 的成功与失败结果。"""

    def setUp(self) -> None:
        """准备文件与合法的探测响应。

        Returns:
            None: 初始化每个测试使用的独立目录。

        Raises:
            OSError: 当临时文件创建失败时抛出。
        """
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "video.mp4"
        self.path.write_bytes(b"fixture")
        self.info = {
            "format": {"duration": "141.067"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p",
                 "width": 1920, "height": 960, "duration": "141.067",
                 "sample_aspect_ratio": "1:1"},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
        }

    def test_decode_errors_are_rejected_even_with_zero_exit(self) -> None:
        """元数据正常时仍须拦截全片解码中发现的错误。

        Returns:
            None: 验证非零退出码及零退出码错误输出均被拒绝。

        Raises:
            AssertionError: 当损坏文件通过校验时抛出。
        """
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), mock.patch(
                "src.video_validation.subprocess.run",
                side_effect=[
                    subprocess.CompletedProcess([], 0, json.dumps(self.info), ""),
                    subprocess.CompletedProcess([], returncode, "", "Invalid data found"),
                ],
            ):
                with self.assertRaisesRegex(AssertionError, "完整解码失败"):
                    VideoValidator().validate(self.path, 141.067)

    def test_video_track_truncation_is_rejected_despite_complete_audio(self) -> None:
        """容器时长完整也不能掩盖只有 55 秒的视频轨。

        Returns:
            None: 验证校验在解码前发现轨道截断。

        Raises:
            AssertionError: 当截断视频未被拒绝时抛出。
        """
        self.info["streams"][0]["duration"] = "55"
        with mock.patch("src.video_validation.subprocess.run", return_value=
                        subprocess.CompletedProcess([], 0, json.dumps(self.info), "")):
            with self.assertRaisesRegex(AssertionError, "视频时长不符"):
                VideoValidator().validate(self.path, 141.067)

    def test_incompatible_codecs_require_conversion(self) -> None:
        """识别 AV1、十位 H.264 和非 AAC 音轨。

        Returns:
            None: 验证完整视频的兼容性分类。

        Raises:
            AssertionError: 当不兼容格式被判为可直接发送时抛出。
        """
        for codec, pixels, audio in (
            ("h264", "yuv420p", "aac"), ("av1", "yuv420p", "aac"),
            ("h264", "yuv420p10le", "aac"), ("h264", "yuv420p", "opus"),
        ):
            self.info["streams"][0].update(codec_name=codec, pix_fmt=pixels)
            self.info["streams"][1]["codec_name"] = audio
            with self.subTest(codec=codec, pixels=pixels, audio=audio), mock.patch(
                "src.video_validation.subprocess.run", side_effect=[
                    subprocess.CompletedProcess([], 0, json.dumps(self.info), ""),
                    subprocess.CompletedProcess([], 0, "", ""),
                ],
            ):
                metadata = VideoValidator().validate(self.path, 141.067)
                self.assertEqual(metadata.requires_transcode,
                                 (codec, pixels, audio) != ("h264", "yuv420p", "aac"))

    def test_failed_output_validation_preserves_input(self) -> None:
        """转码产物校验失败时保留原始下载文件。

        Returns:
            None: 验证转码产物未被作为可发送文件返回。

        Raises:
            AssertionError: 当损坏产物被放行或原件丢失时抛出。
        """
        self.info["streams"][0]["codec_name"] = "av1"
        with mock.patch("src.video_validation.subprocess.run", side_effect=[
            subprocess.CompletedProcess([], 0, json.dumps(self.info), ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]):
            metadata = VideoValidator().validate(self.path)
        downloader = YtDlpVideoDownloader(self.path.parent)
        output = self.path.with_name("video_compressed.mp4")

        def write_output(*args: object) -> None:
            """模拟成功退出但产物损坏的转码。

            Args:
                args (object): 接受原转码方法参数。

            Returns:
                None: 写出诊断用产物。

            Raises:
                OSError: 当写入失败时抛出。
            """
            output.write_bytes(b"bad-output")

        with (
            mock.patch.object(downloader, "_run_ffmpeg_compression", side_effect=write_output),
            mock.patch.object(downloader._validator, "validate", side_effect=
                              AssertionError("视频完整解码失败")),
        ):
            with self.assertRaisesRegex(AssertionError, "完整解码失败"):
                downloader._compress_if_needed(self.path, metadata)
        self.assertEqual(self.path.read_bytes(), b"fixture")
        self.assertEqual(output.read_bytes(), b"bad-output")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "需要 FFmpeg")
class RealVideoValidationTest(unittest.TestCase):
    """用真实媒体文件校验索引损坏和格式转换。"""

    def setUp(self) -> None:
        """生成含音轨的一秒 H.264 测试视频。

        Returns:
            None: 创建独立的合成媒体素材。

        Raises:
            subprocess.CalledProcessError: 当生成失败时抛出。
        """
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.path = self.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=64x32:rate=6:duration=1", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(self.path),
        ], capture_output=True, check=True)

    def test_real_corrupt_sample_table_is_rejected(self) -> None:
        """重现此次错误码被写进 MP4 帧大小表的产物。

        Returns:
            None: 验证真实 ffprobe 报错被拦截。

        Raises:
            AssertionError: 当损坏索引通过校验时抛出。
        """
        content = bytearray(self.path.read_bytes())
        table = content.index(b"stsz")
        sample_size, sample_count = struct.unpack_from(">II", content, table + 8)
        self.assertEqual(sample_size, 0)
        self.assertGreater(sample_count, 0)
        struct.pack_into(">I", content, table + 16, 3199971767)
        broken = self.root / "invalid_sample_size.mp4"
        broken.write_bytes(content)
        with self.assertRaisesRegex(AssertionError, "3199971767"):
            VideoValidator().validate(broken)

    def test_small_compatible_video_is_preserved(self) -> None:
        """完整且兼容的小视频保持原有字节内容。

        Returns:
            None: 验证无需重新压制的文件直接通过。

        Raises:
            AssertionError: 当文件被无故改写时抛出。
        """
        original = self.path.read_bytes()
        metadata = VideoValidator().validate(self.path, 1.0)
        result = YtDlpVideoDownloader(self.root)._compress_if_needed(self.path, metadata)
        self.assertEqual(result, self.path)
        self.assertEqual(result.read_bytes(), original)

    def test_small_ten_bit_rotated_video_is_normalized(self) -> None:
        """小文件也须统一位深、旋转和像素比例，保留显示比例。

        Returns:
            None: 验证实际输出为兼容编码且宽高反映显示方向。

        Raises:
            AssertionError: 当转换后格式或显示比例不符时抛出。
            subprocess.CalledProcessError: 当合成变体失败时抛出。
        """
        ten_bit = self.root / "ten_bit.mp4"
        rotated = self.root / "rotated.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(self.path), "-vf", "setsar=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p10le", "-c:a", "copy", str(ten_bit),
        ], capture_output=True, check=True)
        subprocess.run([
            "ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(ten_bit),
            "-c", "copy", str(rotated),
        ], capture_output=True, check=True)
        metadata = VideoValidator().validate(rotated, 1.0)
        self.assertTrue(metadata.requires_transcode)
        result = YtDlpVideoDownloader(self.root)._compress_if_needed(rotated, metadata)
        validated = VideoValidator().validate(result, 1.0)
        self.assertFalse(validated.requires_transcode)
        self.assertEqual(validated.video_codec, "h264")
        self.assertEqual(validated.pixel_format, "yuv420p")
        self.assertAlmostEqual(validated.width / validated.height, 0.25)
